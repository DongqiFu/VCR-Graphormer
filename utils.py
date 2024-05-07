import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import dgl
import random
from typing import List, Union, Tuple
import numba
from scipy.sparse import csr_matrix, coo_matrix
from torch_geometric.utils import coalesce

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix

def nor_matrix(adj, a_matrix):
    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix

def add_multiple_supernodes(adj, features, cluster_dict, num_supernodes):
    # Coalesce the sparse tensor
    n = adj.size(0)
    
    num_members_in_a_cluster = cluster_dict[0].ndata['_ID'].size(0)
    padded_indices = torch.cat([torch.tensor([cluster_dict[0].ndata['_ID'].tolist(), [n]*num_members_in_a_cluster]),
                                torch.tensor([[n]*num_members_in_a_cluster, cluster_dict[0].ndata['_ID'].tolist()])
                                ], dim=1)
    padded_values = torch.cat([torch.ones(num_members_in_a_cluster), torch.ones(num_members_in_a_cluster)])

    # add the connections of each supernode
    for i in range(1, num_supernodes):
        
        num_members_in_a_cluster = cluster_dict[i].ndata['_ID'].size(0)
        padded_indices = torch.cat([padded_indices,
                                    torch.tensor([cluster_dict[i].ndata['_ID'].tolist(), [n+i]*num_members_in_a_cluster]),
                                    torch.tensor([[n+i]*num_members_in_a_cluster, cluster_dict[i].ndata['_ID'].tolist()])
                                    ], dim=1)
        padded_values = torch.cat([padded_values, torch.ones(num_members_in_a_cluster), torch.ones(num_members_in_a_cluster)])
    
    padded_indices = torch.cat([adj.coalesce().indices(), padded_indices], dim=1)
    padded_values = torch.cat([adj.coalesce().values(), padded_values])
    
    adj = torch.sparse_coo_tensor(indices=padded_indices, values=padded_values, size=(n + num_supernodes, n + num_supernodes))
    features = torch.nn.functional.pad(features.to_dense(), (0, 0, 0, num_supernodes), value=0)

    return adj, features


def re_features(raw_adj_sp, adj, features, hops):
    nodes_features = torch.empty(features.shape[0], hops+1, features.shape[1]+1)
    
    raw_num_nodes = raw_adj_sp.shape[0]

    # renomalize original adj
    raw_adj_sp = raw_adj_sp + sp.eye(raw_adj_sp.shape[0])
    D1 = np.array(raw_adj_sp.sum(axis=1)) ** (-0.5)
    D2 = np.array(raw_adj_sp.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')
    A = raw_adj_sp.dot(D1)
    A = D2.dot(A)
    raw_adj_sp = A
    # sp to tensor, expand the dimension of raw adj    
    raw_adj_sp = raw_adj_sp.tocoo()
    values = raw_adj_sp.data
    indices = np.vstack((raw_adj_sp.row, raw_adj_sp.col))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = adj.shape
    raw_adj_sp = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
    raw_adj_sp = raw_adj_sp.to(features.device)
    
    x = features
    weight = np.sum(range(1, hops+1))
    for i in range(hops):
        # print('hop: ' + str(i))
        x = torch.matmul(raw_adj_sp, x)
        for j in range(raw_num_nodes):
            nodes_features[j, i+1, :] = torch.cat((x[j], torch.tensor([(hops-i)/weight]).to(features.device)))

    return nodes_features


def re_features_push(raw_adj_sp, adj, features, hops, K):
    print('Running PPR (structure-based, virtual connection) ...')
    _, top_k_of_node, neighbors_weights_of_node = topk_ppr_matrix(edge_index=adj.coalesce().indices(), num_nodes=adj.shape[0], alpha=0.85, eps=1e-7, output_node_indices = np.arange(raw_adj_sp.shape[0]), topk=K, max_itr=10000, normalization='row')
    nodes_features = torch.empty(features.shape[0], K+1+hops, features.shape[1]+1)
    
    raw_num_nodes = raw_adj_sp.shape[0]

    # renomalize original adj
    raw_adj_sp = raw_adj_sp + sp.eye(raw_adj_sp.shape[0])
    D1 = np.array(raw_adj_sp.sum(axis=1)) ** (-0.5)
    D2 = np.array(raw_adj_sp.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')
    A = raw_adj_sp.dot(D1)
    A = D2.dot(A)
    raw_adj_sp = A
    # sp to tensor, expand the dimension of raw adj    
    raw_adj_sp = raw_adj_sp.tocoo()
    values = raw_adj_sp.data
    indices = np.vstack((raw_adj_sp.row, raw_adj_sp.col))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = adj.shape
    raw_adj_sp = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
    raw_adj_sp = raw_adj_sp.to(features.device)
    
    x = features
    weight = np.sum(range(1, hops+1))
    for i in range(hops):
        # print('hop: ' + str(i))
        x = torch.matmul(raw_adj_sp, x)
        for j in range(raw_num_nodes):
            nodes_features[j, i+1, :] = torch.cat((x[j], torch.tensor([(hops-i)/weight]).to(features.device)))
    
    print('PPR samples Top K for each node (structure-based, virtual connection) ...')
    progress_bar = tqdm(total=raw_num_nodes)
    
    for i in range(raw_num_nodes):
        nodes_features[i, 0, :] = torch.cat((features[i], torch.tensor([1]).to(features.device)))
        
        topk_indices = list(top_k_of_node[i])
        if len(topk_indices) < K:
            random_neighbors = list(random.sample(range(0, shape[0]), K-len(topk_indices)))
            topk_indices = topk_indices + random_neighbors
            
        topk_neighbor_weights = list(neighbors_weights_of_node[i])
        if len(topk_neighbor_weights) < K:
            topk_neighbor_weights = topk_neighbor_weights + [0]*(K-len(topk_neighbor_weights))
        else:
            topk_neighbor_weights = topk_neighbor_weights[:K]
        topk_neighbor_weights = torch.tensor(topk_neighbor_weights).view(-1, 1).to(features.device)
        
        nodes_features[i, hops+1:, :] = torch.cat((features[topk_indices], topk_neighbor_weights), 1)
        progress_bar.update(1)
        
    progress_bar.close()

    return nodes_features


def re_features_push_content(original_adj, processed_features, features, K, num_labels, labelcluster_to_nodes):
    # adding supernodes to original adj, and get adj
    n = original_adj.size(0)
    
    # for a cluster, connect each ordinary node in a cluster to their super node
    num_members_in_a_cluster = len(labelcluster_to_nodes[0])
    padded_indices = torch.cat([torch.tensor([labelcluster_to_nodes[0].tolist(), [n]*num_members_in_a_cluster]),
                                torch.tensor([[n]*num_members_in_a_cluster, labelcluster_to_nodes[0].tolist()])
                                ], dim=1)
    padded_values = torch.cat([torch.ones(num_members_in_a_cluster), torch.ones(num_members_in_a_cluster)])

    for i in range(1, num_labels):
        
        num_members_in_a_cluster = len(labelcluster_to_nodes[i])
        padded_indices = torch.cat([padded_indices,
                                    torch.tensor([labelcluster_to_nodes[i].tolist(), [n+i]*num_members_in_a_cluster]),
                                    torch.tensor([[n+i]*num_members_in_a_cluster, labelcluster_to_nodes[i].tolist()])
                                    ], dim=1)
        padded_values = torch.cat([padded_values, torch.ones(num_members_in_a_cluster), torch.ones(num_members_in_a_cluster)])
    
    padded_indices = torch.cat([original_adj.coalesce().indices(), padded_indices], dim=1)
    padded_values = torch.cat([original_adj.coalesce().values(), padded_values])
    
    adj = torch.sparse_coo_tensor(indices=padded_indices, values=padded_values, size=(n + num_labels, n + num_labels))
    features = torch.nn.functional.pad(features.to_dense(), (0, 0, 0, num_labels), value=0)
    
    # run pagerank
    print('Running PPR (content-based, virtual connection) ...')
    _, top_k_of_node, neighbors_weights_of_node = topk_ppr_matrix(edge_index=adj.coalesce().indices(), num_nodes=adj.shape[0], alpha=0.85, eps=1e-2, output_node_indices = np.arange(original_adj.shape[0]), topk=K, max_itr=100, normalization='row')
    nodes_features = torch.empty(original_adj.shape[0], K, processed_features.shape[2])

    print('PPR samples Top K for each node (content-based, virtual connection) ...')
    progress_bar = tqdm(total=original_adj.shape[0])
    
    for i in range(original_adj.shape[0]):
        
        topk_indices = list(top_k_of_node[i])
        if len(topk_indices) < K:
            # modified_range = [num for num in range(0, original_adj.shape[0]) if num not in topk_indices]
            # random_neighbors = list(random.sample(modified_range, K-len(topk_indices)))
            random_neighbors = list(random.sample(range(0, original_adj.shape[0]), K-len(topk_indices)))
            topk_indices = topk_indices + random_neighbors
        
        topk_neighbor_weights = list(neighbors_weights_of_node[i]) # first k neighbors' weights
        if len(topk_neighbor_weights) < K:
            topk_neighbor_weights = topk_neighbor_weights + [0]*(K-len(topk_neighbor_weights))
        else:
            topk_neighbor_weights = topk_neighbor_weights[:K]
        topk_neighbor_weights = torch.tensor(topk_neighbor_weights).view(-1, 1).to(features.device)
        
        # nodes_features[i, :, :] = features[topk_indices]
        nodes_features[i, :, :] = torch.cat((features[topk_indices], topk_neighbor_weights), 1)
        
        progress_bar.update(1)
        
    progress_bar.close()
    
    nodes_features = torch.cat((processed_features[:original_adj.shape[0], :, :], nodes_features), 1)

    return nodes_features


def topk_ppr_matrix(edge_index: torch.Tensor,
                    num_nodes: int,
                    alpha: float,
                    eps: float,
                    output_node_indices: Union[np.ndarray, torch.LongTensor],
                    topk: int,
                    max_itr: int,
                    normalization='row') -> Tuple[csr_matrix, List[np.ndarray]]:
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""
    if isinstance(output_node_indices, torch.Tensor):
        output_node_indices = output_node_indices.numpy()

    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()

    _, indptr, out_degree = np.unique(edge_index_np[0],
                                      return_index=True,
                                      return_counts=True)
    indptr = np.append(indptr, len(edge_index_np[0]))

    neighbors, weights = calc_ppr_topk_parallel(indptr, edge_index_np[1], out_degree,
                                                alpha, eps, output_node_indices, max_itr)

    ppr_matrix = construct_sparse(neighbors, weights, (len(output_node_indices), num_nodes))
    ppr_matrix = ppr_matrix.tocsr()
    
    weights_of_all_neighbors = weights

    neighbors = sparsify(neighbors, weights, topk)
    neighbors = [np.union1d(nei, pr) for nei, pr in zip(neighbors, output_node_indices)]

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg_sqrt = np.sqrt(np.maximum(out_degree, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = deg_sqrt[output_node_indices[row]] * ppr_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg_inv = 1. / np.maximum(out_degree, 1e-12)

        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = out_degree[output_node_indices[row]] * ppr_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return ppr_matrix, neighbors, weights_of_all_neighbors


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return coo_matrix((np.concatenate(weights), (i, j)), shape)


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gdc.html#GDC.diffusion_matrix_approx
@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, max_itr) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        # print(i)
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon, max_itr)
        js[i] = np.array(j)
        vals[i] = np.array(val)
    return js, vals


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon, max_itr):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    
    time_to_stop = 0

    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
        
        time_to_stop += 1
        if time_to_stop >= max_itr:
            # print("Maximum time iteration reached of push operation for node " + str(inode))
            break
    # print('PPR stoped at ' + str(time_to_stop) + ' iteration for node ' + str(inode))
            
    return list(p.keys()), list(p.values())


def sparsify(neighbors: List[np.ndarray], weights: List[np.ndarray], topk: int):
    new_neighbors = []
    for n, w in zip(neighbors, weights):
        idx_topk = np.argsort(w)[-topk:]
        new_neighbor = n[idx_topk]
        new_neighbors.append(new_neighbor)

    return new_neighbors