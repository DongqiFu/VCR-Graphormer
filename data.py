import time
import dgl
import torch
import scipy.sparse as sp
import os.path
from dgl.data import PubmedGraphDataset, CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import numpy as np
from dgl.data.utils import split_dataset

from dataset_hetero import load_nc_dataset
from data_utils_hetero import load_fixed_splits

def get_dataset(dataset_name, num_supernodes, split_seed):
    
    if dataset_name == "pubmed":
        dataset = PubmedGraphDataset()
    elif dataset_name == "corafull":
        dataset = CoraFullDataset()
    elif dataset_name == "computer":
        dataset = AmazonCoBuyComputerDataset()
    elif dataset_name == "photo":
        dataset = AmazonCoBuyPhotoDataset()
    elif dataset_name == "cs":
        dataset = CoauthorCSDataset()
    elif dataset_name == "physics":
        dataset = CoauthorPhysicsDataset()
    elif dataset_name == "cora":
        dataset = CoraGraphDataset()
    elif dataset_name == "citeseer":
        dataset = CiteseerGraphDataset()

    graph = dataset[0]
    
    adj = graph.adj()
    features = graph.ndata["feat"]
    labels = graph.ndata["label"]
    
    idx_train, idx_val, idx_test = split_dataset(range(len(labels)), frac_list = [0.6, 0.2, 0.2], shuffle=True, random_state=split_seed)

    graph = dgl.to_bidirected(graph)

    clusters_dict = dgl.metis_partition(g=graph, k=num_supernodes, extra_cached_hops=0, reshuffle=False, balance_ntypes=None,
                                        balance_edges=False, mode='k-way')
    
    features = features.double()
    
    # adj from sparse tensor to sparse matrix 
    raw_adj_sp = sp.coo_matrix((adj.coalesce().values(), (adj.coalesce().indices()[0], adj.coalesce().indices()[1])), shape=adj.shape) 

    return raw_adj_sp, adj, features, labels, idx_train, idx_val, idx_test, clusters_dict

# for Squirrel, Actor, Texas
def get_hetero_dataset(dataset_name, num_supernodes, split_seed):

    file_path = "dataset/" + dataset_name + ".pt"

    data_list = torch.load(file_path)

    adj_tensor = data_list[0]
    features = data_list[1]
    labels = data_list[2]
    
    adj_matrix = sp.coo_matrix(adj_tensor)
    graph = dgl.from_scipy(adj_matrix)

    adj = graph.adj()

    idx_train, idx_val, idx_test = split_dataset(range(len(labels)), frac_list=[0.6, 0.2, 0.2], shuffle=True, random_state=split_seed)

    graph = dgl.to_bidirected(graph)

    clusters_dict = dgl.metis_partition(g=graph, k=num_supernodes, extra_cached_hops=0, reshuffle=False,
                                        balance_ntypes=None,
                                        balance_edges=False, mode='k-way')

    features = features.double()

    # adj from sparse tensor to sparse matrix
    raw_adj_sp = sp.coo_matrix((adj.coalesce().values(), (adj.coalesce().indices()[0], adj.coalesce().indices()[1])),
                               shape=adj.shape)

    return raw_adj_sp, adj, features, labels, idx_train, idx_val, idx_test, clusters_dict


# for Reddit, Aminer, Amazon2M
def get_large_dataset(dataset_str, split_seed, renormalize, num_supernodes):
    """Load data."""
    if os.path.exists("dataset/{}".format(dataset_str)):
        path = "dataset/{}".format(dataset_str)
    else:
        path = "dataset/"
    if dataset_str == 'aminer':
        adj = pkl.load(open(os.path.join(path, "{}.adj.sp.pkl".format(dataset_str)), "rb"))
        features = pkl.load(
            open(os.path.join(path, "{}.features.pkl".format(dataset_str)), "rb"))
        labels = pkl.load(
            open(os.path.join(path, "{}.labels.pkl".format(dataset_str)), "rb"))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
        features = col_normalize(features)

    elif dataset_str in ['reddit']:
        adj = sp.load_npz(os.path.join(path, '{}_adj.npz'.format(dataset_str)))
        features = np.load(os.path.join(path, '{}_feat.npy'.format(dataset_str)))
        labels = np.load(os.path.join(path, '{}_labels.npy'.format(dataset_str)))
        # print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
        # print(dataset_str, features.shape)

    elif dataset_str in ['Amazon2M']:
        adj = sp.load_npz(os.path.join(path, '{}_adj.npz'.format(dataset_str)))
        features = np.load(os.path.join(path, '{}_feat.npy'.format(dataset_str)))
        labels = np.load(os.path.join(path, '{}_labels.npy'.format(dataset_str)))
        # print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        class_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20 * class_num,
                                                                val_size=30 * class_num)
        idx_unlabel = np.concatenate((idx_val, idx_test))

    else:
        raise NotImplementedError

    raw_adj_sp = adj
    
    if renormalize:
        adj = adj + sp.eye(adj.shape[0])
        D1 = np.array(adj.sum(axis=1)) ** (-0.5)
        D2 = np.array(adj.sum(axis=0)) ** (-0.5)
        D1 = sp.diags(D1[:, 0], format='csr')
        D2 = sp.diags(D2[0, :], format='csr')

        A = adj.dot(D1)
        A = D2.dot(A)
        adj = A

    graph = dgl.from_scipy(adj)
    
    start_time = time.time()
    clusters_dict = dgl.metis_partition(g=graph, k=num_supernodes, extra_cached_hops=0, reshuffle=False,
                                            balance_ntypes=None, balance_edges=False, mode='k-way')
    print('Metis paritioning costs {}s'.format(time.time() - start_time))
    
    features = torch.from_numpy(features) 
    
    # sparse matrix to tensor    
    adj = adj.tocoo()
    
    values = adj.data
    indices = np.vstack((adj.row, adj.col))

    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = adj.shape

    adj = torch.sparse.DoubleTensor(i, v, torch.Size(shape))

    return raw_adj_sp, adj, features, labels, idx_train, idx_val, idx_test, idx_unlabel, clusters_dict


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    # print(len(set(train_indices)), len(train_indices))
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


# for arxiv-year
def load_large_hetero_dataset(dataset_name, sub_dataset_name, num_supernodes, split_idx):
    
    ### Load and preprocess data ###
    dataset = load_nc_dataset(dataset_name, sub_dataset_name)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    num_nodes = int(dataset.graph['edge_index'].max()) + 1
    adj_tensor = torch.sparse_coo_tensor(dataset.graph['edge_index'], torch.ones(dataset.graph['edge_index'].shape[1]), (num_nodes, num_nodes))
    features = dataset.graph['node_feat']
    labels = dataset.label

    # adj_matrix = sp.coo_matrix(adj_tensor)
    adj_tensor = adj_tensor.coalesce()
    values = adj_tensor.values().numpy()
    rows = adj_tensor.indices()[0].numpy()
    cols = adj_tensor.indices()[1].numpy()
    adj_matrix = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    
    graph = dgl.from_scipy(adj_matrix)

    adj = graph.adj()
    
    split_idx_lst = load_fixed_splits(dataset_name, sub_dataset_name)
    idx_train, idx_val, idx_test = split_idx_lst[split_idx]['train'], split_idx_lst[split_idx]['valid'], split_idx_lst[split_idx]['test']

    graph = dgl.to_bidirected(graph)

    clusters_dict = dgl.metis_partition(g=graph, k=num_supernodes, extra_cached_hops=0, reshuffle=False,
                                        balance_ntypes=None,
                                        balance_edges=False, mode='k-way')

    features = features.double()

    # adj from sparse tensor to sparse matrix
    raw_adj_sp = sp.coo_matrix((adj.coalesce().values(), (adj.coalesce().indices()[0], adj.coalesce().indices()[1])),
                               shape=adj.shape)

    return raw_adj_sp, adj, features, labels, idx_train, idx_val, idx_test, clusters_dict


def label_cluster(label_matrix, idx_test):
    num_labels = label_matrix.shape[1]
    
    cluster2ids ={}
    
    for i in range(num_labels):
        non_zero_indices = np.nonzero(label_matrix[:, i])[0]
        exc_test = set(non_zero_indices).difference(set(idx_test.tolist()))
        cluster2ids[i] = np.array(list(exc_test))
        
    return num_labels, cluster2ids
        
    
def col_normalize(mx):
    """Column-normalize sparse matrix"""
    scaler = StandardScaler()

    mx = scaler.fit_transform(mx)

    return mx


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])