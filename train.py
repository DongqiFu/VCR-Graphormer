from data import get_dataset, get_large_dataset, get_hetero_dataset, label_cluster

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import TransformerModel
from lr import PolynomialDecayLR
import torch.utils.data as Data
import argparse


def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='pubmed', help='name of dataset')
    parser.add_argument('--device', type=int, default=1, help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
    
    parser.add_argument('--adj_renorm', action='store_true', help='Whether needs re-normalize input adjacency.')
    parser.add_argument('--num_structure_supernodes', type=int, default=30, help='number of supernodes going to be added to the input graph')
    
    # model parameters
    parser.add_argument('--hops', type=int, default=8, help='Hop of neighbors to be calculated')
    parser.add_argument('--num_structure_tokens', type=int, default=8, help='number of structure-aware virtually coonected neighbors')
    parser.add_argument('--num_content_tokens', type=int, default=8, help='number of content-aware virtually coonected neighbors')
    
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64, help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Dropout in the attention layer')
    parser.add_argument('--split_seed', type=int, default=42, help='split seed')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000, help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400, help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')

    return parser.parse_args()

args = parse_args()
print(args)

all_time = time.time()

device = args.device

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.dataset in ['pubmed', 'photo', 'computer', 'cs', 'corafull', 'physics']:
    raw_adj_sp, original_adj, features, labels, idx_train, idx_val, idx_test, cluster_dict = get_dataset(args.dataset, args.num_structure_supernodes, args.split_seed)
    num_labels = len(set(labels.tolist()))
    labelcluster_to_nodes = {value: np.array([index for index, element in enumerate(labels.tolist()) if element == value and index not in idx_test]) for value in set(labels.tolist())}
    labels = labels.to(device)
elif args.dataset in ['squirrel', 'actor', 'texas']:
    raw_adj_sp, original_adj, features, labels, idx_train, idx_val, idx_test, cluster_dict = get_hetero_dataset(args.dataset, args.num_structure_supernodes, args.split_seed)
    num_labels = len(set(labels.tolist()))
    labelcluster_to_nodes = {value: np.array([index for index, element in enumerate(labels.tolist()) if element == value and index not in idx_test]) for value in set(labels.tolist())}
    labels = labels.to(device)
else:
    raw_adj_sp, original_adj, features, labels, idx_train, idx_val, idx_test, _, cluster_dict = get_large_dataset(dataset_str=args.dataset, split_seed=args.split_seed, renormalize=args.adj_renorm, num_supernodes=args.num_structure_supernodes)
    num_labels, labelcluster_to_nodes = label_cluster(labels, idx_test)  # labels is a ndarray matrix, n by c
    labels = np.argmax(labels, axis=1)  # labels is a list, n by 1
    labels = torch.from_numpy(labels).to(device)

adj, features = utils.add_multiple_supernodes(original_adj, features, cluster_dict, args.num_structure_supernodes)

adj = adj.to(device)
features = features.to(device)

processed_features = utils.re_features_push(raw_adj_sp, adj, features, args.hops, args.num_structure_tokens)  # return (N, num_structure_tokens + 1, d)
processed_features = utils.re_features_push_content(original_adj, processed_features, features, args.num_content_tokens, num_labels, labelcluster_to_nodes) # return (N, num_content_tokens + num_structure_tokens + 1, d)


batch_data_train = Data.TensorDataset(processed_features[idx_train], labels[idx_train])
batch_data_val = Data.TensorDataset(processed_features[idx_val], labels[idx_val])
batch_data_test = Data.TensorDataset(processed_features[idx_test], labels[idx_test])


train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle = True)
val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle = True)
test_data_loader = Data.DataLoader(batch_data_test, batch_size=int(args.batch_size/20), shuffle = True)


# model configuration
model = TransformerModel(token_length=args.hops + args.num_structure_tokens + args.num_content_tokens, 
                        n_class=labels.max().item() + 1, 
                        input_dim=features.shape[1] + 1, 
                        n_layers=args.n_layers,
                        num_heads=args.n_heads,
                        hidden_dim=args.hidden_dim,
                        ffn_dim=args.ffn_dim,
                        dropout_rate=args.dropout,
                        attention_dropout_rate=args.attention_dropout).to(device)

print('total params:', sum(p.numel() for p in model.parameters()))


optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
lr_scheduler = PolynomialDecayLR(
                optimizer,
                warmup_updates=args.warmup_updates,
                tot_updates=args.tot_updates,
                lr=args.peak_lr,
                end_lr=args.end_lr,
                power=1.0,
            )


def train_valid_epoch(epoch):
    
    model.train()
    loss_train_b = 0
    acc_train_b = 0
    for _, item in enumerate(train_data_loader):
        
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        optimizer.zero_grad()
        output = model(nodes_features)
        loss_train = F.nll_loss(output, labels)
        loss_train.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_train_b += loss_train.item()
        acc_train = utils.accuracy_batch(output, labels)
        acc_train_b += acc_train.item()
        
    
    model.eval()
    loss_val = 0
    acc_val = 0
    for _, item in enumerate(val_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)



        output = model(nodes_features)
        loss_val += F.nll_loss(output, labels).item()
        acc_val += utils.accuracy_batch(output, labels).item()
        

    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train_b),
        'acc_train: {:.4f}'.format(acc_train_b/len(idx_train)),
        'loss_val: {:.4f}'.format(loss_val),
        'acc_val: {:.4f}'.format(acc_val/len(idx_val)))

    return loss_val, acc_val

def test():

    loss_test = 0
    acc_test = 0
    for _, item in enumerate(test_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)


        model.eval()

        output = model(nodes_features)
        loss_test += F.nll_loss(output, labels).item()
        acc_test += utils.accuracy_batch(output, labels).item()

    print("Test set results:",
        "loss= {:.4f}".format(loss_test),
        "accuracy= {:.4f}".format(acc_test/len(idx_test)))



train_time = time.time()
stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)
for epoch in range(args.epochs):
    loss_val, acc_val = train_valid_epoch(epoch)
    if early_stopping.check([acc_val, loss_val], epoch):
        break

print("Optimization Finished!")
print("Train cost: {:.4f}s".format(time.time() - train_time))
# Restore best model
print('Loading {}th epoch'.format(early_stopping.best_epoch+1))
model.load_state_dict(early_stopping.best_state)

test()

print("Total time cost {}s".format(time.time()-all_time))