import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
import torch
from time import perf_counter
import tabulate

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    index_dict = {}
    label_dict = {}
    phases = ["train", "val", "test"]
    objects = []
    def load_pkl(path):
        with open(path.format(dataset_str, p), 'rb') as f:
            if sys.version_info > (3, 0):
                return pkl.load(f, encoding='latin1')
            else:
                return pkl.load(f)

    for p in phases:
        index_dict[p] = load_pkl("data/ind.{}.{}.x".format(dataset_str, p))
        label_dict[p] = load_pkl("data/ind.{}.{}.y".format(dataset_str, p))

    adj = load_pkl("data/ind.{}.BCD.adj".format(dataset_str))
    adj = adj.astype(np.float32)
    adj = preprocess_adj(adj)

    return adj, index_dict, label_dict

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).transpose().tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    string = re.sub(r'[?|$|.|!]',r'',string)
    string = re.sub(r'[^a-zA-Z0-9 ]',r'',string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def sparse_to_torch_sparse(sparse_mx, device='cuda'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    if device == 'cuda':
        indices = indices.cuda()
        values = torch.from_numpy(sparse_mx.data).cuda()
        shape = torch.Size(sparse_mx.shape)
        adj = torch.cuda.sparse.FloatTensor(indices, values, shape)
    elif device == 'cpu':
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        adj = torch.sparse.FloatTensor(indices, values, shape)
    return adj

def sparse_to_torch_dense(sparse, device='cuda'):
    dense = sparse.todense().astype(np.float32)
    torch_dense = torch.from_numpy(dense).to(device=device)
    return torch_dense

def sgc_precompute(adj, features, degree, index_dict):   # feature = adj_dense ([53210, 53210])
    #assert degree==1, "Only supporting degree 2 now"
    feat_dict = {}
    start = perf_counter()
    train_feats = features[:, index_dict["train"]].cuda()
    train_feats = torch.spmm(adj, train_feats).t()                  #t.() 求转置   train_feats: ([53210, 10183])   adj:([53210, 53210])
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)       #特征标准化
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    train_feats = (train_feats-train_feats_min)/train_feats_range            # train_feats: ([10183, 53210])

    feat_dict["train"] = train_feats
    for phase in ["test", "val"]:
        feats = features[:, index_dict[phase]].cuda()                    #test:([53210, 7532])     val: ([53210, 1131])
        
        feats = torch.spmm(adj, feats).t()                               #test: ([7532, 53210])    val: ([1131, 53210])
        
        feats = feats[:, useful_features_dim]                            #test: ([7532, 53210])    val: ([1131, 53210])
        
        feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu() # adj is symmetric!          test.shape: ([7532, 53210] val.shape: ([1131, 53210])
      
    precompute_time = perf_counter()-start
    return feat_dict, precompute_time

# def sjk_precompute(features, adj, degree, alpha, mode):
#     t = perf_counter()
#     feat_lst = []
#     ori_features = features
#     emb = features
#     for i in range(degree):
#         emb = torch.spmm(adj, emb)
#         feat_lst.append(emb)
#     feat_lst.insert(0, ori_features)

#     if mode =='cat':
#         out = torch.cat(feat_lst, dim = -1) 
#         #print(out.shape)
#     elif mode =='max':     #([2708, 1433, 16]) -> (2,) ->  ([2708, 1433])
#         out = torch.stack(feat_lst, dim=-1).max(dim= -1)[0]
#     precompute_time = perf_counter()-t
    
#     return out, precompute_time

def sjk_precompute(adj, features, degree, index_dict, mode):
    #assert degree==1, "Only supporting degree 2 now"
    feat_dict = {}
    feat_train_list = []
    feat_test_list = []
    feat_val_list = []
    start = perf_counter()
    #print(features.shape)     ([53210, 53210])

    # train_feats = features[:, index_dict["train"]].cuda().t()       ([10183, 53210])
    # print(train_feats.shape)
    # train_feats_max, _ = train_feats.max(dim=0, keepdim=True)       #特征标准化
    # train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    # train_feats_range = train_feats_max-train_feats_min              #([1, 53210])
    # print(train_feats_range.shape)
    # useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze() #([43258])
    # useful_features_dim = train_feats_range.squeeze()
    # print(useful_features_dim.shape)
    # train_feats = train_feats[:, useful_features_dim]
    # train_feats_range = train_feats_range[:, useful_features_dim]
    # train_feats_min = train_feats_min[:, useful_features_dim]
    # ori_train_feats = (train_feats-train_feats_min)/train_feats_range
    # print(ori_train_feats.shape)
    
    # test_feats = features[:, index_dict["test"]].cuda().t()
    # test_feats = test_feats[:, useful_features_dim]
    # ori_test_feats = ((test_feats-train_feats_min)/train_feats_range).cpu()
    # print(ori_test_feats.shape)

    # val_feats = features[:, index_dict["val"]].cuda().t()
    # val_feats = val_feats[:, useful_features_dim]
    # ori_val_feats = ((val_feats-train_feats_min)/train_feats_range).cpu() 
    # print(ori_val_feats.shape)

    train_feats = features[:, index_dict["train"]].cuda()
    test_feats = features[:, index_dict["test"]].cuda()
    val_feats = features[:, index_dict["val"]].cuda()

    #feat_train_list.insert(0,train_feats.t().cpu())
    #feat_test_list.insert(0, test_feats.t().cpu())
    #feat_val_list.insert(0, val_feats.t().cpu)

    # train_feats = features[:, index_dict["train"]].cpu()
    # test_feats = features[:, index_dict["test"]].cpu()
    # val_feats = features[:, index_dict["val"]].cpu()

    train_feats = torch.spmm(adj, train_feats).t()
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    train_feats = (train_feats-train_feats_min)/train_feats_range
        
    feat_train_list.append(train_feats)
        #feat_dict["train"] = train_feats
    
        #for phase in ["test", "val"]:
        
    test_feats = torch.spmm(adj, test_feats).t()
    test_feats = test_feats[:, useful_features_dim]
    test_feats = ((test_feats-train_feats_min)/train_feats_range).cpu() 
        
    feat_test_list.append(test_feats)
        #feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu() # adj is symmetric!

       
    val_feats = torch.spmm(adj, val_feats).t()
    val_feats = val_feats[:, useful_features_dim]
    val_feats = ((val_feats-train_feats_min)/train_feats_range).cpu() 
        
    feat_val_list.append(val_feats)


    train_feats = features[:, index_dict["train"]].cuda()
    test_feats = features[:, index_dict["test"]].cuda()
    val_feats = features[:, index_dict["val"]].cuda()

    # train_feats = features[:, index_dict["train"]].cpu()
    # test_feats = features[:, index_dict["test"]].cpu()
    # val_feats = features[:, index_dict["val"]].cpu()

    
    train_feats = torch.spmm(adj, train_feats)
    train_feats = torch.spmm(adj, train_feats).t()
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    train_feats = (train_feats-train_feats_min)/train_feats_range
        
    feat_train_list.append(train_feats)
        #feat_dict["train"] = train_feats
    
        #for phase in ["test", "val"]:
    test_feats = torch.spmm(adj, test_feats)   
    test_feats = torch.spmm(adj, test_feats).t()
    test_feats = test_feats[:, useful_features_dim]
    test_feats = ((test_feats-train_feats_min)/train_feats_range).cpu() 
        
    feat_test_list.append(test_feats)
        #feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu() # adj is symmetric!

    val_feats = torch.spmm(adj, val_feats) 
    val_feats = torch.spmm(adj, val_feats).t()
    val_feats = val_feats[:, useful_features_dim]
    val_feats = ((val_feats-train_feats_min)/train_feats_range).cpu() 
    print(val_feats.shape)  
    feat_val_list.append(val_feats)

    # feat_train_list.insert(0, ori_train_feats)
    # feat_test_list.insert(0, ori_test_feats)
    # feat_val_list.insert(0, ori_val_feats)
    # print(np.shape(feat_train_list))
    # print(np.shape(feat_test_list))
    # print(np.shape(feat_val_list))

    if mode =='cat':
        out_train = torch.cat(feat_train_list, dim = -1)
        out_test = torch.cat(feat_test_list, dim = -1)
        out_val = torch.cat(feat_val_list, dim = -1)
        print(out_train.shape)
        print(out_test.shape)
        print(out_val.shape)

        #print(out.shape)
    elif mode =='max':     #([2708, 1433, 16]) -> (2,) ->  ([2708, 1433])
        out_train = torch.stack(feat_train_list, dim=-1).max(dim= -1)[0]
        out_test = torch.stack(feat_test_list, dim=-1).max(dim= -1)[0]
        out_val = torch.stack(feat_val_list, dim=-1).max(dim= -1)[0]

    feat_dict["train"] = out_train
    feat_dict["test"] = out_test
    feat_dict["val"] = out_val

    precompute_time = perf_counter()-start
    return feat_dict, precompute_time

def spr_precompute(adj, features, degree, index_dict, mode):
    #assert degree==1, "Only supporting degree 2 now"
    feat_dict = {}
    feat_train_list = []
    feat_test_list = []
    feat_val_list = []
    start = perf_counter()
    #print(features.shape)     ([53210, 53210])

    # train_feats = features[:, index_dict["train"]].cuda().t()       ([10183, 53210])
    # print(train_feats.shape)
    # train_feats_max, _ = train_feats.max(dim=0, keepdim=True)       #特征标准化
    # train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    # train_feats_range = train_feats_max-train_feats_min              #([1, 53210])
    # print(train_feats_range.shape)
    # useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze() #([43258])
    # useful_features_dim = train_feats_range.squeeze()
    # print(useful_features_dim.shape)
    # train_feats = train_feats[:, useful_features_dim]
    # train_feats_range = train_feats_range[:, useful_features_dim]
    # train_feats_min = train_feats_min[:, useful_features_dim]
    # ori_train_feats = (train_feats-train_feats_min)/train_feats_range
    # print(ori_train_feats.shape)
    
    # test_feats = features[:, index_dict["test"]].cuda().t()
    # test_feats = test_feats[:, useful_features_dim]
    # ori_test_feats = ((test_feats-train_feats_min)/train_feats_range).cpu()
    # print(ori_test_feats.shape)

    # val_feats = features[:, index_dict["val"]].cuda().t()
    # val_feats = val_feats[:, useful_features_dim]
    # ori_val_feats = ((val_feats-train_feats_min)/train_feats_range).cpu() 
    # print(ori_val_feats.shape)

    train_feats = features[:, index_dict["train"]].cuda()
    test_feats = features[:, index_dict["test"]].cuda()
    val_feats = features[:, index_dict["val"]].cuda()


    # train_feats = features[:, index_dict["train"]].cpu()
    # test_feats = features[:, index_dict["test"]].cpu()
    # val_feats = features[:, index_dict["val"]].cpu()

    train_feats = torch.spmm(adj, train_feats).t()
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    train_feats = (train_feats-train_feats_min)/train_feats_range
        
    feat_train_list.append(train_feats)
        #feat_dict["train"] = train_feats
    
        #for phase in ["test", "val"]:
        
    test_feats = torch.spmm(adj, test_feats).t()
    test_feats = test_feats[:, useful_features_dim]
    test_feats = ((test_feats-train_feats_min)/train_feats_range).cpu() 
        
    feat_test_list.append(test_feats)
        #feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu() # adj is symmetric!

       
    val_feats = torch.spmm(adj, val_feats).t()
    val_feats = val_feats[:, useful_features_dim]
    val_feats = ((val_feats-train_feats_min)/train_feats_range).cpu() 
        
    feat_val_list.append(val_feats)


    train_feats = features[:, index_dict["train"]].cuda()
    test_feats = features[:, index_dict["test"]].cuda()
    val_feats = features[:, index_dict["val"]].cuda()

    # train_feats = features[:, index_dict["train"]].cpu()
    # test_feats = features[:, index_dict["test"]].cpu()
    # val_feats = features[:, index_dict["val"]].cpu()

    
    train_feats_1 = torch.spmm(adj, train_feats)
    train_feats = torch.spmm(adj, train_feats_1)
    train_feats = (train_feats_1 + train_feats).t()
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    train_feats = (train_feats-train_feats_min)/train_feats_range
        
    feat_train_list.append(train_feats)
        #feat_dict["train"] = train_feats
    
        #for phase in ["test", "val"]:
    test_feats_1 = torch.spmm(adj, test_feats)   
    test_feats = torch.spmm(adj, test_feats_1)
    test_feats = (test_feats_1 + test_feats).t()
    test_feats = test_feats[:, useful_features_dim]
    test_feats = ((test_feats-train_feats_min)/train_feats_range).cpu() 
        
    feat_test_list.append(test_feats)
        #feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu() # adj is symmetric!

    val_feats_1 = torch.spmm(adj, val_feats) 
    val_feats = torch.spmm(adj, val_feats_1)
    val_feats = (val_feats_1 + val_feats).t()
    val_feats = val_feats[:, useful_features_dim]
    val_feats = ((val_feats-train_feats_min)/train_feats_range).cpu() 
    print(val_feats.shape)  
    feat_val_list.append(val_feats)

    # feat_train_list.insert(0, ori_train_feats)
    # feat_test_list.insert(0, ori_test_feats)
    # feat_val_list.insert(0, ori_val_feats)
    # print(np.shape(feat_train_list))
    # print(np.shape(feat_test_list))
    # print(np.shape(feat_val_list))

    if mode =='cat':
        out_train = torch.cat(feat_train_list, dim = -1)
        out_test = torch.cat(feat_test_list, dim = -1)
        out_val = torch.cat(feat_val_list, dim = -1)
        print(out_train.shape)
        print(out_test.shape)
        print(out_val.shape)

        #print(out.shape)
    elif mode =='max':     #([2708, 1433, 16]) -> (2,) ->  ([2708, 1433])
        out_train = torch.stack(feat_train_list, dim=-1).max(dim= -1)[0]
        out_test = torch.stack(feat_test_list, dim=-1).max(dim= -1)[0]
        out_val = torch.stack(feat_val_list, dim=-1).max(dim= -1)[0]

    feat_dict["train"] = out_train
    feat_dict["test"] = out_test
    feat_dict["val"] = out_val

    precompute_time = perf_counter()-start
    return feat_dict, precompute_time


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def print_table(values, columns, epoch):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
