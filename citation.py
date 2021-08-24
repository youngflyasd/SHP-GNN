import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, set_seed, shp_precompute
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.beta, args.delta, args.cuda)

model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda, args.degree, args.combine)

if args.model == "SGC": features, precompute_time = shp_precompute(features, adj, args.degree, args.combine)


print("{:.4f}s".format(precompute_time))

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, r_x = model(train_features)
        r_o = output
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output, _ = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time, r_x, r_o                   #r_x：训练时未分类的原始X, r_o:最后一次训练的输出（标签），

def test_regression(model, test_features, test_labels):
    model.eval()
    output, r_x = model(test_features) 
    return accuracy(output, test_labels), output, r_x      #r_x：测试时未分类的原始X

def plot_with_labels(lowDWeights, labels,i):
    plt.cla()
    # 降到二维了，分别给x和y
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # 遍历每个点以及对应标签
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255/9 * s)) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer')

    plt.savefig("{}.jpg".format(i))

def plot_with_label(lowDWeights, labels, i):
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        if s == 0:
            plt.scatter(x, y, s=7, c= "#d71345")  #红色
        if s == 1:
            plt.scatter(x, y, s=7, c= "#2C9F2C")  #绿色
        if s == 2:
            plt.scatter(x, y, s=7, c= "#2b4490")  #蓝色
        if s == 3:
            plt.scatter(x, y, s=7, c= "#f391a9")  #蓝色
        if s == 4:
            plt.scatter(x, y, s=7, c= "#121a2a")  #蓝色
        if s == 5:
            plt.scatter(x, y, s=7, c= "#D9D919")  #蓝色
        if s == 6:
           plt.scatter(x, y, s=7, c= "#00a6ac")  #蓝色
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')
    plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0,wspace=0)
    plt.savefig("cora_{}_test_max.jpg".format(i))
    plt.savefig('cora_{}_test_max.eps'.format(i), dpi=600)
    
if args.model == "SGC":  
    model, acc_val, train_time, r_x, pre_lab = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test, pre_lab_test, r_x_test = test_regression(model, features[idx_test], labels[idx_test])  #pre_lab_test:训练输出标签， r_x_test：测试时未分类的原始X

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))


#=============================================================================================
#t-SNE

#idx_tsne = range(3000)
#pre_labels = torch.max(pre_lab.data, 1)[1].cpu().numpy()
#last_features = r_x.cpu().numpy()


# pre_labels = torch.max(pre_lab_test.data, 1)[1].cpu().numpy()      #将标签从tensor转成numpy GPU -> CPU  
# last_features = r_x_test.cpu().numpy()                             #特征 GPU -> CPU


# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)  #特征降维
# low_dim_embs = tsne.fit_transform(last_features)
# plot_with_label(low_dim_embs, pre_labels, 64)                        #画特征
