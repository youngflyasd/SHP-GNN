## Simple Hierarchical PageRank Graph Convolutional Networks

### Overview
This repo contains an example implementation of the simple hierarchical PageRank graph convolutional network (SHP-GNN) model.

This code is based on SGC. SGC removes the nonlinearities and collapes the weight matrices in Graph Convolutional Networks (GCNs) and is essentially a linear model. However, it also cannot beat GCNs in most benchmarks. SHP-GNN is a new baseline method based on PageRank.

SHP-GNN achieves competitive performance while saving much training time. Basically the computational cost is very close to SSGC.

Dataset | Metric |
:------:|:------:|
Cora    | Acc: 84.0 %     
Citeseer| Acc: 73.9 %     
Pubmed  | Acc: 80.8 %    


This home repo contains the implementation for citation networks (Cora, Citeseer, and Pubmed).

### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Data
We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).

### Usage
$ python citation.py --dataset cora     --combine max --degree 16 --epochs 500 
$ python citation.py --dataset citeseer --combine max --degree 4 --epochs 150 
$ python citation.py --dataset pubmed   --combine max --degree 32 --epochs 150 

### Acknowledgement
This repo is modified from [SGC](https://github.com/Tiiiger/SGC), and [SSGC](https://github.com/allenhaozhu/SSGC).

