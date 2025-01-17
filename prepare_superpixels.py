# coding: utf-8

import numpy as np
#import torch
import pickle
import time
import os
import sys
from data.superpixels import SuperPixDatasetDGL 
from data.data import LoadData
#from torch.utils.data import DataLoader
from data.superpixels import SuperPixDataset


# In[ ]:


start = time.time()

DATASET_NAME = 'CIFAR10'
dataset = SuperPixDatasetDGL(DATASET_NAME) 

print('Time (sec):',time.time() - start) # 636s=10min


# In[ ]:


#def plot_histo_graphs(dataset, title):
#    # histogram of graph sizes
#    graph_sizes = []
#    for graph in dataset:
#        graph_sizes.append(graph[0].number_of_nodes())
#        #graph_sizes.append(graph[0].number_of_edges())
#    plt.figure(1)
#    plt.hist(graph_sizes, bins=20)
#    plt.title(title)
#    plt.show()
#    graph_sizes = torch.Tensor(graph_sizes)
#    print('nb/min/max :',len(graph_sizes),graph_sizes.min().long().item(),graph_sizes.max().long().item())
    
#plot_histo_graphs(dataset.train,'trainset')
#plot_histo_graphs(dataset.val,'valset')
#plot_histo_graphs(dataset.test,'testset')


# In[ ]:


print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))

print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])


# In[ ]:


start = time.time()

with open('data/superpixels/CIFAR10.pkl','wb') as f:
        pickle.dump([dataset.train,dataset.val,dataset.test],f)
        
print('Time (sec):',time.time() - start) # 58s


# # Test load function

# In[ ]:


DATASET_NAME = 'CIFAR10'
dataset = LoadData(DATASET_NAME) # 54s
trainset, valset, testset = dataset.train, dataset.val, dataset.test


# In[ ]:


start = time.time()

batch_size = 10
collate = SuperPixDataset.collate
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)

print('Time (sec):',time.time() - start) # 0.0001s

