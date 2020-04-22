#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:45:11 2020

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:22:25 2020

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:07:08 2020

@author: yujia
"""

import argparse
import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn import Module
import numpy as np
import torch.nn.functional as F
from sinkhorn import Sinkhorn_custom
from snn import SNet

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 



parser = argparse.ArgumentParser(description='Process some integers.')
# data generation
parser.add_argument('--data_type', type=str, default='normal',
                    help='normal | mix_normal | uniform')
parser.add_argument('--n', type=int, default=int(1e3),
                    help='total number of data')
parser.add_argument('--train_level', type=float, default=0.8,
                    help='proportion of the total data used for training')
parser.add_argument('--d1', type=int, default=10,
                    help='dimension of the features on the first platform')
parser.add_argument('--d2', type=int, default=2,
                    help='dimension of the features on the second platform')
parser.add_argument('--noise_level', type=float, default=0.5,
                    help='noise added')
parser.add_argument('--seed_data', type=int, default=1,
                    help='seed for data generation')

# training
parser.add_argument('--train_iter', type=int, default=200,
                    help='total number of traning steps')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--lr_R', type=float, default=1e-3,
                    help='learning rate for regression')
parser.add_argument('--method', type=str, default='all',
                    help='all | partial')
parser.add_argument('--seed_train', type=int, default=0,
                    help='seed for traning')

# print
parser.add_argument('--print_every', type=int, default=1,
                    help='number of epochs for each print')
parser.add_argument('--visual', type=int, default=1,
                    help='whether visual is wanted')
parser.add_argument('--save_val_result', type=str, default=None,
                    help='path to store the validation result')

args = parser.parse_args()


np.random.seed(args.seed_data)
torch.manual_seed(args.seed_train)


n = args.n
d1 = args.d1
d2 = args.d2
noise = args.noise_level

max_iter = args.train_iter
bs = args.batch_size

lr_R = args.lr_R
method = args.method
print_every = args.print_every

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.data_type == 'normal':
    x = np.random.normal(0,1,size=[n, d1+d2])
elif args.data_type == 'mix_normal':
    x = np.concatenate([np.random.normal(-1,1,size=[int(n/2), d1+d2]), 
                        np.random.normal(2,4,size=[int(n/2), d1+d2])], axis=0)
elif args.data_type == 'uniform':
    x = np.random.uniform(0,1,size=[n, d1+d2])


w = np.random.normal(size=[d1+d2,1])
y = x.dot(w) + noise*np.random.normal(size=[n,1])

index = np.random.permutation(n)
num_train = int(n*args.train_level)
index_train = index[:num_train]
index_val = index[num_train:]

if method=='all':
    x_train = x[index_train,:]
    y_train = y[index_train,:]
    x_val = x[index_val,:]
    y_val = y[index_val,:]
elif method=='partial':
    x_train = x[index_train,d1:]
    y_train = y[index_train,:]
    x_val = x[index_val,d1:]
    y_val = y[index_val,:]
else:
    raise NotImplementedError
    
data_train = np.concatenate([x_train, y_train], axis=-1)
data_val = np.concatenate([x_val, y_val], axis=-1)    

class Iterator():
    def __init__(self, x):
        
        self.x = torch.FloatTensor(x).to(device)
        self.current_index = 0
        self.num_data = self.x.size(0)
        
    def get_batch(self, batch_size=32):
        if self.current_index+batch_size>self.num_data:
            self.current_index = batch_size
            rand_index = torch.randperm(self.num_data)
            self.x = self.x[rand_index]
        else:
            self.current_index = self.current_index + batch_size
        return self.x[self.current_index-batch_size:self.current_index]
    
if method=='all':
    Rmodel = torch.nn.Linear(d1+d2, 1, bias=False).to(device)
elif method=='partial':
    Rmodel = torch.nn.Linear(d2, 1, bias=False).to(device)


optimizer_R = torch.optim.SGD(
    Rmodel.parameters(), lr=lr_R, momentum=0.9, weight_decay=5e-4)


data_iterator_train = Iterator(data_train)
data_iterator_val = Iterator(data_val)

loss_list = []
batch_loss_list = []
epoch_count = 0
for batch_idx in range(max_iter):
#    print(batch_idx)
    
    optimizer_R.zero_grad()
    # Get data
    batch = data_iterator_train.get_batch(bs)
    
    batch_x = batch[:, :-1]
    batch_y = batch[:,-1]
    
    pred = Rmodel(batch_x).squeeze(-1)

    loss = torch.mean((batch_y-pred)**2)
    loss.backward()
    optimizer_R.step()
       
    batch_loss_list.append(loss.item())
#    print('loss:', batch_loss_list[-1])
    
    if batch_idx % int(n/bs) == 0 and batch_idx!=0:
        epoch_count += 1
        loss_list.append(np.mean(batch_loss_list))
        batch_loss_list = []
        
        if epoch_count % print_every == 0:
            print('epoch:', epoch_count, 'loss:', loss_list[-1])
    
    
#    print(loss, list(Rmodel.parameters()))
#%%
            
# visual
if args.visual:
    plt.plot(loss_list)
    plt.show()

#%%
# val
val_bs = 100
Rmodel.eval()
loss_val_list = []
y_val_list = []
for batch_idx in range(int(int(n-n*args.train_level)/val_bs)):
    # Get data
    optimizer_R.zero_grad()
    # Get data
    batch = data_iterator_val.get_batch(bs)
    
    batch_x = batch[:, :-1]
    batch_y = batch[:,-1]
    
    pred = Rmodel(batch_x).squeeze(-1)

    loss_val = torch.sum((batch_y-pred)**2)
    loss_val_list.append(loss_val.item())
    y_val_list.extend(list(batch_y.data.numpy()))

residual = np.sum(loss_val_list)
var_y = np.sum((np.asarray(y_val_list) - np.mean(y_val_list))**2)
print('val error is', residual/len(y_val_list))
print('rel val error is', residual/var_y)
    
if args.save_val_result:
    with open(args.save_val_result, 'a') as f:
        f.write('*'*80+'\n')
        f.write(str(args)+'\n')
        f.write('Result: RSS:'+str(residual/len(y_val_list))+', RSS/TSS'+str(residual/var_y)+'\n')
        


