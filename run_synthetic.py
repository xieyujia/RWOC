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
parser.add_argument('--n', type=int, default=int(1e5),
                    help='total number of data')
parser.add_argument('--train_level', type=float, default=0.8,
                    help='proportion of the total data used for training')
parser.add_argument('--d1', type=int, default=3,
                    help='dimension of the features on the first platform')
parser.add_argument('--d2', type=int, default=2,
                    help='dimension of the features on the second platform')
parser.add_argument('--noise_level', type=float, default=0.1,
                    help='noise added')
parser.add_argument('--seed_data', type=int, default=1,
                    help='seed for data generation')

# training
parser.add_argument('--train_iter', type=int, default=2000,
                    help='total number of traning steps')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--lr_S', type=float, default=1e-2,
                    help='learning rate for learning S, useful for when nn unroll')
parser.add_argument('--lr_R', type=float, default=1e-2,
                    help='learning rate for regression')
parser.add_argument('--method', type=str, default='sinkhorn_stablized',
                    help='nn | sinkhorn_naive | sinkhorn_stablized | sinkhorn_manual | sinkhorn_robust')
parser.add_argument('--epsilon', type=float, default=1e-3,
                    help='entropy regularization coefficient, used for Sinkhorn')
parser.add_argument('--max_inner_iter', type=int, default=200,
                    help='inner iteration number, used for Sinkhorn')
parser.add_argument('--unroll_steps', type=int, default=5,
                    help='number of unrolling steps, used for nn')
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

lr_S = args.lr_S
lr_R = args.lr_R
method = args.method
epsilon = args.epsilon
max_inner_iter = args.max_inner_iter
unroll_steps = args.unroll_steps
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

x1 = x[:, :d1]
x2 = x[index_train, d1:]
y_permute = y[index_train, :]
x2_y = np.concatenate([x2, y_permute], axis=1)

x2_val = x[index_val, d1:]
y_val = y[index_val, :]
x2_y_val = np.concatenate([x2_val, y_val], axis=1)

class Iterator():
    def __init__(self, x):
        
        self.x = torch.FloatTensor(x).to(device)
        self.current_index = 0
        self.num_data = self.x.size(0)
        
    def get_batch(self, batch_size=32):
        if self.current_index+batch_size>self.num_data:
            self.current_index = batch_size
        else:
            self.current_index = self.current_index + batch_size
        return self.x[self.current_index-batch_size:self.current_index]

if method=='nn':
    Smodel = SNet(bs, bs, 2).to(device)
    optimizer_S = torch.optim.SGD(
            Smodel.parameters(), lr=lr_S, momentum=0.9, weight_decay=5e-4)
elif method == 'sinkhorn_naive':
    Smodel = Sinkhorn_custom(method='naive', epsilon=epsilon, max_iter = max_inner_iter)
elif method == 'sinkhorn_stablized':
    Smodel = Sinkhorn_custom(method='stablized', epsilon=epsilon, max_iter = max_inner_iter)
elif method == 'sinkhorn_manual':
    Smodel = Sinkhorn_custom(method='manual', epsilon=epsilon, max_iter = max_inner_iter)
elif method == 'sinkhorn_robust':
    Smodel = Sinkhorn_custom(method='robust', epsilon=epsilon, max_iter = max_inner_iter, \
                             )
    
Rmodel = torch.nn.Linear(d1+d2, 1, bias=False).to(device)


optimizer_R = torch.optim.SGD(
    Rmodel.parameters(), lr=lr_R, momentum=0.9, weight_decay=5e-4)

data_iterator1 = Iterator(x1)
data_iterator2 = Iterator(x2_y)
data_iterator_val = Iterator(x2_y_val)

loss_list = []
batch_loss_list = []
epoch_count = 0
for batch_idx in range(max_iter):
    # Get data
    batch_x1 = data_iterator1.get_batch(bs)
    batch_x2_y = data_iterator2.get_batch(bs)
    
    batch_x2 = batch_x2_y[:, :-1]
    batch_y = batch_x2_y[:,-1]
    
    batch_x1 = batch_x1.unsqueeze(0).repeat(bs, 1, 1)
    batch_x2 = batch_x2.unsqueeze(1).repeat(1, bs, 1)
    batch_x = torch.cat([batch_x1, batch_x2], dim=-1)
    pred = Rmodel(batch_x.view(bs*bs, -1)).view(bs, bs)
    
    batch_y = batch_y.unsqueeze(1).repeat(1, bs)
    
    C = (batch_y-pred)**2
#    C = C.sum(-1)
    C = C / (C.max().detach())
    
    
    
    if method=='nn':
        
        C_detached = C.detach()
    
        for unroll_step in range(unroll_steps):
            optimizer_S.zero_grad()
            S = Smodel(C_detached)
            loss = torch.sum(S*C_detached)
            loss.backward()
            optimizer_S.step()
    #        print(loss.item())
            if unroll_step == 0:
                S_backup = [param.data.clone() for param in Smodel.parameters()]
    
        optimizer_R.zero_grad()
        S = Smodel(C)
        loss = torch.sum(S*C)
        loss.backward()
        optimizer_R.step()
        
        for param_backup, param_now in zip(S_backup, Smodel.parameters()):
            param_now.data = param_backup.clone()
            
        
    else:
        optimizer_R.zero_grad()
        S = Smodel(C)
        loss = torch.sum(S*C)
        loss.backward()
        optimizer_R.step()
        
    batch_loss_list.append(loss.item())
    
    
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
    
    plt.imshow(S[0,:,:].data.numpy())
    plt.show()

#%%
# val
val_bs = 100
Rmodel.eval()
Smodel.eval()
data_iterator1.current_index = 0
loss_val_list = []
for batch_idx in range(int(int(n-n*args.train_level)/val_bs)):
    # Get data
    batch_x1 = data_iterator1.get_batch(val_bs)
    batch_x2_y = data_iterator_val.get_batch(val_bs)
    
    batch_x2 = batch_x2_y[:, :-1]
    batch_y = batch_x2_y[:,-1]
    
    batch_x1 = batch_x1.unsqueeze(0).repeat(val_bs, 1, 1)
    batch_x2 = batch_x2.unsqueeze(1).repeat(1, val_bs, 1)
    batch_x = torch.cat([batch_x1, batch_x2], dim=-1)
    pred = Rmodel(batch_x.view(val_bs*val_bs, -1)).view(val_bs, val_bs)
    
    batch_y = batch_y.unsqueeze(1).repeat(1, val_bs)
    
    C = (batch_y-pred)**2
#    C = C.sum(-1)
    C = C / (C.max().detach())
    
    S = Smodel(C)
    loss_val = torch.sum(S*C)
    loss_val_list.append(loss_val.item()/val_bs)

residual = np.mean(loss_val_list)
print('val error is', residual)
    
if args.save_val_result:
    with open(args.save_val_result, 'a') as f:
        f.write('*'*80+'\n')
        f.write(str(args)+'\n')
        f.write('Result: '+str(residual)+'\n')
        


