#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:57:39 2020

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
parser.add_argument('--n', type=int, default=int(1e2),
                    help='total number of data')
parser.add_argument('--train_level', type=float, default=1.,
                    help='proportion of the total data used for training')
parser.add_argument('--d1', type=int, default=1,
                    help='dimension of the features on the first platform')
parser.add_argument('--d2', type=int, default=1,
                    help='dimension of the features on the second platform')
parser.add_argument('--noise_level', type=float, default=0.0,
                    help='noise added')
parser.add_argument('--seed_data', type=int, default=1,
                    help='seed for data generation')

# training
parser.add_argument('--train_iter', type=int, default=500,
                    help='total number of traning steps')
parser.add_argument('--lr_R', type=float, default=1e-3,
                    help='learning rate for regression')
parser.add_argument('--method', type=str, default='sinkhorn_stablized',
                    help='nn | sinkhorn_naive | sinkhorn_stablized | sinkhorn_manual | sinkhorn_robust')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='entropy regularization coefficient, used for Sinkhorn')
parser.add_argument('--max_inner_iter', type=int, default=200,
                    help='inner iteration number, used for Sinkhorn')
parser.add_argument('--unroll_steps', type=int, default=5,
                    help='number of unrolling steps, used for nn')
parser.add_argument('--rho1', type=float, default=0.1,
                    help='relaxition for the first marginal')
parser.add_argument('--rho2', type=float, default=0.1,
                    help='relaxition for the second marginal')
parser.add_argument('--eta', type=float, default=1e-3,
                    help='grad for projected gradient descent for robust OT')
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
bs = n
max_iter = args.train_iter

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

w_for_print = np.squeeze(w)

index = np.random.permutation(n)

x1 = x[:, :d1]
x2 = x[index, d1:]
y_permute = y[index, 0]

x1 = torch.FloatTensor(x1).to(device)
x2 = torch.FloatTensor(x2).to(device)
y_permute = torch.FloatTensor(y_permute).to(device)


if method == 'sinkhorn_naive':
    Smodel = Sinkhorn_custom(method='naive', epsilon=epsilon, max_iter = max_inner_iter)
elif method == 'sinkhorn_stablized':
    Smodel = Sinkhorn_custom(method='stablized', epsilon=epsilon, max_iter = max_inner_iter)
elif method == 'sinkhorn_manual':
    Smodel = Sinkhorn_custom(method='manual', epsilon=epsilon, max_iter = max_inner_iter)
elif method == 'sinkhorn_robust':
    Smodel = Sinkhorn_custom(method='robust', epsilon=epsilon, max_iter = max_inner_iter, \
                            rho1=args.rho1, rho2=args.rho2, eta=args.eta )
    
Rmodel = torch.nn.Linear(d1+d2, 1, bias=False).to(device)


optimizer_R = torch.optim.SGD(
    Rmodel.parameters(), lr=lr_R, momentum=0.9, weight_decay=5e-4)


loss_list = []
epoch_count = 0
for batch_idx in range(max_iter):
#    print(batch_idx)
    # Get data
    
    batch_x1 = x1.unsqueeze(0).repeat(bs, 1, 1)
    batch_x2 = x2.unsqueeze(1).repeat(1, bs, 1)
    batch_x = torch.cat([batch_x1, batch_x2], dim=-1)
    pred = Rmodel(batch_x.view(bs*bs, -1)).view(bs, bs)
    
    batch_y = y_permute.unsqueeze(1).repeat(1, bs)
    
    C = (batch_y-pred)**2
#    C = C.sum(-1)
#    C = C / (C.max().detach())
            
    Smodel.epsilon = args.epsilon*(C.max().detach())
    optimizer_R.zero_grad()
    S = Smodel(C)
    loss = torch.sum(S*C)
    loss.backward()
    optimizer_R.step()
    if loss.data.item()>1e10:
        break
    loss_list.append(loss.item())
#    print('loss:', batch_loss_list[-1])
    
    epoch_count += 1
        
    if epoch_count % print_every == 0:
        print('epoch:', epoch_count, 'loss:', loss_list[-1])
        
        print(w_for_print)
        print(list(Rmodel.parameters())[0].data.numpy())
        print('-'*20)
    
    
#    print(loss, list(Rmodel.parameters()))

#%%
plt.plot(loss_list)
plt.ylim([0,1])
plt.show()
        #%%
    
print(w_for_print)
print(list(Rmodel.parameters())[0].data)

