#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:06:01 2020

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
from Stochastic_EM import stochastic_em, stochastic_em_modified

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
parser.add_argument('--shuffle_level', type=float, default=0.1,
                    help='proportion of the total data used for shuffle')
parser.add_argument('--d', type=int, default=2,
                    help='dimension of the features on the first platform')
parser.add_argument('--snr', type=float, default=100,
                    help='noise added')
parser.add_argument('--noise_level', type=float, default=None,
                    help='noise added')
parser.add_argument('--seed_data', type=int, default=5,
                    help='seed for data generation')

# training
parser.add_argument('--train_iter', type=int, default=1,
                    help='total number of traning steps')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='batch size')
parser.add_argument('--lr_S', type=float, default=1e-2,
                    help='learning rate for learning S, useful for when nn unroll')
parser.add_argument('--lr_R', type=float, default=5e-5,
                    help='learning rate for regression')
parser.add_argument('--method', type=str, default='sinkhorn_stablized',
                    help='oracle | ls | am | sinkhorn_naive | sinkhorn_stablized | sinkhorn_manual | sinkhorn_robust')
parser.add_argument('--epsilon', type=float, default=1e-4,
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
parser.add_argument('--seed_train', type=int, default=1,
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
d = args.d
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

if args.data_type == 'normal':
    x = np.random.normal(1,1,size=[n, d])
    x_val = np.random.normal(1,1,size=[n, d])
elif args.data_type == 'mix_normal':
    x = np.concatenate([np.random.normal(-1,1,size=[int(n/2), d1+d2]), 
                        np.random.normal(2,4,size=[int(n/2), d1+d2])], axis=0)
    x_val = np.concatenate([np.random.normal(-1,1,size=[int(n/2), d1+d2]), 
                        np.random.normal(2,4,size=[int(n/2), d1+d2])], axis=0)
elif args.data_type == 'uniform':
    x = np.random.uniform(0,1,size=[n, d1+d2])
    x_val = np.random.uniform(0,1,size=[n, d1+d2])


w = np.random.normal(size=[d,1])

if args.snr is not None:
    w_norm = np.linalg.norm(w)
    noise = np.sqrt(w_norm**2/args.snr)

y = x.dot(w) + noise*np.random.normal(size=[n,1])

num_shuffle = int(n*args.shuffle_level)
index_shuffle = np.random.permutation(num_shuffle)

index = np.random.permutation(n)

if args.method!='oracle':
    y_permute = y
    y_permute[:num_shuffle,:] = y[index_shuffle, :]
    
    x = x[index,:]
    y = y_permute[index,:]
    

y_val = x_val.dot(w) + noise*np.random.normal(size=[n,1])
var_y = np.sum((y_val - np.mean(y_val))**2)


class Iterator():
    def __init__(self, x, y):
        
        self.x = torch.FloatTensor(x).to(device)
        self.y = torch.FloatTensor(y).to(device)
        self.current_index = 0
        self.num_data = self.x.size(0)
        
        rand_index = torch.randperm(self.num_data)
        self.x = self.x[rand_index]
        self.y = self.y[rand_index]
        
    def get_batch(self, batch_size=32):
        if self.current_index+batch_size>self.num_data:
            self.current_index = batch_size
            rand_index = torch.randperm(self.num_data)
            self.x = self.x[rand_index]
            self.y = self.y[rand_index]
        else:
            self.current_index = self.current_index + batch_size
        return self.x[self.current_index-batch_size:self.current_index], \
               self.y[self.current_index-batch_size:self.current_index]
    

data_iterator = Iterator(x,y)

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
                            rho1=args.rho1, rho2=args.rho2, eta=args.eta )

if args.method!='oracle' and args.method!='ls':

    loss_list = []
    batch_loss_list = []
    epoch_count = 0
    best_in_set = 0
    best_est = None
    lam = 1e-4
    y_sort = np.sort(y, axis=0)
    for batch_idx in range(int(1e4)):

        batch_x1, batch_x2_y = data_iterator.get_batch(d)
        with torch.no_grad():
            x_inv = torch.inverse(batch_x1.transpose(0,1).mm(batch_x1)+lam*torch.eye(d, device=device)).mm(batch_x1.transpose(0,1))
            w_est = x_inv.mm(batch_x2_y).cpu().numpy()
    
        error = np.sum(np.abs(np.sort(x.dot(w_est), axis=0)-y_sort),axis=1)
        in_set = np.sum(error<1e-2)

        if best_in_set<in_set:
            best_in_set = in_set
            index_sort = np.argsort(x.dot(w_est)[:,0])
            x_sort = x[index_sort]

            x_inv = np.linalg.inv(x_sort.T.dot(x_sort)+lam*np.eye(d)).dot(x_sort.T)
            best_est = x_inv.dot(y_sort)

        
        if batch_idx % int(n/d) == 0 and batch_idx!=0:
            epoch_count += 1
            
            if epoch_count % print_every == 0:
                print('epoch:', epoch_count, 'loss:', best_in_set)

    if best_est is not None:
        pred=x_val.dot(best_est)
        residual= np.sum((y_val-pred)**2)
        print('initial error: ', residual/var_y)
    else:
        print('No good model found!')
        residual = 1

if args.method=='em':
    # result will be very bad without this init
    coefs, rss_tss = stochastic_em_modified(x_sort,y_sort, w_init=best_est)
    
    pred = x_val.dot(coefs)
    rss_val = np.sum((pred - y_val) ** 2)
    if args.save_val_result:
        with open(args.save_val_result, 'a') as f:
            f.write('*'*80+'\n')
            f.write('Seed: '+str(args.seed_data)+'\n')
            f.write("Varol: "+str(best_in_set)+"  error: "+str(residual/var_y)+'\n')
            f.write('EM: '+str(rss_tss)+"  error: "+str(rss_val/var_y)+'\n')

    print(rss_val/var_y)

elif args.method=='oracle' or args.method=='ls' or residual/var_y<10:
    
    Rmodel = torch.nn.Linear(d, 1, bias=False)#.to(device)

    if args.method!='oracle' and args.method!='ls':
        for parameter in Rmodel.parameters():
            parameter.data = torch.FloatTensor(best_est).transpose(0,1)
        
    
    
    optimizer_R = torch.optim.SGD(
        Rmodel.parameters(), lr=lr_R, momentum=0.9, weight_decay=5e-4)
    Rmodel = Rmodel.to(device)
    loss_list = []
    batch_loss_list = []
    epoch_count = 0
    best_residual = 10
    best_loss = 1e3
    residual_list = []
    for batch_idx in range(max_iter):


        if args.method=='oracle' or args.method=='ls':
            optimizer_R.zero_grad()

            batch_x, batch_y = data_iterator.get_batch(bs)     
            
            pred = Rmodel(batch_x)
        
            loss = torch.mean((batch_y-pred)**2)
            loss.backward()
            optimizer_R.step()
        
        elif args.method=='am':
            optimizer_R.zero_grad()

            batch_x, batch_y = data_iterator.get_batch(bs)   
            
            pred = Rmodel(batch_x)
            pred_sort, _ = torch.sort(pred, dim=0)
            y_sort, _ = torch.sort(batch_y, dim=0)
            loss = F.l1_loss(pred_sort, y_sort)*bs
            loss.backward()
            optimizer_R.step()
            
            
        else:
            batch_x, batch_y = data_iterator.get_batch(bs)
            
            pred = Rmodel(batch_x)
            
            batch_y = batch_y.unsqueeze(1)
            
            C = (batch_y-pred)**2
            C = C.sum(-1)
        
            Smodel.epsilon = args.epsilon*(C.max().detach())
            optimizer_R.zero_grad()
            S = Smodel(C)
            loss = torch.sum(S*C)
            loss.backward()

            
            optimizer_R.step()
            if loss.data.item()>1e10:
                break
        batch_loss_list.append(loss.item())
        
        if batch_idx % int(n/bs) == 0 and batch_idx!=0:
            epoch_count += 1
            loss_list.append(np.mean(batch_loss_list))
            batch_loss_list = []
            
            if epoch_count % print_every == 0:
                print('epoch:', epoch_count, 'loss:', loss_list[-1], 'residual:', best_residual/var_y)
                
            if loss_list[-1]<best_loss:
                best_loss = loss_list[-1]
                Rmodel.eval()
                x_val_tensor = torch.FloatTensor(x_val)
                if torch.cuda.is_available():
                    x_val_tensor = x_val_tensor.cuda()
                pred2=Rmodel(x_val_tensor).detach().cpu().numpy()
                best_residual= np.sum((y_val-pred2)**2)
                residual_list.append(best_residual/var_y)

                print('current error: ', best_residual/var_y)

    print('best loss', best_loss)
    print('final error: ', best_residual/var_y)
    plt.plot(residual_list)
    
        
    if args.save_val_result:
        with open(args.save_val_result, 'a') as f:
            f.write('*'*80+'\n')
            f.write('Seed: '+str(args.seed_data)+', '+str(args.seed_train)+'\n')
            f.write("Varol: "+str(best_in_set)+"  error: "+str(residual/var_y)+'\n')
            f.write('ROBOT: '+str(best_loss)+"  error: "+str(best_residual/var_y)+'\n')


