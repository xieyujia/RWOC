#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar 19 16:22:25 2020

@author: yujia
"""

import argparse
import numpy as np
import time
import torch

from sinkhorn import Sinkhorn_custom


device = 'cpu'


class Iterator():
    def __init__(self, x):
        self.x = torch.FloatTensor(x).to(device)
        self.current_index = 0
        self.num_data = self.x.size(0)
        
    def get_batch(self, batch_size=32):
        if self.current_index + batch_size > self.num_data:
            self.current_index = batch_size
            rand_index = torch.randperm(self.num_data)
            self.x = self.x[rand_index]
        else:
            self.current_index = self.current_index + batch_size
        return self.x[self.current_index - batch_size : self.current_index]


class Robot():
    def __init__(self, args):
        self.bs = args.batch_size
        self.epsilon = args.epsilon
        self.max_iter = args.train_iter

        """ define model and optimizer """
        if args.method == 'sinkhorn_naive':
            Smodel = Sinkhorn_custom(method='naive', epsilon=args.epsilon, \
                max_iter=args.max_inner_iter)
        elif args.method == 'sinkhorn_stablized':
            Smodel = Sinkhorn_custom(method='stablized', epsilon=args.epsilon, \
                max_iter=args.max_inner_iter)
        elif args.method == 'sinkhorn_manual':
            Smodel = Sinkhorn_custom(method='manual', epsilon=args.epsilon, \
                max_iter=args.max_inner_iter)
        elif args.method == 'sinkhorn_robust':
            Smodel = Sinkhorn_custom(method='robust', \
                epsilon=args.epsilon, max_iter=args.max_inner_iter, \
                rho1=args.rho1, rho2=args.rho2, eta=args.eta)
            
        Rmodel = torch.nn.Linear(args.d1 + args.d2, 1, bias=False).to(device)

        optimizer_R = torch.optim.SGD(
            Rmodel.parameters(), lr=args.lr_R, momentum=0.9, weight_decay=5e-4)

        self.Smodel, self.Rmodel, self.optimizer_R = Smodel, Rmodel, optimizer_R

    def train(self, train_data):
        self.Smodel.train()
        self.Rmodel.train()

        bs = self.bs
        data_x1 = Iterator(train_data[0])
        data_x2 = Iterator(train_data[1])
        data_y = Iterator(train_data[2])

        start = time.time() 
        for batch_idx in range(self.max_iter):

            batch_x1 = data_x1.get_batch(bs)
            batch_x2 = data_x2.get_batch(bs)
            batch_y = data_y.get_batch(bs).squeeze()

            batch_x1 = batch_x1.unsqueeze(0).repeat(bs, 1, 1)
            batch_x2 = batch_x2.unsqueeze(1).repeat(1, bs, 1)
            batch_x = torch.cat([batch_x1, batch_x2], dim=-1)

            pred = self.Rmodel(batch_x.view(bs*bs, -1)).view(bs, bs)
            batch_y = batch_y.unsqueeze(1).repeat(1, bs)
            C = (batch_y - pred) ** 2
            
            self.Smodel.epsilon = self.epsilon * (C.max().detach())

            self.optimizer_R.zero_grad()
            
            S = self.Smodel(C)
            loss = torch.sum(S * C)
            loss.backward()
            
            self.optimizer_R.step()

            if batch_idx % 1 == 0:
                elapse = time.time() - start
                print('iter %d,\tloss: %.5f,\telapse: %.3fs' % (batch_idx, loss, elapse))
                start = time.time()
            if loss.data.item() > 1e10:
                print('*****Overflow Detected*****')
                break

    def eval(self, val_data):
        self.Rmodel.eval()
        self.Smodel.eval()

        bs = val_data[0].shape[0]
        data_x1 = Iterator(val_data[0])
        data_x2 = Iterator(val_data[1])
        data_y = Iterator(val_data[2])

        with torch.no_grad():

            batch_x1 = data_x1.get_batch(bs)
            batch_x2 = data_x2.get_batch(bs)
            batch_y = data_y.get_batch(bs).squeeze()

            batch_x1 = batch_x1.unsqueeze(0).repeat(bs, 1, 1)
            batch_x2 = batch_x2.unsqueeze(1).repeat(1, bs, 1)
            batch_x = torch.cat([batch_x1, batch_x2], dim=-1)

            pred = self.Rmodel(batch_x.view(bs*bs, -1)).view(bs, bs)
            batch_y_repeat = batch_y.unsqueeze(1).repeat(1, bs)

            C = (batch_y_repeat - pred) ** 2
            
            self.Smodel.epsilon = self.epsilon * (C.max().detach())

            S = self.Smodel(C)
            loss = torch.sum(S * C)

            RSS = loss.item()
            TSS = torch.sum((batch_y - batch_y.mean()) ** 2).item()
            return RSS / TSS            
