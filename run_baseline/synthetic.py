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


class Trainer():
    def __init__(self, args):
        self.bs = args.batch_size
        self.epsilon = args.epsilon
        self.max_iter = args.train_iter
        self.n = args.n
        self.print_every = args.print_every
        self.train_level = args.train_level

        """ generate data """
        self.data_iterator1, self.data_iterator2, self.data_iterator_val = \
            self.generate_data(args)

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


    @staticmethod
    def generate_data(args):
        n, d1, d2 = args.n, args.d1, args.d2

        if args.data_type == 'normal':
            x = np.random.normal(0, 1, size=[n, d1+d2])
        elif args.data_type == 'mix_normal':
            x = np.concatenate([np.random.normal(-1, 1, size=[int(n/2), d1+d2]), 
                                np.random.normal(2, 4, size=[int(n/2), d1+d2])],
                                axis=0)
        elif args.data_type == 'uniform':
            x = np.random.uniform(0, 1, size=[n, d1+d2])

        w = np.random.normal(size=(d1+d2, 1))
        y = x.dot(w) + args.noise_level * np.random.normal(size=(n, 1))

        index = np.random.permutation(n)
        num_train = int(n * args.train_level)
        index_train = index[:num_train]
        index_val = index[num_train:]

        x1 = x[:, :d1]
        x2 = x[index_train, d1:]
        y_permute = y[index_train, :]
        x2_y = np.concatenate([x2, y_permute], axis=1)

        x2_val = x[index_val, d1:]
        y_val = y[index_val, :]
        x2_y_val = np.concatenate([x2_val, y_val], axis=1)

        return Iterator(x1), Iterator(x2_y), Iterator(x2_y_val)

    def train(self):
        self.Smodel.train()
        self.Rmodel.train()

        bs = self.bs

        loss_list = []
        batch_loss_list = []
        epoch_count = 0
        start = time.time()
        for batch_idx in range(self.max_iter):

            batch_x1 = self.data_iterator1.get_batch(bs)
            batch_x2_y = self.data_iterator2.get_batch(bs)
            
            batch_x2 = batch_x2_y[:, :-1]
            batch_y = batch_x2_y[:,-1]

            batch_x1 = batch_x1.unsqueeze(0).repeat(bs, 1, 1)
            batch_x2 = batch_x2.unsqueeze(1).repeat(1, bs, 1)
            batch_x = torch.cat([batch_x1, batch_x2], dim=-1)

            pred = self.Rmodel(batch_x.view(bs*bs, -1)).view(bs, bs)
            batch_y = batch_y.unsqueeze(1).repeat(1, bs)
            C = (batch_y - pred) ** 2
            
            self.Smodel.epsilon = self.epsilon * (C.max().detach())

            self.optimizer_R.zero_grad()
            
            S = self.Smodel(C)
            print(C.shape, S.shape)
            print((S*C).shape)
            exit(0)
            loss = torch.sum(S * C)
            loss.backward()
            
            self.optimizer_R.step()
            if loss.data.item() > 1e10:
                break
            
            batch_loss_list.append(loss.item())
            if batch_idx % int(self.n / bs) == 0 and batch_idx != 0:
                epoch_count += 1
                loss_list.append(np.mean(batch_loss_list))
                batch_loss_list = []
                if epoch_count % self.print_every == 0:
                    elapse = time.time() - start
                    print('epoch: %d, loss: %.5f, time: %.3fs' % (epoch_count, loss_list[-1], elapse))
                    start = time.time()

    def eval(self):
        self.Rmodel.eval()
        self.Smodel.eval()

        val_bs = 100

        self.data_iterator1.current_index = 0
        loss_val_list = []
        y_val_list = []
        for batch_idx in range(int(int(self.n-self.n*self.train_level)/val_bs)):
            batch_x1 = self.data_iterator1.get_batch(val_bs)
            batch_x2_y = self.data_iterator_val.get_batch(val_bs)
            
            batch_x2 = batch_x2_y[:, :-1]
            batch_y = batch_x2_y[:,-1]
            
            batch_x1 = batch_x1.unsqueeze(0).repeat(val_bs, 1, 1)
            batch_x2 = batch_x2.unsqueeze(1).repeat(1, val_bs, 1)
            batch_x = torch.cat([batch_x1, batch_x2], dim=-1)
            pred = self.Rmodel(batch_x.view(val_bs*val_bs, -1)).view(val_bs, val_bs)
            
            batch_y_repeat = batch_y.unsqueeze(1).repeat(1, val_bs)
            
            C = (batch_y_repeat - pred) ** 2
            S = self.Smodel(C)
            loss_val = torch.sum(S * C)
            loss_val_list.append(loss_val.item())
            y_val_list.extend(list(batch_y.cpu().data.numpy()))

        residual = np.sum(loss_val_list)
        var_y = np.sum((np.asarray(y_val_list) - np.mean(y_val_list))**2)
        print('val error is', residual / len(y_val_list))
        print('rel val error is', residual / var_y)
            

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    """ data generation """
    parser.add_argument('--data_type', type=str, default='normal',
                        help='normal | mix_normal | uniform')
    parser.add_argument('--n', type=int, default=int(1e3),
                        help='total number of data')
    parser.add_argument('--train_level', type=float, default=0.9,
                        help='proportion of the total data used for training')
    parser.add_argument('--d1', type=int, default=1,
                        help='dimension of the features on the first platform')
    parser.add_argument('--d2', type=int, default=2,
                        help='dimension of the features on the second platform')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='noise added')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for data generation')

    """ training """
    parser.add_argument('--train_iter', type=int, default=100,
                        help='total number of traning steps')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size')
    parser.add_argument('--lr_R', type=float, default=5e-5,
                        help='learning rate for regression')
    parser.add_argument('--method', type=str, default='sinkhorn_robust',
                        help='sinkhorn_naive | sinkhorn_stablized | sinkhorn_manual | sinkhorn_robust')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                        help='entropy regularization coefficient, used for Sinkhorn')
    parser.add_argument('--max_inner_iter', type=int, default=100,
                        help='inner iteration number, used for Sinkhorn')
    parser.add_argument('--rho1', type=float, default=0.1,
                        help='relaxition for the first marginal')
    parser.add_argument('--rho2', type=float, default=0.1,
                        help='relaxition for the second marginal')
    parser.add_argument('--eta', type=float, default=1e-3,
                        help='grad for projected gradient descent for robust OT')

    """ printing """
    parser.add_argument('--print_every', type=int, default=1,
                        help='number of epochs for each print')

    args = parser.parse_args()

    """ set random seed """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """ train the model """
    trainer = Trainer(args)
    trainer.train()
    trainer.eval()


if __name__ == '__main__':
    main()
