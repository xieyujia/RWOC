#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:53:16 2020

@author: yujia
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class SBlock(nn.Module):
    '''the BasicBlock.'''

    def __init__(self, n, m):
        super(SBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(n)
        self.l1 = nn.Linear(n, n)
        self.bn2 = nn.BatchNorm1d(m)
        self.l2 = nn.Linear(m, m)

    def forward(self, x):
        out = F.relu(self.bn1(x.transpose(-2, -1)))
        out = self.l1(out).transpose(-2, -1)
        out = self.l2(F.relu(self.bn2(out)))
        return out


class SNet(nn.Module):
    def __init__(self, n, m, num_blocks):
        super(SNet, self).__init__()
        self.n = n
        self.m = m
        layers = [SBlock(n, m) for _ in range(num_blocks)]
        self.layers = nn.Sequential(*layers)
        self.softmax1 = nn.Softmax(-2)
        self.softmax2 = nn.Softmax(-1)

    def forward(self, x):
        if x.dim()==2:
            x = x.unsqueeze(0)
        out = self.layers(x)
#        out = self.softmax1(out)
#        out = self.softmax2(out)
        
        visited_line = torch.zeros(self.n, device=out.device, requires_grad=False)
        mask = torch.zeros([self.n, self.m], device=out.device, requires_grad=False)
        one_n = torch.ones(self.n, device=out.device, requires_grad=False)
        out_copy = out.detach().clone()
        for line in range(self.n):
            index = torch.argmax(out_copy)
            i = index / self.m
            j = index % self.m
            visited_line[i] = 1
            mask[:,j] = one_n - visited_line
            out_copy[0,:,j] = -float('inf')
            out_copy[0,i,:] = -float('inf')
        out = out.masked_fill(mask.to(torch.bool), -float('inf'))
        out = self.softmax2(out)
        return out


class SNN(torch.nn.Module):
    def __init__(self, n, m, num_blocks):
        super(SNN, self).__init__()
        self.n = n
        self.m = m
        self.model = SNet(n, m, num_blocks)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, scores1, scores2):
        n = scores1.size(-1)
        m = scores2.size(-1)

        scores1 = scores1.view([1, n, 1])
        scores2 = scores2.view([1, 1, m])
        
        C = (scores1-scores2)**2
        C = C / (C.max().detach())
      
        Gamma = self.model(C)

        return Gamma