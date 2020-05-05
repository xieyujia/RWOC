#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:10:44 2020

@author: yujia
"""


import torch

import torch.nn.functional as F
from torch.autograd import Function

import numpy as np

def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    bs, n, m = C.size()

    v = torch.ones([bs, 1, m])/(m)
    G = torch.exp(-C/epsilon)
    if torch.cuda.is_available():
        v = v.cuda()

    for i in range(max_iter):
        u = mu/(G*v).sum(-1, keepdim=True)
        v = nu/(G*u).sum(-2, keepdim=True)

    Gamma = u*G*v
    return Gamma

def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    bs, n, m = C.size()

    f = torch.zeros([bs, n, 1])
    g = torch.zeros([bs, 1, m])
    if torch.cuda.is_available():
        f = f.cuda()
        g = g.cuda()

    epsilon_log_mu = epsilon*torch.log(mu)
    epsilon_log_nu = epsilon*torch.log(nu)

    def min_epsilon_row(Z, epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon, -1, keepdim=True)
    
    def min_epsilon_col(Z, epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon, -2, keepdim=True)

    for i in range(max_iter):
        f = min_epsilon_row(C-g, epsilon)+epsilon_log_mu
        g = min_epsilon_col(C-f, epsilon)+epsilon_log_nu
        
    Gamma = torch.exp((-C+f+g)/epsilon)
    return Gamma


    
def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    
    nu_ = nu[:,:,:-1]
    Gamma_ = Gamma[:,:,:-1]

    bs, n, k_ = Gamma.size()
    
    inv_mu = 1./(mu.view([1,-1]))  #[1, n]
    Kappa = torch.diag_embed(nu_.squeeze(-2)) \
            -torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_)   #[bs, k, k]
    nugget = 1e-10*torch.diag(torch.ones([k_-1], device=Kappa.device)).unsqueeze(0)
    inv_Kappa = torch.inverse(Kappa+nugget) #[bs, k, k]
#    print(inv_mu.max(), inv_Kappa.max())
    Gamma_mu = inv_mu.unsqueeze(-1)*Gamma_
    L = Gamma_mu.matmul(inv_Kappa) #[bs, n, k]
    G1 = grad_output_Gamma * Gamma #[bs, n, k+1]
    
    g1 = G1.sum(-1)
    G21 = (g1*inv_mu).unsqueeze(-1)*Gamma  #[bs, n, k+1]
    g1_L = g1.unsqueeze(-2).matmul(L)  #[bs, 1, k]
    G22 = g1_L.matmul(Gamma_mu.transpose(-1,-2)).transpose(-1,-2)*Gamma  #[bs, n, k+1]
    G23 = - F.pad(g1_L, pad=(0, 1), mode='constant', value=0)*Gamma  #[bs, n, k+1]
    G2 = G21 + G22 + G23  #[bs, n, k+1]
#    print(Gamma_mu.max())
    del g1, G21, G22, G23, Gamma_mu
    
    g2 = G1.sum(-2).unsqueeze(-1) #[bs, k+1, 1]
    g2 = g2[:,:-1,:]  #[bs, k, 1]
    G31 = - L.matmul(g2)*Gamma  #[bs, n, k+1]
    G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1,-2), pad=(0, 1), mode='constant', value=0)*Gamma  #[bs, n, k+1]
    G3 = G31 + G32  #[bs, n, k+1]

    grad_C = (-G1+G2+G3)/epsilon  #[bs, n, k+1]
#    print(inv_Kappa.max(),torch.max(grad_C), epsilon)
    return grad_C

class TopKFunc(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):
        
        with torch.no_grad():
            if epsilon/C.max()>1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma!=Gamma)):
                    print('Nan appeared in Gamma, re-computing...')
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):
        
        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        #Gamma [bs, n, k+1]   
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None
    
    
    
def robust_sinkhorn_forward(C, mu, nu, epsilon, max_iter, rho1, rho2, eta):
    bs, n, m = C.size()
    small_value = 1e-5
    f = torch.zeros([bs, n, 1])
    g = torch.zeros([bs, 1, m])
    if torch.cuda.is_available():
        f = f.cuda()
        g = g.cuda()

    mu0 = mu.clone()
    nu0 = nu.clone()

    def min_epsilon_row(Z, epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon, -1, keepdim=True)
    
    def min_epsilon_col(Z, epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon, -2, keepdim=True)

    for i in range(max_iter):
        epsilon_log_mu = epsilon*torch.log(mu)
        epsilon_log_nu = epsilon*torch.log(nu)
        f = min_epsilon_row(C-g, epsilon)+epsilon_log_mu
        g = min_epsilon_col(C-f, epsilon)+epsilon_log_nu
        
        #gradient on mu and nu
#        mu -= eta*f
        mu = mu-eta*(f+epsilon*torch.log(mu))
        mu = mu*(mu>=0).float()+small_value
        mu = mu/torch.sum(mu, dim=1)
        delta_mu0 = mu-mu0
        current_diff = torch.norm(delta_mu0,dim=1)
        if current_diff > rho1:
            mu = mu0 + delta_mu0*(rho1/current_diff)
            mu = mu/torch.sum(mu, dim=1)
            
#        print(mu)
            
#        nu -= eta*g
        nu = nu-eta*(g+epsilon*torch.log(nu))
        nu = nu*(nu>=0).float()+small_value
        nu = nu/torch.sum(nu, dim=-1)
        delta_nu0 = nu-nu0
        current_diff = torch.norm(delta_nu0,dim=-1)
        if current_diff > rho2:
            nu = nu0 + delta_nu0*(rho2/current_diff)
            nu = nu/torch.sum(nu, dim=-1)
#        print(nu)
        
    Gamma = torch.exp((-C+f+g)/epsilon)
    return Gamma, mu, nu, f, g



class TopKFunc_robust(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter, rho1, rho2, eta):
        
        with torch.no_grad():
            Gamma, mu1, nu1, f, g = robust_sinkhorn_forward(C, mu, nu, epsilon, max_iter, rho1, rho2, eta)
            ctx.save_for_backward(mu, nu, Gamma, mu1, nu1, f, g)
            ctx.epsilon = epsilon
#            print('-----------------------------\n')
        return Gamma
    

    @staticmethod
    def backward(ctx, grad_output_Gamma):
        
        epsilon = ctx.epsilon
        mu, nu, Gamma, mu1, nu1, f, g = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        #Gamma [bs, n, k+1]   
        with torch.no_grad():
            bs, n, m = Gamma.size()
            n1 = int(n/2)
            m1 = int(m/2)
            small_value = 1e-10
             
            ones = torch.ones([bs], device=Gamma.device,requires_grad=False)
            
            x1 = torch.sum((mu1-mu)[:, :n1, 0], dim=-1)
            x2 = torch.sum((mu1-mu)[:, n1:, 0], dim=-1)
            b1 = torch.sum(f[:, :n1, 0]+epsilon*torch.log(mu1[:,:n1,0]), dim=-1, keepdim=True)
            b2 = torch.sum(f[:, n1:, 0]+epsilon*torch.log(mu1[:,n1:,0]), dim=-1, keepdim=True)
            weights = torch.stack([torch.stack([ones*n1,x1],dim=-1), torch.stack([ones*(n-n1),x2],dim=-1)], dim=-2)
            weights = torch.eye(2, device=Gamma.device).unsqueeze(0)+weights
            lambda1 = torch.inverse(weights).bmm(torch.stack([b1, b2],dim=-2)) 
#            print(lambda1)
            
            x1 = torch.sum((nu1-nu)[:, :m1, 0], dim=-1)
            x2 = torch.sum((nu1-nu)[:, m1:, 0], dim=-1)
            b1 = torch.sum(g[:, :m1, 0]+epsilon*torch.log(nu1[:,:m1,0]), dim=-1, keepdim=True)
            b2 = torch.sum(g[:, m1:, 0]+epsilon*torch.log(nu1[:,m1:,0]), dim=-1, keepdim=True)
            weights = torch.stack([torch.stack([ones*m1,x1],dim=-1), torch.stack([ones*(m-m1),x2],dim=-1)], dim=-2)
            weights = torch.eye(2, device=Gamma.device).unsqueeze(0)+weights
            lambda2 = torch.inverse(weights).bmm(torch.stack([b1, b2],dim=-2)) 
#            print(lambda2)
            
            mu_adjust = mu1 + epsilon/(2*lambda1[:,1]+epsilon/mu1)
            nu_adjust = nu1 + epsilon/(2*lambda2[:,1]+epsilon/nu1)
            
#            print(mu_adjust, nu_adjust)
            
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu_adjust, nu_adjust, epsilon)
#            print(torch.norm(grad_C))
        return grad_C, None, None, None, None, None, None, None


class Sinkhorn_custom(torch.nn.Module):
    def __init__(self, method='manual', epsilon=0.1, max_iter = 200, rho1=0.1, rho2=0.1, eta=1e-2):
        super(Sinkhorn_custom, self).__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.method = method
        self.rho1 = rho1
        self.rho2 = rho2
        self.eta = eta
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, C):
        if C.dim() == 2:
            C = C.unsqueeze(0)
            
        _, n, m = C.size()
      
        mu = torch.ones([1, n, 1], requires_grad=False, device=self.device)/n
        nu = torch.ones([1, 1, m], requires_grad=False, device=self.device)/m
        
        if self.method == 'manual':
            Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter)
        elif self.method == 'naive':
            Gamma = sinkhorn_forward(C, mu, nu, self.epsilon, self.max_iter)
        elif self.method == 'stablized':
            Gamma = sinkhorn_forward_stablized(C, mu, nu, self.epsilon, self.max_iter)
        elif self.method == 'robust':
            Gamma = TopKFunc_robust.apply(C, mu, nu, self.epsilon, self.max_iter, self.rho1, self.rho2, self.eta)
        
        return Gamma*min(m,n)
