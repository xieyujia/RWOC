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
#    del g1, G21, G22, G23, Gamma_mu
    
    g2 = G1.sum(-2).unsqueeze(-1) #[bs, k+1, 1]
    g2 = g2[:,:-1,:]  #[bs, k, 1]
    G31 = - L.matmul(g2)*Gamma  #[bs, n, k+1]
    G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1,-2), pad=(0, 1), mode='constant', value=0)*Gamma  #[bs, n, k+1]
    G3 = G31 + G32  #[bs, n, k+1]

    grad_C = (-G1+G2+G3)/epsilon  #[bs, n, k+1]
##    print(inv_Kappa.max(),torch.max(grad_C), epsilon)
##    print(G3)
#    H1 = torch.diag_embed(inv_mu) + L.matmul(Gamma_mu.transpose(-1,-2))
#    H2 = F.pad(-L, pad=(0, 1), mode='constant', value=0)
#    H4 = F.pad(inv_Kappa, pad=(0, 1, 0, 1), mode='constant', value=0)
#    
#    A1 = torch.cat([torch.cat([H1, H2],dim=-1), torch.cat([H2.transpose(-2,-1), H4],dim=-1)], dim=-2)
#    A_inv_core = A1 #+ plus
#    
#    dphi = A_inv_core[:,:n,:n].unsqueeze(-1)*Gamma.unsqueeze(-3)  \
#           + A_inv_core[:,:n,n:].unsqueeze(-2)*Gamma.unsqueeze(-3) 
#    dzeta = A_inv_core[:,n:,:n].unsqueeze(-1)*Gamma.unsqueeze(-3)  \
#           + A_inv_core[:,n:,n:].unsqueeze(-2)*Gamma.unsqueeze(-3) 
#           
#    G = grad_output_Gamma*Gamma
#    grad_C = -G + (G.sum(-1).unsqueeze(-1).unsqueeze(-1)*dphi).sum(-3)  \
#                + (G.sum(-2).unsqueeze(-1).unsqueeze(-1)*dzeta).sum(-3) 
#                
##    print(inv_Kappa)
##    print(grad_C)
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
    epsilon1 = epsilon
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
        mu = mu-eta*(f+epsilon1*torch.log(mu))
        mu = mu*(mu>=0)+small_value
        mu = mu/torch.sum(mu, dim=1)
        delta_mu0 = mu-mu0
        current_diff = torch.norm(delta_mu0,dim=1)
        if current_diff > rho1:
            mu = mu0 + delta_mu0*(rho1/current_diff)
            mu = mu/torch.sum(mu, dim=1)
            
#        print(mu)
            
#        nu -= eta*g
        nu = nu-eta*(g+epsilon1*torch.log(nu))
        nu = nu*(nu>=0)+small_value
        nu = nu/torch.sum(nu, dim=-1)
        delta_nu0 = nu-nu0
        current_diff = torch.norm(delta_nu0,dim=-1)
        if current_diff > rho2:
            nu = nu0 + delta_nu0*(rho2/current_diff)
            nu = nu/torch.sum(nu, dim=-1)
#        print(nu)
        
    Gamma = torch.exp((-C+f+g)/epsilon)
    return Gamma, mu, nu, f, g

def sinkhorn_robust_backward_get_inverse(grad_output_Gamma, Gamma, mu, nu, epsilon):
    
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
    H1 = torch.diag_embed(inv_mu) + L.matmul(Gamma_mu.transpose(-1,-2))
    H2 = F.pad(-L, pad=(0, 1), mode='constant', value=0)
    H4 = F.pad(inv_Kappa, pad=(0, 1, 0, 1), mode='constant', value=0)
    return H1, H2, H4

class TopKFunc_robust(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter, rho1, rho2, eta):
        
        with torch.no_grad():
            Gamma, mu1, nu1, f, g = robust_sinkhorn_forward(C, mu, nu, epsilon, max_iter, rho1, rho2, eta)
            ctx.save_for_backward(mu, nu, Gamma, mu1, nu1, f, g)
            ctx.epsilon = epsilon
            ctx.rho1 = rho1
            ctx.rho2 = rho2
#            print('-----------------------------\n')
        return Gamma
    

    @staticmethod
    def backward(ctx, grad_output_Gamma):
        
        epsilon = ctx.epsilon
        mu, nu, Gamma, mu1, nu1, f, g = ctx.saved_tensors
        rho1 = ctx.rho1
        rho2 = ctx.rho2
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
            
            x1 = torch.sum((nu1-nu)[:, :m1, 0], dim=-1)
            x2 = torch.sum((nu1-nu)[:, m1:, 0], dim=-1)
            b1 = torch.sum(g[:, :m1, 0]+epsilon*torch.log(nu1[:,:m1,0]), dim=-1, keepdim=True)
            b2 = torch.sum(g[:, m1:, 0]+epsilon*torch.log(nu1[:,m1:,0]), dim=-1, keepdim=True)
            weights = torch.stack([torch.stack([ones*m1,x1],dim=-1), torch.stack([ones*(m-m1),x2],dim=-1)], dim=-2)
            weights = torch.eye(2, device=Gamma.device).unsqueeze(0)+weights
            lambda2 = torch.inverse(weights).bmm(torch.stack([b1, b2],dim=-2)) 
#            print(lambda2)
            
            D1_inv = epsilon/(2*lambda1[:,1,:]+epsilon/mu1.squeeze(-1))
            D2_inv = epsilon/(2*lambda2[:,1,:]+epsilon/nu1.squeeze(-2))
            mu_adjust = mu1 + D1_inv.unsqueeze(-1)
            nu_adjust = nu1 + D2_inv.unsqueeze(-2)
            
            D_inv = torch.cat([D1_inv, D2_inv],dim=-1)

            H1, H2, H4 = sinkhorn_robust_backward_get_inverse(grad_output_Gamma, Gamma, mu_adjust, nu_adjust, epsilon)
            A1 = torch.cat([torch.cat([H1, H2],dim=-1), torch.cat([H2.transpose(-2,-1), H4],dim=-1)], dim=-2)
            A2 = - A1*D_inv.unsqueeze(-2)
            A4 = torch.diag_embed(D_inv) + A1*D_inv.unsqueeze(-1)
            
            B11 = torch.cat([torch.ones([1, n], device=Gamma.device), torch.zeros([1, m], device=Gamma.device)],dim=-1)
            B12 = 1-B11
            B13 = torch.cat([2*(mu1-mu).squeeze(-1), torch.zeros([1, m], device=Gamma.device)],dim=-1)
            B14 = torch.cat([torch.zeros([1, n], device=Gamma.device), 2*(nu1-nu).squeeze(-2)],dim=-1)
            B1 = torch.stack([B11, B12, B13, B14],dim=-1)
            
            C1 = torch.stack([B11, B12, B13*lambda1[:,1], B14*lambda2[:,1]],dim=-1).transpose(-1,-2)

            residual1 = torch.norm(mu1-mu)**2 - rho1
            residual2 = torch.norm(nu1-nu)**2 - rho2
            res = torch.FloatTensor([0,0,residual1.item(), residual2.item()]).to(Gamma.device)
            plus_kernel = torch.diag_embed(res).unsqueeze(0) - C1.bmm(A4).bmm(B1)
            plus_kernel_inv = torch.inverse(plus_kernel)
            
            plus = A2.matmul(B1).matmul(plus_kernel_inv).matmul(C1).matmul(A2.transpose(-1,-2))
            A_inv_core = A1 + plus
            
            dphi = A_inv_core[:,:n,:n].unsqueeze(-1)*Gamma.unsqueeze(-3)  \
                   + A_inv_core[:,:n,n:].unsqueeze(-2)*Gamma.unsqueeze(-3) 
            dzeta = A_inv_core[:,n:,:n].unsqueeze(-1)*Gamma.unsqueeze(-3)  \
                   + A_inv_core[:,n:,n:].unsqueeze(-2)*Gamma.unsqueeze(-3) 
                   
            G = grad_output_Gamma*Gamma
            grad_C = -G + (G.sum(-1).unsqueeze(-1).unsqueeze(-1)*dphi).sum(-3)  \
                        + (G.sum(-2).unsqueeze(-1).unsqueeze(-1)*dzeta).sum(-3) 
#            grad_C = grad_C/epsilon
#            print(Gamma)
#            print(grad_C)
#            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu_adjust, nu_adjust, epsilon)
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
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'

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
        
        return Gamma*n