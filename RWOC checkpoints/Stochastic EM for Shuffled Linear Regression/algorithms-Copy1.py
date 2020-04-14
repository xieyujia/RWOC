import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.stats import norm
from utils import *

# Stochastic EM Algorithm
def em_mcmc(x,y,fit_intercept=False,steps=15,return_all_weights=False,verbose=False, enhanced=False, groups=None, mcmc_steps=None, interval_between_mcmc_steps=None):
    import time

    lr =  LinearRegression(fit_intercept=fit_intercept)
    n, d = x.shape
    lr.fit(x,y)
    coefs = lr.coef_.T
    s = np.sqrt(np.sum(np.square(y - x.dot(coefs)))/(n-d)) #RMS of residuals

    ws = list()

    for i in list(range(steps)):

        order = np.arange(n)
        y_mean = np.zeros(n)

        if mcmc_steps is None:
            mcmc_steps = np.max([int(n*np.log(n)),100])
        if interval_between_mcmc_steps is None:
            interval_between_mcmc_steps = n/10
        burn_steps = int(mcmc_steps/2)
        mean_step_counter = 0
        interval_mcmc_counter = 0

        for m_step in range(mcmc_steps):
            order_ = np.copy(order)

            i = np.random.randint(0,n)

            if (enhanced):
                if not(groups is None):
                    indices_in_same_group = np.where(groups==groups[i])[0]
                    j = np.random.choice(indices_in_same_group)
                else:
                    raise ValueError("Group Information Must Be Provided for Enhanced SLS")
            else:
                j = np.random.randint(0,n)

            order_[i] = order[j]
            order_[j] = order[i]

            p1 = log_prob_of_order(x,y,coefs,s,order)
            p2 = log_prob_of_order(x,y,coefs,s,order_)

            if p1 < p2:
                order = order_
            else:
                if np.random.random() < np.exp(p2-p1): #since they're log probabilities
                    order = order_

            if m_step>=burn_steps:
                interval_mcmc_counter += 1
                if (interval_mcmc_counter>=interval_between_mcmc_steps):
                    y_mean += y.flatten()[order]
                    mean_step_counter += 1
                    interval_mcmc_counter = 0

        y = y_mean/mean_step_counter
        y = y.reshape(-1,1)

        lr.fit(x,y)
        coefs = lr.coef_.T
        s = np.sqrt(np.sum(np.square(y - x.dot(coefs)))/(n-d)) #RMS of residuals
        ws.append(coefs)

    if (return_all_weights):
        return ws
    return coefs

# Hard EM Algorithm
def sls(x, y, steps=15, alpha=0, w_initial=None, return_all_weights = False, enhanced=False, groups=None, fit_intercept=False, n_starts=1):
    if alpha==0:
        lr =  LinearRegression(fit_intercept=fit_intercept)
    else:
        lr = Ridge(fit_intercept=fit_intercept, alpha=alpha)

    order = np.argsort(y.flatten())
    y = y[order]
    x = x[order]
    errors = list()
    optimal_score = float("-inf")
    optimal_ws = list()

    for i_start in range(n_starts):
        ws = list()
        if w_initial is None and n_starts==1:
            lr.fit(x, y)
            w = lr.coef_.T
        elif w_initial is None:
            if i_start>0:
                y_ = np.random.permutation(y)
            elif i_start==0: #try the current permutation first
                y_ = y
            lr.fit(x, y_)
            w = lr.coef_.T
        elif w_initial=='mean':
            y_ = np.copy(y)
            for i in np.unique(groups):
                idx = np.where(groups==i)[0]
                y_[idx] = np.mean(y[idx])
            lr.fit(x, y_)
            w = lr.coef_.T
        else:
            w = w_initial

        for _ in list(range(steps)):

            if (enhanced):
                if not(groups is None):
                    x = enhanced_sort(x, w, groups)
                else:
                    raise ValueError("Group Information Must Be Provided for Enhanced SLS")
            else:
                order = np.argsort(x.dot(w).flatten())
                x = x[order]

            lr.fit(x, y)
            w = lr.coef_.T
            ws.append(w)

        if lr.score(x,y)>optimal_score:
            optimal_score = lr.score(x,y)
            optimal_ws = ws.copy()

    if (return_all_weights):
        return optimal_ws
    return optimal_ws[-1]
