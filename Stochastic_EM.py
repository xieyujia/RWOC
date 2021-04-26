import numpy as np
from sklearn.linear_model import LinearRegression


def log_prob_of_order(x,y,coefs,s,order):
    eps = 1e-6
    y_ = y[order]
    log_prob = np.sum(-np.square(x.dot(coefs)-y_)/(2*(s+eps)**2))
    return log_prob


def stochastic_em(x,y,w_init=None, fit_intercept=False,steps=15,\
    return_all_weights=False,verbose=False, enhanced=False, \
    groups=None, mcmc_steps=None, interval_between_mcmc_steps=None):

    n, d = x.shape
    
    lr =  LinearRegression(fit_intercept=fit_intercept)
    if w_init is not None:
        coefs = w_init
    else:
        lr.fit(x,y)
        coefs = lr.coef_.T
    s = np.sqrt(np.sum(np.square(y - x.dot(coefs)))/(n-d)) #RMS of residuals

    ws = list()

    permutation = np.diag(np.ones(n))
    
    for i in list(range(steps)):
        order = np.arange(n)
        y_mean = np.zeros(n)
        
        permutation_mean = np.zeros([n,n])

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
                    # print(np.sum(permutation[order]))
                    permutation_mean += permutation[order]
                    # print(order)
                    y_mean += y.flatten()[order]
                    mean_step_counter += 1
                    interval_mcmc_counter = 0

        # print(permutation_mean)
        y = y_mean/mean_step_counter
        # print(mean_step_counter)
        permutation_mean = permutation_mean / mean_step_counter
        y = y.reshape(-1,1)
        # print(y_mean)
        
        lr.fit(x,y)
        coefs = lr.coef_.T
        s = np.sqrt(np.sum(np.square(y - x.dot(coefs)))/(n-d)) #RMS of residuals
        ws.append(coefs)

    pred = x.dot(coefs)
    rss = np.sum((pred - y) ** 2)
    tss = np.sum((y - y.mean()) ** 2)

    return coefs, rss / tss, permutation_mean


def stochastic_em_modified(x,y,w_init=None, fit_intercept=False,steps=15,\
    return_all_weights=False,verbose=False, enhanced=False, \
    groups=None, mcmc_steps=None, interval_between_mcmc_steps=None):

    n, d = x.shape
    if w_init is not None:
        coefs = w_init
    else:
        lr =  LinearRegression(fit_intercept=fit_intercept)
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

        lr =  LinearRegression(fit_intercept=fit_intercept)
        lr.fit(x,y)
        coefs = lr.coef_.T
        s = np.sqrt(np.sum(np.square(y - x.dot(coefs)))/(n-d)) #RMS of residuals
        ws.append(coefs)
        
        index_sort = np.argsort(x.dot(coefs)[:,0])
        x = x[index_sort]
        y = np.sort(y, axis=0)

    pred = x.dot(coefs)
    rss = np.sum((pred - y) ** 2)
    tss = np.sum((y - y.mean()) ** 2)

    return coefs, rss / tss
