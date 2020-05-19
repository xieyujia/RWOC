import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.stats import norm

import Methods


def mse(x,y,w): #actually returns RMSE
    assert x.dot(w).shape == y.shape
    return np.sqrt(np.mean(np.square(x.dot(w)-y)))


def log_prob_of_order(x,y,coefs,s,order):
    eps = 1e-6
    y_ = y[order]
    log_prob = np.sum(-np.square(x.dot(coefs)-y_)/(2*(s+eps)**2))
    return log_prob


def calc_relative_error(w0, w):
    w = np.array(w); w0 = np.array(w0)
    w = w.flatten()
    w0 = w0.flatten()
    return np.linalg.norm(w-w0)/np.linalg.norm(w0)


def calc_error(w0, w, multi=False):
    w = np.array(w); w0 = np.array(w0)
    if (multi):
        w = w.reshape(w0.shape)
        return np.linalg.norm(w-w0,axis=1)
    else:
        w = w.flatten()
        w0 = w0.flatten()
        return np.linalg.norm(w-w0)


def calc_normalized_error(w0, w, multi=False):
    w = np.array(w); w0 = np.array(w0)
    d = w.flatten().shape
    diff = w - w0
    err = np.abs(diff)
    err_total = np.sum(err)
    return err_total/d


def calc_rmse(y_,y):
    return np.sqrt(np.mean((y.flatten() - y_.flatten())**2))


def normalize(y):
    return (y-np.min(y))/(np.max(y) - np.min(y))


def shuffle_up_to_delta(y, eps, return_groups=False):
    vals = np.linspace(np.min(y),np.max(y),(np.max(y)-np.min(y))/eps+1)
    groups = np.copy(y)
    #print("Actual delta:",vals[1]-vals[0])
    y_ = y.copy()
    for i in range(len(vals)-1):
        idx = np.where(np.logical_and(y>=vals[i], y<=vals[i+1]))
        y_[idx] = np.random.permutation(y[idx])
        groups[idx] = i
    #plt.plot(y, y_,'.')
    if (return_groups):
        return y_, groups
    return y_


def enhanced_sort(x, w, groups):
    for i in np.unique(groups):
        idx = np.where(groups==i)[0]
        order = np.argsort(x.dot(w).flatten()[idx])
        x[idx] = x[idx][order]
    return x


def sorted_distance(y, y_):
    y = np.sort(y.flatten())
    y_ = np.sort(y_.flatten())
    return np.linalg.norm(y-y_)


def ols(x,y,alpha=0.01,fit_intercept=False):
    if alpha==0:
        lr =  LinearRegression(fit_intercept=fit_intercept)
    else:
        lr = Ridge(fit_intercept=fit_intercept, alpha=alpha)
    lr.fit(x, y)
    w = lr.coef_.T
    return w


def shuffle_within_num_groups_by_feature(features, labels, feature_index, n_clusters = 1, fraction_missorted=0, random_assignment=False):
    n,d = features.shape
    labels = labels.flatten()

    cluster_vector = np.repeat(range(n_clusters), n//n_clusters)
    padding = np.repeat([n_clusters-1],n%n_clusters)
    cluster_vector = np.concatenate((cluster_vector, padding))

    #Handle the missorts (they get randomly put into a bin ahead or behind)
    missort_idx = np.random.permutation(n)[:int(fraction_missorted*n)]
    missort = list()
    for i in missort_idx:
        if cluster_vector[i]==0:
            missort.append(1)
        elif cluster_vector[i]==n_clusters:
            missort.append(-1)
        else:
            if np.random.random()<0.5:
                missort.append(-1)
            else:
                missort.append(1)
    cluster_vector[missort_idx] += np.array(missort, dtype=int)

    #sort the features and labels
    order = np.argsort(features[:,feature_index])
    #if (random_assignment):
        #order = np.random.permutation(order)

    features = features[order]
    labels = labels[order]


    #randomize the label ordering
    for i in range(n_clusters):
        idx = np.where(cluster_vector==i)
        labels[idx] = np.random.permutation(labels[idx])
        features[idx] = np.random.permutation(features[idx])

    return features, labels, cluster_vector


def shuffle_within_num_groups(features, labels, n_clusters = 1, fraction_missorted=0, random_assignment=False):
    n,d = features.shape
    labels = labels.flatten()

    cluster_vector = np.repeat(range(n_clusters), n//n_clusters)
    padding = np.repeat([n_clusters-1],n%n_clusters)
    cluster_vector = np.concatenate((cluster_vector, padding))

    #Handle the missorts (they get randomly put into a bin ahead or behind)
    missort_idx = np.random.permutation(n)[:int(fraction_missorted*n)]
    missort = list()
    for i in missort_idx:
        if cluster_vector[i]==0:
            missort.append(1)
        elif cluster_vector[i]==n_clusters:
            missort.append(-1)
        else:
            if np.random.random()<0.5:
                missort.append(-1)
            else:
                missort.append(1)
    cluster_vector[missort_idx] += np.array(missort, dtype=int)


    #sort the features and labels
    order = np.argsort(labels)
    #if (random_assignment):
        #order = np.random.permutation(order)

    features = features[order]
    labels = labels[order]


    #randomize the label ordering
    for i in range(n_clusters):
        idx = np.where(cluster_vector==i)
        labels[idx] = np.random.permutation(labels[idx])
        features[idx] = np.random.permutation(features[idx])

    return features, labels, cluster_vector


def calc_angle(w0, w):
    w = np.array(w); w0 = np.array(w0)
    w = w.flatten()
    w0 = w0.flatten()
    return np.arccos(np.dot(w,w0)/(np.linalg.norm(w)*np.linalg.norm(w0)))


def semi_mean_ols_y(features,labels, groups, alpha=0.01):
    if alpha==0:
        lr =  LinearRegression(fit_intercept=False)
    else:
        lr = Ridge(fit_intercept=False, alpha=alpha)

    xs = list()
    ys = list()
    for i in np.unique(groups):
        idxs = np.where(groups==i)[0]
        y = np.mean(labels[idxs],axis=0)
        for idx in idxs:
            x = features[idx,:]
            xs.append(x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    #print(xs.shape, ys.shape)
    lr.fit(xs, ys)
    return lr.coef_.T


def semi_mean_ols_x(features,labels, groups, alpha=0.01):
    if alpha==0:
        lr =  LinearRegression(fit_intercept=False)
    else:
        lr = Ridge(fit_intercept=False, alpha=alpha)
    xs = list()
    ys = list()
    for i in np.unique(groups):
        idxs = np.where(groups==i)[0]
        x = np.mean(features[idxs,:],axis=0)
        for idx in idxs:
            xs.append(x)
            y = labels[idx]
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    #print(xs.shape, ys.shape)
    lr.fit(xs, ys)
    return lr.coef_.T


def mean_ols(features,labels, groups, alpha=0.01):
    if alpha==0:
        lr =  LinearRegression(fit_intercept=False)
    else:
        lr = Ridge(fit_intercept=False, alpha=alpha)
    xs = list()
    ys = list()
    for i in np.unique(groups):
        idxs = np.where(groups==i)[0]
        x = np.mean(features[idxs,:],axis=0)
        y = np.mean(labels[idxs],axis=0)
        for idx in idxs:
            xs.append(x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    lr.fit(xs, ys)
    return lr.coef_.T


def mean_ols_with_noise(features,labels, groups, noise=0.01, alpha=0.01):
    if alpha==0:
        lr =  LinearRegression(fit_intercept=False)
    else:
        lr = Ridge(fit_intercept=False, alpha=alpha)
    xs = list()
    ys = list()
    for i in np.unique(groups):
        for _ in range(100):
            idx = np.where(groups==i)[0]
            x = np.mean(features[idx,:],axis=0)
            xs.append(x)
            y = np.mean(labels[idx],axis=0)
            y = y + np.random.normal(0,noise,size=y.shape)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    #print(xs.shape, ys.shape)
    lr.fit(xs, ys)
    return lr.coef_.T


def generate_distribution(dim=2, n=100, noise=0, dist='normal', mean=0, var=1, WA=None, bias=False, n_clusters=None):
    """generates data of a given dimension and distribution with given parameters
    WA -- if you would like to set the weight matrix, provide it here
    """
    if (dist=='normal'):
        X = np.random.normal(mean,var, size=[n, dim]);
    elif (dist=='half-normal-uniform'):
        X1 = np.random.rand(n, dim//2)-mean*0.5
        X2 = np.random.normal(mean, var, size=[n, dim//2])
        X = np.concatenate((X1, X2), axis=1)
    elif (dist=='uniform'):
        X = var*np.random.rand(n, dim)+mean
    elif (dist=='2normals'):
        X1 = np.random.normal(-mean,var, size=[n/2, dim])
        X2 = np.random.normal(mean,var, size=[n-n/2, dim]);
        X = np.concatenate((X1, X2), axis=0)
    elif (dist=='exponential'):
        X = np.random.exponential(var, size=[n, dim]);
    else:
        raise NameError('Invalid distribution: ' + str(dist))
    if (bias):
        X[:,0] = 1 #set the first column to be all 1s for the "intercept" term
    if (WA is None):
        WA = np.random.normal(0,1,size=[dim, 1]); #Actual weights
    else:
        WA = np.array(WA)
        WA = WA.reshape(dim, 1)
    y = np.dot(X,WA) + noise*np.random.normal(0,1,size=[n,1]); #Ordered labels
    if (n_clusters is None):
        return X, y, WA
    else:
        cluster_vector = np.tile(range(n_clusters), n//n_clusters)
        cluster_vector = np.concatenate((cluster_vector, range(n%n_clusters)))
        np.random.shuffle(cluster_vector)
        return X, y, WA, cluster_vector


def load_dataset_clusters(filename, normalize=True, n_clusters = 1):
    data = np.genfromtxt(filename, delimiter=',',skip_header=2)
    n,d = data.shape

    cluster_vector = np.tile(range(n_clusters), n//n_clusters)
    cluster_vector = np.concatenate((cluster_vector, range(n%n_clusters)))
    np.random.shuffle(cluster_vector)

    #normalize
    if (normalize):
        data = (data-np.min(data, axis=0))
        data = data/(np.max(data, axis=0)+0.01)

    labels = data[:,-1].copy()
    features = data[:,:].copy()
    features[:,-1] = 1 #add a bias column

    return features, labels, cluster_vector


def stochastic_em(x,y,fit_intercept=False,steps=15,\
    return_all_weights=False,verbose=False, enhanced=False, \
    groups=None, mcmc_steps=None, interval_between_mcmc_steps=None):

    import time

    lr =  LinearRegression(fit_intercept=fit_intercept)
    n, d = x.shape
    lr.fit(x,y)
    coefs = lr.coef_.T
    s = np.sqrt(np.sum(np.square(y - x.dot(coefs)))/(n-d)) #RMS of residuals

    ws = list()

    for i in list(range(steps)):
        print('iter %d' % i)
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

    pred = x.dot(coefs)
    rss = np.sum((pred - y) ** 2)
    tss = np.sum((y - y.mean()) ** 2)

    return coefs, rss / tss


def generate_data(n, d1, d2):
    n_full, d1, d2 = 2*n, d1, d2

    x = np.random.normal(0, 1, size=[n_full, d1+d2])
    
    w = np.random.normal(size=(d1+d2, 1))
    y = x.dot(w) + 0.1 * np.random.normal(size=(n_full, 1))

    x_train = x[:n, :]
    x_val = x[n:, :]

    y_train = y[:n]
    y_val = y[n:]
    index = np.random.permutation(n)

    def build_dataset(x, y, index): 
        # x1 is not shuffled, i.e., has 1-1 correspondense with y;
        # x2 is shuffled
        x1 = x[:, :d1]
        x2 = x[:, d1:]
        x2 = x2[index, :]
        return (x1, x2, y)

    train_data = build_dataset(x_train, y_train, index)
    val_data = build_dataset(x_val, y_val, index)
    return train_data, val_data, (x, y, w)


def eval(x1, x2, y, w, single_x=False):
    """
    Input: x1 and x2 are aligned with y
    """
    if not single_x:
        x = np.concatenate((x1, x2), axis=1)
    else:
        assert x2 is None
        x = x1

    pred = x.dot(w)
    RSS = np.sum((pred - y) ** 2)
    TSS = np.sum((y - y.mean()) ** 2)
    return RSS / TSS


if __name__ == '__main__':
    # d = 30
    # noise = 1
    # iters = 5
    # ns = np.linspace(100,500,5,dtype=int)

    # error_soft = np.zeros((iters,len(ns)))
    # for i in range(iters):
    #     print(i, end='| ')
    #     x_, y__, w0_ = generate_distribution(n=np.max(ns), dim=d,  dist='normal', bias=False, noise=0)
    #     y_ = y__ + np.random.normal(0,noise,y__.shape)
    #     for n_i, n in enumerate(ns):
    #         print(n, end=' ')
    #         y = y_[:n]; x = x_[:n]
    #         y = np.random.permutation(y)
    #         weights = stochastic_em(x,y,steps=50,return_all_weights=True)
    #         error = calc_error(w0_, weights[-1])
    #         error_soft[i,n_i] = error
    # print(error_soft)

    err_em = []
    err_ls = []
    for i in range(10):
        data, _, _ = generate_data(1000, 3, 200)
        x1, x2, y = data[0], data[1], data[2]

        _, err = stochastic_em(x2, y, steps=15)
        err_em.append(err)
        print('Stochstic EM:\t%.5e' % (err))

        x = np.concatenate((x1, x2), axis=1)
        w, _ = Methods.ordinary_least_squares(y, x, 0)
        err = eval(x1, x2, y, w)
        err_ls.append(err)
        print('Least Squares:\t%.5e' % (err))
        print()

    err_em = np.array(err_em)
    err_ls = np.array(err_ls)
    print('Stochstic EM:\tmean=%.5e,\tstd=%.5e' % (err_em.mean(), err_em.std()))
    print('Least Squares:\tmean=%.5e,\tstd=%.5e' % (err_ls.mean(), err_ls.std()))
