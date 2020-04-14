import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.stats import norm

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
