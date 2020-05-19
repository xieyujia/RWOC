import copy
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


# baselines
def euclidean_distance_matrix(pts1, pts2):
    """
    Calculate distance matrix
    :param pts1: (num1, D)
    :param pts2: (num2, D)
    :return: a cost matrix (num1, num2)
    """
    num1 = pts1.shape[0]
    num2 = pts2.shape[0]
    dim = pts1.shape[1]
    pts1 = np.reshape(pts1, (num1, 1, dim))
    pts2 = np.reshape(pts2, (1, num2, dim))
    cost = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=2))
    return cost


def ridge_regression(ys, xs, gamma: float=None):
    dim = xs.shape[1]
    if gamma is None:
        w = np.linalg.inv(xs.T @ xs) @ (xs.T @ ys)
    else:
        w = np.linalg.inv(xs.T @ xs + gamma * np.eye(dim)) @ (xs.T @ ys)
    return w


def ordinary_least_squares(ys, xs, gamma: float=None):
    """
    Eq.(11) in the following Reference:
    [1] Abid, Abubakar, and James Zou.
        "Stochastic EM for shuffled linear regression."
        arXiv preprint arXiv:1804.00681 (2018).

    :param ys: (num, 1) labels
    :param xs: (num, dim) features
    :param gamma: weight of regularizer
    :return:
        w: (dim, 1) coefficents of model
        sigma2: estimated variance of noise
    """
    num, dim = xs.shape
    w = ridge_regression(ys, xs, gamma)
    sigma2 = np.sum((xs @ w - ys) ** 2) / (num - dim)
    return w, sigma2


def posterior_permutation(ys, xs, trans, w, sigma2: float=10.):
    """
    Eqs.(9, 10) in the following Reference:
    [1] Abid, Abubakar, and James Zou.
        "Stochastic EM for shuffled linear regression."
        arXiv preprint arXiv:1804.00681 (2018).

    :param ys: (num, 1) labels
    :param xs: (num, dim) features
    :param trans: (num, num) permutation matrix
    :param w: (dim, 1) coefficients of model
    :param sigma2: estimated variance of noise
    :return: prob(trans, xs, ys, w) under a Gaussian noise assumption
    """
    residual = np.sum((trans @ xs @ w - ys) ** 2) / sigma2
    prob = np.exp(-residual)
    return prob


def hard_em(ys, xs, more_data, gamma: float=None, max_iter: int=50):
    """
    Hard EM for shuffled regression
    Reference:
    [1] Abid, Abubakar, and James Zou.
        "Stochastic EM for shuffled linear regression."
        arXiv preprint arXiv:1804.00681 (2018).

    :param ys: (num, 1) labels
    :param xs: (num, dim) features
    :param gamma: the weight of ridge regression, None for pure least-squares estimation
    :param max_iter: the number of EM iterations
    :return:
        w: the (dim, 1) coefficients of model
    """
    num_samples = ys.shape[0]
    w = ridge_regression(ys, xs, gamma)
    idx_y = np.argsort(ys[:, 0])
    ys_sorted = ys[idx_y, :]
    for i in range(max_iter):
        yhat = xs @ w
        idx_yhat = np.argsort(yhat[:, 0])
        trans = np.eye(num_samples)
        trans = trans[:, idx_yhat]
        xs = trans @ xs
        w = ridge_regression(ys_sorted, xs, gamma)
    return w


def stochastic_em(ys, xs, more_data, gamma: float=None, max_iter: int=50):
    """
    Stochastic EM for shuffled regression
    Reference:
    [1] Abid, Abubakar, and James Zou.
        "Stochastic EM for shuffled linear regression."
        arXiv preprint arXiv:1804.00681 (2018).

    :param ys: (num, 1) labels
    :param xs: (num, dim) features
    :param gamma: the weight of ridge regression, None for pure least-squares estimation
    :param max_iter: the number of EM iterations
    :return:
        w: the (dim, 1) coefficients of model
    """
    sampling_steps = int(max_iter * np.log(max_iter))
    burn_steps = max_iter
    gap = int(max_iter / 10)
    num, dim = xs.shape
    w, sigma2 = ordinary_least_squares(ys, xs, gamma)

    trans_a = np.eye(num)
    for i in range(max_iter):
        trans = np.zeros((num, num))
        for j in range(sampling_steps):
            # swap two randomly chosen rows of trans_a
            idx = np.random.permutation(num)
            trans_b = copy.deepcopy(trans_a)
            trans_b[[idx[0], idx[1]], :] = trans_b[[idx[1], idx[0]], :]
            q_a = posterior_permutation(ys, xs, trans_a, w, sigma2)
            q_b = posterior_permutation(ys, xs, trans_b, w, sigma2)
            if q_b > q_a * np.random.rand():
                trans_a = trans_b
            if j > burn_steps and j % gap == 0:
                trans += trans_a

        trans = trans / sampling_steps
        w, sigma2 = ordinary_least_squares(trans.T @ ys, xs, gamma)

    x = np.concatenate((xs, trans.T @ more_data), axis=1)
    w, _ = ordinary_least_squares(trans.T @ ys, x, gamma)
    return w, trans


def gmm_em(ys, xs, gamma: float=None, max_iter: int=50):
    """
    A gmm-based EM algorithm for fuzzy regression (or called point cloud alignment)
    Reference:
    [1] Myronenko, Andriy, and Xubo Song.
        "Point set registration: Coherent point drift."
        IEEE transactions on pattern analysis and machine intelligence 32.12 (2010): 2262-2275.

    :param ys: (num, 1) labels
    :param xs: (num, dim) features
    :param gamma: the weight of ridge regression, None for pure least-squares estimation
    :param max_iter: the number of EM iterations
    :return:
        w: the (dim, 1) coefficients of model
    """
    num_x, dim = xs.shape
    num_y = ys.shape[0]
    w = np.zeros((dim, 1))
    for i in range(max_iter):
        cost = euclidean_distance_matrix(ys, xs @ w)
        sigma2 = np.sum(cost ** 2) / (num_x * num_y * dim)
        prob = np.exp(-cost ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)
        posterior = prob / np.sum(prob, axis=1, keepdims=True)

        ys_mat = np.sqrt(posterior) * np.repeat(ys, num_x, axis=1)
        ys_mat = np.reshape(ys_mat, (num_x * num_y, 1))
        xs_mat = np.repeat(np.reshape(np.sqrt(posterior), (num_y, num_x, 1)), dim, axis=2) * \
            np.repeat(np.reshape(xs, (1, num_x, dim)), num_y, axis=0)
        xs_mat = np.reshape(xs_mat, (num_x * num_y, dim))
        w = ridge_regression(ys_mat, xs_mat, gamma)
    return w


# OT-based methods
def learning_gmm(data: np.ndarray, num_clusters: int, covariance_type: str='full', bayesian: bool=False):
    if bayesian:
        gmm = BayesianGaussianMixture(n_components=num_clusters,
                                      covariance_type=covariance_type)
    else:
        gmm = GaussianMixture(n_components=num_clusters,
                              covariance_type=covariance_type)

    gmm.fit(data)
    return gmm.weights_, gmm.means_, gmm.covariances_


def wasserstein_gaussian(mu1, mu2, sigma1, sigma2):
    u1, s1, v1 = np.linalg.svd(sigma1)
    sigma1_2 = u1 @ np.diag(np.sqrt(s1)) @ v1
    tmp = sigma1_2 @ sigma2 @ sigma1_2
    u2, s2, v2 = np.linalg.svd(tmp)
    tmp2 = u2 @ np.diag(np.sqrt(s2)) @ v2
    tmp3 = sigma1 + sigma2 - 2 * tmp2
    wd = np.linalg.norm(mu1 - mu2) ** 2 + np.trace(tmp3)
    return wd


def sliced_fgwd(pts1, pts2, more_data, num_samples: int=2000, alpha: float=0.1,
                num_trials: int=100, num_projs: int=3, gamma: float=None):
    """
    Sliced fused Gromov-Wasserstein distance
    Reference:
    [1] Titouan, Vayer, et al.
        "Sliced Gromov-Wasserstein."
        Advances in Neural Information Processing Systems. 2019.

    :param pts1: (num1, dim1) source feature points
    :param pts2: (num2, dim2) target feature points
    :param num_samples: in each trial, the number of samples randomly selected from pts1, pts2
    :param alpha: 1 for Sliced Wasserstein distance, 0 for Sliced Gromov-Wasserstein
    :param num_trials: the number of trials for sub-sampling
    :param num_projs: the number of random projections used in each trial
    :param gamma: the weight of regularizer
    :return:
        the coefficients of model
        the fused GW distance fgwd
    """
    num1, dim1 = pts1.shape[0], pts1.shape[1]
    num2, dim2 = pts2.shape[0], pts2.shape[1]
    dim = max([dim1, dim2])
    trans = np.zeros((num1, num2))

    fgwd_sum = 0
    weights = 0
    for t in range(num_trials):
        idx1 = np.random.permutation(num1)[:num_samples]
        idx2 = np.random.permutation(num2)[:num_samples]
        proj = np.random.randn(dim, num_projs)
        tmp1 = pts1[idx1, :] @ proj[:dim1, :]
        tmp2 = pts2[idx2, :] @ proj[:dim2, :]

        for i in range(num_projs):
            x1 = copy.deepcopy(tmp1[:, i])
            x2 = copy.deepcopy(tmp2[:, i])
            x1 -= np.mean(x1)
            x1 /= np.std(x1)
            x2 /= np.std(x2)
            x2 -= np.mean(x2)
            id1 = np.argsort(x1)
            id2 = np.argsort(x2)
            id3 = id2[::-1]

            dw1 = np.sum((x1[id1] - x2[id2]) ** 2)
            dw2 = np.sum((x1[id1] - x2[id3]) ** 2)

            cost1 = pairwise_distances(np.expand_dims(x1[id1], axis=1),
                                       np.expand_dims(x1[id1], axis=1))
            cost2 = pairwise_distances(np.expand_dims(x2[id2], axis=1),
                                       np.expand_dims(x2[id2], axis=1))
            cost3 = pairwise_distances(np.expand_dims(x2[id3], axis=1),
                                       np.expand_dims(x2[id3], axis=1))

            dgw1 = np.sum((cost1 ** 2 - cost2 ** 2) ** 2)
            dgw2 = np.sum((cost1 ** 2 - cost3 ** 2) ** 2)

            if alpha * dw1 + (1 - alpha) * dgw1 <= alpha * dw2 + (1 - alpha) * dgw2:
                fgwd = alpha * dw1 + (1 - alpha) * dgw1
                weight = np.exp(-fgwd / (num_samples * num_samples))
                for n in range(num_samples):
                    r = idx1[id1[n]]
                    c = idx2[id2[n]]
                    trans[r, c] += weight
            else:
                fgwd = alpha * dw2 + (1 - alpha) * dgw2
                weight = np.exp(-fgwd / (num_samples * num_samples))
                for n in range(num_samples):
                    r = idx1[id1[n]]
                    c = idx2[id3[n]]
                    trans[r, c] += weight
            weights += weight
            fgwd_sum += (fgwd / (num_samples * num_samples))

        # print('{}/{} trial'.format(t + 1, num_trials))
    trans = trans / np.sum(trans)
    # trans = num2 * trans
    trans /= np.sum(trans, axis=0, keepdims=True)
    # w = ridge_regression(trans.T @ pts1, pts2, gamma)

    # pts1 is x2, pts2 is label, more_data is x1,
    # we align x2 with label, then perform linear regression
    pts = np.concatenate((trans.T @ pts1, more_data), axis=1)
    w = ridge_regression(pts2, pts, gamma)

    return w, trans, fgwd_sum / (num_trials * num_projs)


def sinkhorn_fgwd(pts1, pts2, more_data, alpha: float=0.1, outer_iter: int=50, inner_iter: int=10, gamma: float=None):
    """
    Sinkhorn fused Gromov-Wasserstein distance
    Reference:
    [1] Xu, Hongteng, Dixin Luo, and Lawrence Carin.
        "Scalable Gromov-Wasserstein Learning for Graph Partitioning and Matching."
        arXiv preprint arXiv:1905.07645 (2019).
    [2] Xu, Hongteng, et al.
        "Gromov-Wasserstein Learning for Graph Matching and Node Embedding."
        International Conference on Machine Learning. 2019.

    :param pts1: (num1, dim1) source feature points
    :param pts2: (num2, dim2) target feature points
    :param alpha: 1 for Sliced Wasserstein distance, 0 for Sliced Gromov-Wasserstein
    :param outer_iter: the number of outer iterations
    :param inner_iter: the number of inner iterations
    :param gamma: the weight of regularizer
    :return:
        the coefficients of model
        the fused GW distance fgwd
    """
    num1 = pts1.shape[0]
    num2 = pts2.shape[0]

    cost1 = euclidean_distance_matrix(pts1, pts1)
    cost2 = euclidean_distance_matrix(pts2, pts2)

    p_s = np.sum(np.exp(-cost1 ** 2), axis=1, keepdims=True)
    p_s /= np.sum(p_s)
    p_t = np.sum(np.exp(-cost2 ** 2), axis=1, keepdims=True)
    p_t /= np.sum(p_t)
    if pts1.shape[1] == pts2.shape[1]:
        cost_w = alpha * euclidean_distance_matrix(pts1, pts2)
    else:
        cost_w = alpha

    f1_st = np.repeat((cost1 ** 2) @ p_s, num2, axis=1)
    f2_st = np.repeat(((cost2 ** 2) @ p_t).T, num1, axis=0)
    cost_st = f1_st + f2_st

    trans = p_s @ p_t.T
    cost = 0
    for o in range(outer_iter):
        cost_gw = (1 - alpha) * (cost_st - 2 * (cost1 @ trans @ cost2.T))
        cost = cost_w + cost_gw
        beta = 0.01 * np.max(np.abs(cost))
        kernel = np.exp(-cost / beta) * trans
        a = np.ones((num1, 1)) / num1
        b = 0
        for i in range(inner_iter):
            b = p_t / (np.matmul(kernel.T, a))
            a = p_s / np.matmul(kernel, b)
        trans = (a @ b.T) * kernel

    # trans = num2 * trans
    fgwd = np.sum(cost * trans)
    trans /= np.sum(trans, axis=0, keepdims=True)
    # w = ridge_regression(trans.T @ pts1, pts2, gamma)

    # pts1 is x2, pts2 is label, more_data is x1,
    # we align x2 with label, then perform linear regression
    pts = np.concatenate((trans.T @ pts1, more_data), axis=1)
    w = ridge_regression(pts2, pts, gamma)

    return w, trans, fgwd


def hierarchical_gwd(ys, xs, num_clusters: int = 50, outer_iter: int=50, inner_iter: int=10, gamma: float=None):
    weight_y, means_y, covarainces_y = learning_gmm(ys, num_clusters=num_clusters)
    weight_x, means_x, covarainces_x = learning_gmm(xs, num_clusters=num_clusters)
    num_y = weight_y.shape[0]
    num_x = weight_x.shape[0]
    wd_y = np.zeros((num_y, num_y))
    wd_x = np.zeros((num_x, num_x))
    for i in range(num_y):
        for j in range(num_y):
            if i != j:
                wd_y[i, j] = wasserstein_gaussian(means_y[i, :],
                                                  means_y[j, :],
                                                  covarainces_y[i, :, :],
                                                  covarainces_y[j, :, :])
    for i in range(num_x):
        for j in range(num_x):
            if i != j:
                wd_x[i, j] = wasserstein_gaussian(means_x[i, :],
                                                  means_x[j, :],
                                                  covarainces_x[i, :, :],
                                                  covarainces_x[j, :, :])

    cost1 = wd_y ** 0.5
    cost2 = wd_x ** 0.5
    num1 = num_y
    num2 = num_x
    p_s = np.zeros((num1, 1))
    p_s[:, 0] = weight_y
    p_t = np.zeros((num2, 1))
    p_t[:, 0] = weight_x

    f1_st = np.repeat((cost1 ** 2) @ p_s, num2, axis=1)
    f2_st = np.repeat(((cost2 ** 2) @ p_t).T, num1, axis=0)
    cost_st = f1_st + f2_st

    trans = p_s @ p_t.T
    cost = 0
    for o in range(outer_iter):
        cost = cost_st - 2 * (cost1 @ trans @ cost2.T)
        beta = 0.01 * np.max(np.abs(cost))
        kernel = np.exp(-cost / beta) * trans
        a = np.ones((num1, 1)) / num1
        b = 0
        for i in range(inner_iter):
            b = p_t / (np.matmul(kernel.T, a))
            a = p_s / np.matmul(kernel, b)
        trans = (a @ b.T) * kernel

    # trans = num2 * trans
    fgwd = np.sum(cost * trans)
    # trans /= np.sum(trans, axis=0, keepdims=True)
    w = ridge_regression(trans.T @ means_y, means_x, gamma=0)
    # w = ridge_regression(pts1, trans @ pts2, gamma)
    # print(np.max(trans.T @ pts1), np.min(trans.T @ pts1))
    return w, fgwd



def sinkhorn_iwd(ys, xs, gamma: float=None, max_iter: int=50):
    """
    A gmm-based EM algorithm for fuzzy regression (or called point cloud alignment)
    Reference:
    [1] Myronenko, Andriy, and Xubo Song.
        "Point set registration: Coherent point drift."
        IEEE transactions on pattern analysis and machine intelligence 32.12 (2010): 2262-2275.

    :param ys: (num, 1) labels
    :param xs: (num, dim) features
    :param gamma: the weight of ridge regression, None for pure least-squares estimation
    :param max_iter: the number of EM iterations
    :return:
        w: the (dim, 1) coefficients of model
    """
    num_x, dim = xs.shape
    p_x = np.ones((num_x, 1)) / num_x
    num_y = ys.shape[0]
    p_y = np.ones((num_y, 1)) / num_y
    w = np.zeros((dim, 1))
    trans = p_y @ p_x.T
    for i in range(max_iter):
        cost = euclidean_distance_matrix(ys, xs @ w)
        for o in range(50):
            beta = 0.1 * np.max(np.abs(cost))
            kernel = np.exp(-cost / beta) * trans
            a = np.ones((num_y, 1)) / num_y
            b = 0
            for j in range(10):
                b = p_x / (np.matmul(kernel.T, a))
                a = p_y / np.matmul(kernel, b)
            trans = (a @ b.T) * kernel

        # trans = num_x * trans
        # w = ridge_regression(ys, trans @ xs, gamma)
        w = ridge_regression(trans.T @ ys, xs, gamma)
    return w


# def self_moments_estimator(ys, xs):
#     """
#     The self-moments estimator in the following reference for shuffled linear regression
#     [1] Abid, Abubakar, Ada Poon, and James Zou.
#         "Linear regression with shuffled labels."
#         arXiv preprint arXiv:1705.01342 (2017).
#
#     :param ys: (num, 1) labels
#     :param xs: (num, dim) features
#     :return:
#         w: (dim, 1) the coefficients of model
#     """
#     w = np.zeros((xs.shape[1], 1))
#     w[:, 0] = shuffled_stats.linregress(xs, ys[:, 0], estimator='SM')
#     return w
