import argparse
import copy
import numpy as np
import pickle
import time
import torch
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

import Methods

from Robot import Robot
from Stochastic_EM import stochastic_em


""" trail parameters """
gamma = 0.0  # ridge regression parameter

num_repeat = 10
seeds = np.arange(num_repeat) + 1926

methods = ['Oracle', 'LS', 'Stochastic_EM', 'Sliced-GW', 'Sinkhorn-GW', 'Robot']


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


def one_trail(result, args, repeat_num):
    # train and val: (x1, x2, y); unshuffled: (x, y, w)
    train_data, val_data, unshuffled_data = generate_data(args)
    x1, x2, y = train_data[0], train_data[1], train_data[2]
    # x1_val, x2_val, y_val = val_data[0], val_data[1], val_data[2]

    # oracle least squares
    start = time.time()
    w, _ = Methods.ordinary_least_squares(unshuffled_data[1], unshuffled_data[0], gamma)
    err = eval(unshuffled_data[0], None, unshuffled_data[1], w, single_x=True)
    result['Oracle'].append(err)
    elapse = time.time() - start
    print('Oracle:\t\t%.5e, elapse: %.3fs' % (err, elapse))

    # ordinary least squares
    start = time.time()
    x = np.concatenate((x1, x2), axis=1)
    w, _ = Methods.ordinary_least_squares(y, x, gamma)
    err = eval(x1, x2, y, w)
    result['LS'].append(err)
    elapse = time.time() - start
    print('Least Squares:\t%.5e, elapse: %.3fs' % (err, elapse))

    # stochastic-EM
    start = time.time()
    _, err = stochastic_em(x2, y, steps=15)
    elapse = time.time() - start
    result['Stochastic_EM'].append(err)
    print('Stochastic EM:\t%.5e, elapse: %.3fs' % (err, elapse))

    # sliced GW, align x2 with y
    start = time.time()
    w, trans, _ = Methods.sliced_fgwd(x2, y, x1, num_samples=x2.shape[0],
        alpha=0.1, num_trials=100, num_projs=args.max_baseline_iter, gamma=gamma)
    err = eval(trans.T @ x2, x1, y, w)
    result['Sliced-GW'].append(err)
    elapse = time.time() - start
    print('Sliced-GW:\t%.5e, elapse: %.3fs' % (err, elapse))

    # sinkhorn GW, align x2 with y
    start = time.time()
    w, trans, _ = Methods.sinkhorn_fgwd(x2, y, x1,
        outer_iter=100, inner_iter=args.max_baseline_iter, gamma=gamma)
    err = eval(trans.T @ x2, x1, y, w)
    result['Sinkhorn-GW'].append(err)
    elapse = time.time() - start
    print('Sinkhorn-GW:\t%.5e, elapse: %.3fs' % (err, elapse))

    # Robot
    print('Running Robot...')
    start = time.time()
    Robot_solver = Robot(args)
    Robot_solver.train(train_data)
    err = Robot_solver.eval(train_data)
    result['Robot'].append(err)
    elapse = time.time() - start
    print('Robot:\t\t%.5e, elapse: %.3fs' % (err, elapse))


def trail(args):
    result = {}
    for m in methods:
        result[m] = []

    for it in range(num_repeat):
        print('Trail %d' % (it + 1))
        np.random.seed(seeds[it])
        torch.manual_seed(seeds[it])
        one_trail(result, args, it)
        print('\n')

    for name in methods:
        if len(result[name]) == 0:
            continue
        arr = np.array(result[name])
        print('%s\t: mean=%.5e,\tstd=%.5e' % (name, arr.mean(), arr.std()))

    return result


def generate_data(args):

    n, d1, d2 = 2*args.n, args.d1, args.d2

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

    x_train = x[:args.n, :]
    x_val = x[args.n:, :]

    y_train = y[:args.n]
    y_val = y[args.n:]
    index = np.random.permutation(args.n)

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


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    """ parameters for data generation """
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

    """ parameters for Robot """
    parser.add_argument('--method', type=str, default='sinkhorn_robust',
                        help='sinkhorn_naive | sinkhorn_stablized | sinkhorn_manual | sinkhorn_robust')
    parser.add_argument('--train_iter', type=int, default=100,
                        help='total number of traning steps')
    parser.add_argument('--max_inner_iter', type=int, default=100,
                        help='inner iteration number, used for Sinkhorn')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size')
    parser.add_argument('--lr_R', type=float, default=5e-5,
                        help='learning rate for regression')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                        help='entropy regularization coefficient, used for Sinkhorn')
    parser.add_argument('--rho1', type=float, default=0.1,
                        help='relaxition for the first marginal')
    parser.add_argument('--rho2', type=float, default=0.1,
                        help='relaxition for the second marginal')
    parser.add_argument('--eta', type=float, default=1e-3,
                        help='grad for projected gradient descent for robust OT')

    """ parameters for baseline methods """
    parser.add_argument('--max_baseline_iter', type=int, default=10,
                        help='maximum inner iteration of baseline methods')

    args = parser.parse_args()

    result = trail(args)
    path = 'results/results.pkl'
    pickle.dump(result, open(path, 'wb'))


if __name__ == '__main__':
    main()
