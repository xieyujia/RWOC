import numpy as np
import pickle
import matplotlib.pyplot as plt


folder = 'fix_d1/'
files = ['2', '5', '10', '20', '50']
names = [folder + 'results_d2_' + i + '.pkl' for i in files]

# folder = 'fix_d2/'
# files = ['3', '5', '10', '20', '50']
# names = [folder + 'results_d1_' + i + '.pkl' for i in files]

# folder = 'noise/'
# files = ['0', '001', '01', '05', '1']
# names = [folder + 'results_noise_' + i + '.pkl' for i in files]

# folder = 'num/'
# files = ['100', '200', '500', '1000', '5000']
# names = [folder + 'results_n_' + i + '.pkl' for i in files]


methods = ['Oracle', 'LS', 'Stochastic_EM', 'Sliced-GW', 'Sinkhorn-GW', 'Robot']
colors = ['tab:blue', 'tab:red', 'tab:cyan', 'tab:green', 'tab:purple', 'tab:orange']


def load_file(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_data(data):
    def get_arr(arr):
        temp = np.array(arr)
        return np.log10(temp)

    plt.figure()
    num_points = len(data)
    x = np.arange(num_points)
    for idx, m in enumerate(methods):
        # if m == 'Stochastic_EM':
        #     continue

        mean = [get_arr(data[i][m]).mean() for i in range(num_points)]
        std = [get_arr(data[i][m]).std() for i in range(num_points)]
        mean = np.array(mean)
        std = np.array(std)

        plt.plot(x, mean, 's-', label=m, color=colors[idx])
        plt.fill_between(x, mean-std, mean+std, color=colors[idx], alpha=0.2)

    plt.xticks(x, files)
    plt.xlabel(r'$d_2$')
    plt.ylabel(r'$\log_{10}$(error)')
    plt.legend()
    plt.savefig('plot.png', dpi=300)
    plt.show()



def main():
    data = [load_file(name) for name in names]
    plot_data(data)


if __name__ == '__main__':
    main()
