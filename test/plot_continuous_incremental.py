import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot(results, title, loc=4, bbox=None):
    times = results.shape[1]
    # plt.style.use('ggplot')
    x_axis = np.arange(times)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x_axis, results[3], color='gold', marker='s', label='Retrain-all', ms=8, lw=3)
    plt.plot(x_axis, results[2], color='orangered', marker='^', label='IDAVFL (Ours)', ms=8, lw=3)
    plt.plot(x_axis, results[1], color='royalblue', marker='o', label='Retrain-new', ms=8, lw=3)
    plt.plot(x_axis, results[0], color='purple', marker='v', label='Fine-tuning', ms=8, lw=3)
    plt.grid(True)
    plt.legend(fontsize='x-large', loc=loc, bbox_to_anchor=bbox)
    plt.ylabel("Macro F1", fontsize='x-large')
    plt.xlabel("Timestamp", fontsize='x-large')
    # plt.title(title, fontsize='xx-large')
    plt.savefig('./plt.pdf')
    print(results)
    plt.show()


def plot_multi(results, title, loc=4, bbox=None):
    times = results.shape[1]
    # plt.style.use('ggplot')
    x_axis = np.arange(times)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x_axis, results[3], color='gold', marker='s', label='Retrain-all', ms=8, lw=3)
    plt.plot(x_axis, results[2], color='orangered', marker='^', label='IDAVFL (Ours)', ms=8, lw=3)
    plt.plot(x_axis, results[1], color='royalblue', marker='o', label='Retrain-new', ms=8, lw=3)
    plt.plot(x_axis, results[0], color='purple', marker='v', label='Fine-tuning', ms=8, lw=3)
    plt.grid(True)
    plt.legend(fontsize='x-large', loc=loc, bbox_to_anchor=bbox)
    plt.ylabel("Macro F1", fontsize='x-large')
    plt.xlabel("Timestamp", fontsize='x-large')
    plt.title(title, fontsize='xx-large')
    plt.show()

def plot_multi_CI(results, title, loc=4, bbox=None):
    times = results.shape[1]
    # plt.style.use('ggplot')
    x_axis = np.arange(times)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x_axis, results[3], color='gold', marker='s', label='Using All', ms=8, lw=3)
    plt.plot(x_axis, results[2], color='orangered', marker='^', label='Distillation+BiC', ms=8, lw=3)
    plt.plot(x_axis, results[1], color='purple', marker='v', label='Distillation', ms=8, lw=3)
    plt.plot(x_axis, results[0], color='royalblue', marker='o', label='Retrain', ms=8, lw=3)
    plt.grid(True)
    plt.legend(fontsize='x-large', loc=loc, bbox_to_anchor=bbox)
    plt.ylabel("Macro F1", fontsize='x-large')
    plt.xlabel("Timestamp", fontsize='x-large')
    plt.title(title, fontsize='xx-large')
    plt.show()

def plot_multi_dCI(results, title, loc=4, bbox=None):
    times = results.shape[1]
    # plt.style.use('ggplot')
    x_axis = np.arange(times)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x_axis, results[3], color='gold', marker='s', label='Using All', ms=8, lw=3)
    plt.plot(x_axis, results[2], color='orangered', marker='^', label='Distillation+BiC', ms=8, lw=3)
    plt.plot(x_axis, results[1], color='purple', marker='v', label='Distillation', ms=8, lw=3)
    plt.plot(x_axis, results[0], color='royalblue', marker='o', label='Retrain', ms=8, lw=3)
    plt.grid(True)
    plt.legend(fontsize='x-large', loc=loc, bbox_to_anchor=bbox)
    plt.ylabel("Macro F1", fontsize='x-large')
    plt.xlabel("Timestamp", fontsize='x-large')
    plt.title(title, fontsize='xx-large')
    plt.show()


def plot_multi_sCI(results, title, loc=4, bbox=None):
    # plt.style.use('ggplot')
    size = results.shape[1]
    x = np.arange(size)  # x轴刻度标签位置
    width = 0.2  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    ax = plt.subplot()
    ax.boxplot(results)
    # plt.bar(x - 1.5 * width, results[0], width, color='royalblue', label='Retrain')
    # plt.bar(x - 0.5 * width, results[1], width, color='purple', label='Distillation')
    # plt.bar(x + 0.5 * width, results[2], width, color='orangered', label='Distillation+BiC')
    # plt.bar(x + 1.5 * width, results[3], width, color='gold', label='Using All')
    ax.set_xticklabels(['Retrain', 'Distillation', 'Distillation+BiC', 'Using All'])
    plt.title(title, fontsize='xx-large')
    plt.legend(fontsize='x-large', loc=loc, bbox_to_anchor=bbox)
    plt.show()


if __name__ == '__main__':

    choice = 3

    if choice == 1:
        bo = 4
        results = np.loadtxt('../record/EPS5/continuous_incremental/random/seed{}_rs{}_lam0.95.txt'.format(bo, bo))
        plot(results, 'EPS5k: Random', 3)
    elif choice == 1.1:
        # even
        bo = 80
        # results = np.loadtxt('../record/EPS5/continuous_incremental/even/seed{}_rs{}_lam0.95.txt'.format(bo, bo))
        results = np.loadtxt('../record/EPS5/continuous_incremental/even/aver.txt')
        plot(results, 'EPS5k: Even', 2)
    elif choice == 1.2:
        # bi-incline
        bo = 10
        results = np.loadtxt('../record/EPS5/continuous_incremental/bi-incline/seed{}_rs{}_lam0.95.txt'.format(bo, bo))
        plot(results, 'EPS5k: Incline-Decline', loc=8)
    elif choice == 1.3:
        # Incline
        bo = 9
        results = np.loadtxt('../record/EPS5/continuous_incremental/incline/seed{}_rs{}_lam0.95.txt'.format(bo, bo))
        plot(results, 'EPS5k: Incline', loc=3)
    elif choice == 2:
        # Random
        results = np.loadtxt('../record/DCC/continuous_incremental/random/aver_seed9-14_lam0.95.csv', delimiter=',')
        plot(results, 'DCC: Random', 3)
    elif choice == 3:
        # Random
        results = np.loadtxt('../record/BCW/continuous_incremental/random/a_copy_of_aver_seed9-14_lam0.95.csv', delimiter=',')
        plot(results, 'BCW: Random', 3)
    elif choice == 4:
        results = np.loadtxt('../record/HAR/continuous_incremental/aver_seed10_rs10_lam0.95.csv',
                             delimiter=',')
        plot_multi(results, 'HAR', 3)
    elif choice == 5:
        results = np.loadtxt('../record/HAR/continuous_incremental/sCI_aver_seed5_rs5_lam0.67.csv',
                             delimiter=',')
        plot_multi_sCI(results, 'HAR')

    print()
