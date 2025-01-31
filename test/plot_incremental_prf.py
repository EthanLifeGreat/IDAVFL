import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_f1_macro(results, title, loc=4, bbox=None):
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
    print(results)
    plt.show()


def plot_prf(results, loc=4, bbox=None):
    times = results.shape[0]
    # plt.style.use('ggplot')
    metrics = ['Macro F1',
               'Precision of Positive Samples',
               'Recall of Positive Samples',
               'F1-score of Positive Samples',
               'Precision of Negative Samples',
               'Recall of Negative Samples',
               'F1-score of Negative Samples']
    for i in range(len(metrics)):
        plt.figure(str(time.time()))
        metric = metrics[i]
        x_axis = np.arange(times)
        result = results[:, i:28 + i:7].T
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(x_axis, result[3], color='gold', marker='s', label='Retrain-all', ms=8, lw=3)
        plt.plot(x_axis, result[2], color='orangered', marker='^', label='IDAVFL (Ours)', ms=8, lw=3)
        plt.plot(x_axis, result[1], color='royalblue', marker='o', label='Retrain-new', ms=8, lw=3)
        plt.plot(x_axis, result[0], color='purple', marker='v', label='Fine-tuning', ms=8, lw=3)
        plt.grid(True)
        loc = 8 if (i == 5) or (i == 6) else 3
        plt.legend(fontsize='x-large', loc=loc, bbox_to_anchor=bbox)
        plt.ylabel(metric, fontsize='x-large')
        plt.xlabel("Timestamp", fontsize='x-large')
        print(result)
        plt.savefig('../record/DCC/continuous_incremental/random/{}.pdf'.format(metric))
        plt.show()


if __name__ == '__main__':
    # results_f1ma = np.loadtxt('../record/DCC/continuous_incremental/random/a_copy_of_aver_seed9-14_lam0.95.csv',
    #                           delimiter=',')
    # plot_f1_macro(results_f1ma, 'DCC: Random', 3)

    results_prf = np.loadtxt('../record/DCC/continuous_incremental/random/a_copy_of_aver_seed9-14_lam0.95_detail.csv',
                             delimiter=',')
    plot_prf(results_prf, 3)
