import numpy as np
import matplotlib.pyplot as plt


def plot(data, title, loc=4, bbox=None):
    # plt.style.use('ggplot')
    x_axis = np.arange(data.shape[1])
    plt.plot(x_axis, data[1], color='gold', marker='^', label='Precision', ms=8, lw=3)
    plt.plot(x_axis, data[2], color='orangered', marker='v', label='Recall', ms=8, lw=3)
    plt.plot(x_axis, data[0], color='purple', marker='s', label='Macro-F1', ms=8, lw=3)
    # plt.plot(x_axis, results[0], color='royalblue', marker='o', label='Re-Train', ms=8, lw=3)
    plt.grid(True)
    x_ticks = ['1', '2', '4', '6', '8', '10']
    plt.xticks(x_axis, x_ticks)
    plt.legend(fontsize='x-large', loc=loc, bbox_to_anchor=bbox)
    plt.xlabel("Number of Passive Parties", fontsize='x-large')
    plt.ylim([0, 0.8])
    plt.title(title, fontsize='xx-large')
    plt.show()


if __name__ == '__main__':
    f_name = '../record/EPS5/5_times_5+1_multi-passive_parties_results.txt'

    results = np.loadtxt(f_name)

    results = results.T

    plot(results, 'EPS5k: Multi-Passive Party Results', loc=4)
