import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


name = 'IDVFL'


def plot(results, title, loc=4, bbox=None):
    times = results.shape[1]
    # plt.style.use('ggplot')
    x_axis = np.arange(times)
    plt.figure(time.time())
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x_axis, results[3], color='gold', marker='s', label='Using All', ms=8, lw=3)
    plt.plot(x_axis, results[2], color='orangered', marker='^', label=name, ms=8, lw=3)
    plt.plot(x_axis, results[1], color='royalblue', marker='o', label='Retrain', ms=8, lw=3)
    plt.plot(x_axis, results[0], color='purple', marker='v', label='Fine-tuning', ms=8, lw=3)
    plt.grid(True)
    plt.legend(fontsize='x-large', loc=loc, bbox_to_anchor=bbox)
    plt.ylabel("Macro F1", fontsize='x-large')
    plt.xlabel("Timestamp", fontsize='x-large')
    plt.title(title, fontsize='xx-large')
    print(results)
    plt.savefig('{}.pdf'.format(title[:3]))
    plt.show()


if __name__ == '__main__':
    results = np.loadtxt("D:\\1.学术\\3.联邦学习\\7.20240414\\BCW random pic.csv", delimiter=',')
    plot(results, 'BCW: Random', 3)
    results = np.loadtxt("D:\\1.学术\\3.联邦学习\\7.20240414\\DCC random pic.csv", delimiter=',')
    plot(results, 'DCC: Random', 3)

    print()
