from args import *
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    a = BCW_Arguments()
    # a = EPS5_Arguments()

    plt.figure(figsize=(8, 8))
    sns.heatmap(a.data_frame.corr(), vmin=-1, vmax=1)
    plt.show()

