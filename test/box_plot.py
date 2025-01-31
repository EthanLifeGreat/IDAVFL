import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data_name = "HAR"
data_name = "EPS5"
# data_name = "DCC"
# data_name = "BCW"

a = np.array(pd.read_csv("../record/{}/metrics.csv".format(data_name), index_col=0))
b = np.array(pd.read_csv("../FATE_record/{}/hetero_nn_results.csv".format(data_name), index_col=None, header=None))
c = np.array(pd.read_csv("../FATE_record/{}/sbt_results.csv".format(data_name), index_col=None, header=None))
d = np.concatenate((b, c), 1)
name_list = ['Macro F1-score', 'Macro Precision', 'Macro Recall', 'Accuracy']
plt.figure(figsize=(24, 8))
model_names = ["Baseline\nLow", "Baseline\nLow SAE", "Ours", "Baseline\nHigh SAE",
               "Baseline\nHigh", "Hetero-\nNN", "Secure\nBoost"]
for i in range(4):
    at = a[:, range(i, 20+i, 4)]
    at = np.concatenate((at, d[:, range(i, 8+i, 4)]), 1)
    ax = plt.subplot(141+i)
    ax.boxplot(at)
    plt.title(name_list[i])
    ax.set_xticklabels(model_names)

plt.suptitle("Prediction result of {}-folds on dataset {}".format(a.shape[0], data_name))
plt.show()
