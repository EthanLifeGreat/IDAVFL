import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
csv_path_list = ['../record/HAR/continuous_incremental/dCI_2CI_15times_CI_result_seed_{}.csv'.format(i)
                 for i in range(1, 11)]
idc = ["Walking, W.U.    ",
       "Walking, W.D.    ",
       "Walking, Sitting ",
       "Walking, Standing",
       "Walking, Laying  ",
       "W.U., W.D.       ",
       "W.U., Sitting    ",
       "W.U., Standing   ",
       "W.U., Laying     ",
       "W.D., Sitting    ",
       "W.D., Standing   ",
       "W.D., Laying     ",
       "Sitting, Standing",
       "Sitting, Laying  ",
       "Standing, Laying ",
       "Overall          "
       ]
diffs = []
hue_df = pd.DataFrame(columns=["Name of Incremental Classes", "Method", "Macro F1-score"])
for csv_path in csv_path_list:
    a = np.loadtxt(csv_path, delimiter=',')
    diff = a[:, 2] - a[:, 1]
    diff = diff.reshape(16, 1)
    diffs.append(diff)
    hue_df = hue_df.append(pd.DataFrame([[idc[i], "without  NAC", a[:, 1][i]] for i in range(16)],
                                        columns=["Name of Incremental Classes", "Method", "Macro F1-score"]))
    hue_df = hue_df.append(pd.DataFrame([[idc[i], "with NAC", a[:, 2][i]] for i in range(16)],
                                        columns=["Name of Incremental Classes", "Method", "Macro F1-score"]))

diffs = np.concatenate(diffs, axis=1)
mean = np.mean(diffs, axis=1)
var = np.std(diffs, axis=1)

# show_table = np.concatenate([range(16), mean, var], axis=1)
print("index\t\t\t\tmean\tstd-dev")
for i in range(16):
    print("{}\t{:.3f}\t{:.3f}".format(idc[i], mean[i], var[i]))


# plt.figure(dpi=150)
# df = pd.DataFrame(diffs.T, columns=idc)
# sns.boxplot(data=df)
# plt.xticks(rotation=45, fontsize=6)
# plt.xticks(fontsize=6)
# plt.axhline(0, color='gray')
# plt.xlabel('Name of Incremental Classes', fontsize=8)
# plt.ylabel('F1 increase bought by NAC', fontsize=8)
# plt.tight_layout()
# plt.savefig('../plt.pdf')
# plt.show()

plt.figure(dpi=150, figsize=(20, 10))
sns.boxplot(data=hue_df, x="Name of Incremental Classes", width=0.5,
            y="Macro F1-score", hue="Method", palette=["#53accd", "#ea96a3"])
plt.xticks(rotation=10, fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('Name of Incremental Classes', fontsize=12)
plt.ylabel('Macro F1-score', fontsize=12)
plt.tight_layout()
plt.savefig('../plt.pdf')
plt.show()
