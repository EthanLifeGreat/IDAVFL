import numpy as np
import matplotlib.pyplot as plt

# time = [0.4852022, 0.4906988, 0.6264487, 2.0329192]
# names = ["Fine-Tune", "Retrain-new", "IDAVFL(Ours)", "Retrain-all"]
# fig_name = './IDAVFL-time.pdf'

time = [1.9459132, 2.1738257, 2.1830406, 5.0072947]
names = ["Retrain-new", "Distillation", "Distillation+BiC", "Retrain-all"]
fig_name = './CI-time.pdf'


f, ax = plt.subplots(figsize=(8, 6))
# plt.ylim([0, 2.5])
plt.ylim([0, 6])
bar = plt.bar(np.arange(len(time)), time, tick_label=names, width=0.65)
bar[0].set_color('purple')  # A6CEE3
bar[1].set_color('royalblue')  # FDBF6F
bar[2].set_color('orangered')  # CAB2D6
bar[3].set_color('gold')  # FFFF99
plt.tick_params(labelsize=16)
# plt.title('Average time cost of different algorithms', fontsize='xx-large')
for x, y in enumerate(time):
    plt.text(x - 0.112, y + 0.01, "%.3f" % y)
plt.xlabel('Method', fontsize=22)
plt.ylabel('Time (s)', fontsize=22)
plt.savefig(fig_name)

plt.show()
