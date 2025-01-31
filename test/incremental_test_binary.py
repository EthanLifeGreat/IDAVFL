import time
from algorithm.encoder_matching_incremental_binary_compare import encoder_matching_incremental, print_incremental_set
from args import *

# arguments = BCW_Arguments()
# arguments = DCC_Arguments()
arguments = EPS5_Arguments()

time_start = time.time()

times = 10
arguments.display_test_rec = False
arguments.display_cr = False

results = None
for i in range(times):
    if i == 0:
        results = encoder_matching_incremental(arguments)
    else:
        results += encoder_matching_incremental(arguments)

results /= times
ft = results[0]
dis = results[1]
bic = results[2]
all_in_once = results[3]

print_incremental_set()
print("{} times compare:      Test set           Train set".format(times))
print(
    "                         F-score  Accuracy  F-score  Accuracy")
print(
    "Fine-Tune (Baseline):   {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(ft[0], ft[1], ft[2], ft[3]))
print(
    "Distillation (Ablation):{:.3f}    {:.3f}    {:.3f}    {:.3f}".format(dis[0], dis[1], dis[2], dis[3]))
print(
    "Distillation + BiC:     {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(bic[0], bic[1], bic[2], bic[3]))
print(
    "Using all data once:    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(all_in_once[0], all_in_once[1],
                                                                          all_in_once[2],
                                                                          all_in_once[3]))

time_end = time.time()
print('Time Elapsed:{:.2f} min'.format((time_end - time_start) / 60))

#
# training set 1 size:1750
# training set 2 size:1750
# test set size:1500
# 10 times compare:      Test set           Train set
#                          F-score  Accuracy  F-score  Accuracy
# Fine-Tune (Baseline):   0.473    0.520    0.472    0.516
# Distillation (Ablation):0.540    0.600    0.534    0.592
# Distillation + BiC:     0.542    0.601    0.536    0.592
# Using all data once:    0.594    0.617    0.592    0.611
# Time Elapsed:3.35 min
