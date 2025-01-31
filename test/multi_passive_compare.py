import time

from algorithm.encoder_matching_multi_passive import encoder_matching_train, encoder_matching_test
from args import *
from utils import utils

arguments = EPS5_Arguments()

n_party_B_list = [1, 2, 4, 6, 8, 10]

times = 1


def get_idc_list(n, size, all_idc):
    all_idc = list(all_idc)
    choices = []
    q, r = divmod(size, n)
    for i in range(n):
        if i < r:
            choices.append(all_idc[(q+1) * i:(q + 1) * (i + 1)])
        else:
            choices.append(all_idc[r+q*i:r+q*(i+1)])

    return choices


time_start = time.time()
aver_results = 0
for _ in range(times):
    results = []
    for n_passives in n_party_B_list:
        arguments.n_passives = n_passives
        arguments.party_B_list = get_idc_list(n_passives, arguments.input_B_size, arguments.party_B_idc)
        train_dataset, test_dataset = utils.datasets(arguments)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=arguments.batch_size,
                                                   shuffle=arguments.train_loader_shuffle)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset),
                                                  shuffle=False)
        encoder_matching_train(train_loader, arguments)
        result = encoder_matching_test(test_loader, arguments)
        results.append(result)
    results = np.array(results)
    aver_results += results
aver_results /= times
time_end = time.time()
print('Time Elapsed:{:.2f} min'.format((time_end - time_start) / 60))
print(aver_results)

np.savetxt(arguments.rec_path + "{}_times_{}+1_multi-passive_parties_results.txt".format(times, len(n_party_B_list)-1),
           aver_results)
