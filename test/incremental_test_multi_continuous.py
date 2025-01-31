import copy
import random
import time

from algorithm.encoder_matching import *
from sklearn.model_selection import train_test_split
from algorithm.encoder_matching_incremental import classifier_distillation_training, bic_training, \
    encoder_matching_test_with_bic, classifier_fine_tune, classifier_distillation_training_CI
from utils.utils import tensor_scale
from torch.utils.data import DataLoader
from plot_continuous_incremental import plot_multi

from args import HAR_Arguments
from sklearn.utils import shuffle


def setup_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True


ta = 1
bo = 0
seed = bo
rs = bo
lam = 0.95
times = 7
setup_seed(seed)

arguments = HAR_Arguments()
arguments.display_test_rec = False
arguments.display_cr = False
fine_tune_scale = 1e-1
num_methods = 4  # Re-train, Fine-tune, Distillation, Joint-training


class ContinuousDataLoaders:
    def __init__(self):
        x_train = np.loadtxt("data/UCI HAR Dataset/train/X_train.txt", dtype=np.float32)
        y_train = np.loadtxt("data/UCI HAR Dataset/train/y_train.txt", dtype=np.int8) - 1
        x_test = np.loadtxt("data/UCI HAR Dataset/test/X_test.txt", dtype=np.float32)
        y_test = np.loadtxt("data/UCI HAR Dataset/test/y_test.txt", dtype=np.int8) - 1
        n_tr = x_train.shape[0]
        n_te = x_test.shape[0]
        x_train = np.hstack([np.arange(1, n_tr + 1).reshape([-1, 1]), x_train])
        x_test = np.hstack([np.arange(1, n_te + 1).reshape([-1, 1]), x_test])
        if arguments.data_scale:
            x_train[:, arguments.train_idc] = \
                tensor_scale(x_train[:, arguments.train_idc], arguments.data_scale_method)
            x_test[:, arguments.train_idc] = \
                tensor_scale(x_test[:, arguments.train_idc], arguments.data_scale_method)

        # class sort
        idc = np.argsort(y_train)
        x_train = x_train[idc]
        y_train = y_train[idc]

        self.test_loader = self.data_loader(x_test, y_test)

        # split train into 6 classes
        train_data_each_class = []
        num_train_data_each_class = []
        rest_x, rest_y = x_train, y_train
        for class_no in range(arguments.num_classes-1):
            n = np.sum(y_train == class_no)
            num_train_data_each_class.append(n)
            this_class_x, rest_x, this_class_y, rest_y = \
                train_test_split(rest_x, rest_y, train_size=n, shuffle=False)
            train_data_each_class.append([this_class_x, this_class_y])
        num_train_data_each_class.append(rest_y.shape[0])
        train_data_each_class.append([rest_x, rest_y])
        self.train_data_each_class = train_data_each_class

        # the fraction for each class to give at each time stamp
        self.num_give_out_data_each_time_each_class = []
        for j in range(arguments.num_classes):
            """
                3:1:  :2:   :1, where '2' is at (j+1)-th position.
            """
            frac = [3] + [1 for _ in range(j)] + [2] + [1 for _ in range(arguments.num_classes-j-1)]
            frac = [0] + frac
            frac = np.array(frac)
            frac = frac / np.sum(frac)
            accumulate_frac = np.cumsum(frac)
            num_give_out_data_each_time = np.rint(accumulate_frac * num_train_data_each_class[j])
            self.num_give_out_data_each_time_each_class.append(num_give_out_data_each_time)

        self.all_seen_data_x_list, self.all_seen_data_y_list = [], []
        self.t = 0

    def next_loader(self):
        data_x_list = []
        data_y_list = []
        for j in range(arguments.num_classes):
            data_x, data_y = self.train_data_each_class[j]
            data_idx = (int(self.num_give_out_data_each_time_each_class[j][self.t]),
                        int(self.num_give_out_data_each_time_each_class[j][self.t+1]))
            data_x = data_x[data_idx[0]:data_idx[1]]
            data_y = data_y[data_idx[0]:data_idx[1]]
            data_x_list.append(data_x.copy())
            data_y_list.append(data_y.copy())
        x_data = np.vstack(data_x_list)
        y_data = np.hstack(data_y_list)
        ntl = self.data_loader(x_data, y_data)
        self.all_seen_data_x_list.append(x_data)
        self.all_seen_data_y_list.append(y_data)
        asl = self.data_loader(np.vstack(self.all_seen_data_x_list), np.hstack(self.all_seen_data_y_list))

        self.t += 1
        return ntl, asl

    def this_loader(self):
        t = self.t - 1
        data_x_list = []
        data_y_list = []
        for j in range(arguments.num_classes):
            data_x, data_y = self.train_data_each_class[j]
            data_idx = (int(self.num_give_out_data_each_time_each_class[j][t]),
                        int(self.num_give_out_data_each_time_each_class[j][t+1]))
            data_x = data_x[data_idx[0]:data_idx[1]]
            data_y = data_y[data_idx[0]:data_idx[1]]
            data_x_list.append(data_x.copy())
            data_y_list.append(data_y.copy())
        x_data = np.vstack(data_x_list)
        y_data = np.hstack(data_y_list)
        ttl = self.data_loader(x_data, y_data)

        return ttl

    @staticmethod
    def data_loader(x_data, y_data):
        x_data, y_data = shuffle(x_data, y_data, random_state=rs)
        x_data = torch.from_numpy(x_data).to(torch.float32)
        y_data = torch.from_numpy(y_data).type(torch.LongTensor)
        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=arguments.batch_size,
                                 shuffle=False)

        return data_loader


time_start = time.time()

aver_fs, aver_results = 0, 0

for k in range(ta):
    dataLoaderGenerator = ContinuousDataLoaders()
    next_time_loader, all_seen_loader = dataLoaderGenerator.next_loader()
    test_loader = dataLoaderGenerator.test_loader

    encoder_matching_train(next_time_loader, arguments)
    teacher_classifier = copy.deepcopy(arguments.classifier)
    ft_classifier = copy.deepcopy(arguments.classifier)
    res0 = encoder_matching_test_continuous(test_loader, arguments, binary=False)

    fs = np.hstack([res0[0] for _ in range(num_methods)])
    results = np.hstack([res0 for _ in range(num_methods)])

    time_spans = np.array([0 for _ in range(num_methods)], 'float32')

    for t in range(times-1):
        next_time_loader, all_seen_loader = dataLoaderGenerator.next_loader()
        train_loader = dataLoaderGenerator.this_loader()

        # Fine-tuning
        s = time.time()
        arguments.classifier = classifier_fine_tune(ft_classifier, train_loader, arguments, scale=fine_tune_scale)
        time_spans[1] += time.time() - s
        results_ft = encoder_matching_test_continuous(test_loader, arguments, binary=False)
        ft_classifier = copy.deepcopy(arguments.classifier)

        # Re-train
        train_loader = dataLoaderGenerator.this_loader()
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V,
                                                   train_loader, arguments)
        time_spans[0] += time.time() - s
        results_rt = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        # Distillation: Ours
        train_loader = dataLoaderGenerator.this_loader()
        s = time.time()
        # arguments.classifier = classifier_distillation_training(
        #     teacher_classifier, 6, next_time_loader, arguments, lam=lam)
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V,
                                                   train_loader, arguments)
        time_spans[2] += time.time() - s
        results_dis = encoder_matching_test_continuous(test_loader, arguments, binary=False)
        # teacher_classifier = copy.deepcopy(arguments.classifier)

        # All in once
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V,
                                                   all_seen_loader, arguments)
        time_spans[3] += time.time() - s
        results_aio = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        fs = np.vstack([fs, np.hstack([res[0] for res in [results_ft, results_rt, results_dis, results_aio]])])
        results = np.vstack([results, np.hstack([results_ft, results_rt, results_dis, results_aio])])

    fs = np.array(fs).T
    time_end = time.time()
    time_spans /= (times - 1)
    print("Average Time Used:" + str(time_spans))

    np.savetxt('{}continuous_incremental/seed{}_rs{}_lam{}.csv'.format(arguments.rec_path, seed, rs, lam), fs,
               fmt='%.3f', delimiter=',')
    np.savetxt('{}continuous_incremental/seed{}_rs{}_lam{}_detail.csv'.format(arguments.rec_path, seed, rs, lam),
               results,
               fmt='%.3f', delimiter=',')
    print('Time Elapsed:{:.2f} min'.format((time_end - time_start) / 60))
    aver_fs += fs
    aver_results += results
    seed += 1
    rs += 1
    setup_seed(seed)
aver_fs /= ta
aver_results /= ta
np.savetxt('{}continuous_incremental/aver_seed{}_rs{}_lam{}.csv'.format(arguments.rec_path, seed, rs, lam), aver_fs,
           fmt='%.3f', delimiter=',')
np.savetxt('{}continuous_incremental/aver_seed{}_rs{}_lam{}_detail.csv'.format(arguments.rec_path, seed, rs, lam),
           aver_results,
           fmt='%.3f', delimiter=',')

plot_multi(aver_fs, 'HAR', None)

print(aver_fs)
