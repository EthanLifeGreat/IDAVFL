# deprecated
import copy
import random
import time

import numpy as np

from algorithm.encoder_matching import *
from sklearn.model_selection import train_test_split
from algorithm.encoder_matching_incremental import classifier_distillation_training, bic_training, \
    encoder_matching_test_with_bic, classifier_fine_tune, classifier_distillation_training_CI
from utils.utils import tensor_scale
from torch.utils.data import DataLoader
from plot_continuous_incremental import plot_multi_CI

from args import HAR_Arguments
from sklearn.utils import shuffle


def setup_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True


ta = 5
bo = 5
seed = bo
rs = bo
lam = None
num_exemplar = 300
num_first_classes = 2
num_all_classes = 6
times = num_all_classes - num_first_classes + 1
setup_seed(seed)
val_frac = 0.05

arguments = HAR_Arguments()
arguments.display_test_rec = False
arguments.display_cr = False
fine_tune_scale = 1e-1
num_methods = 4  # Re-train, distill, distill+BiC, Joint-training


class ContinuousDataLoaders:
    def __init__(self, num_first_classes, test=True):
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
        idc = np.argsort(y_test)
        x_test = x_test[idc]
        y_test = y_test[idc]
        n_tr1 = np.where(y_train == num_first_classes)[0][0]

        x_first, x_rest, y_first, y_rest = train_test_split(x_train, y_train, train_size=int(n_tr1), shuffle=False)
        x_first_tr, x_first_va, y_first_tr, y_first_va = train_test_split(x_first, y_first, test_size=val_frac,
                                                                          random_state=seed, shuffle=True,
                                                                          stratify=y_first)
        _, x_train_old_exp, _, y_train_old_exp = train_test_split(x_first_tr, y_first_tr,
                                                                  test_size=num_exemplar, stratify=y_first_tr,
                                                                  random_state=seed, shuffle=True)
        self.exp_cache = [x_train_old_exp, y_train_old_exp]
        self.val_cache = [x_first_va, y_first_va]
        self.all_cache = [x_first, y_first]
        self.rest_cache = [x_rest, y_rest]
        self.test_cache = [x_test, y_test]
        self.train_loader = self.data_loader(x_first_tr, y_first_tr)
        n_te = np.where(self.test_cache[1] == num_first_classes)[0][0]
        self.test_loader = self.data_loader(self.test_cache[0][:n_te], self.test_cache[1][:n_te])
        self.all_loader = None
        self.val_loader = None
        self.t = num_first_classes

    def next_loader(self):
        self.t += 1
        assert self.t <= num_all_classes
        if self.t == num_all_classes:
            x_first, x_rest, y_first, y_rest = self.rest_cache[0], None, self.rest_cache[1], None
            self.test_loader = self.data_loader(self.test_cache[0], self.test_cache[1])
        else:
            n_next = np.where(self.rest_cache[1] == self.t)[0][0]
            x_first, x_rest, y_first, y_rest = train_test_split(self.rest_cache[0], self.rest_cache[1],
                                                                train_size=int(n_next),
                                                                shuffle=False)
            # get test
            n_te = np.where(self.test_cache[1] == self.t)[0][0]
            self.test_loader = self.data_loader(self.test_cache[0][:n_te], self.test_cache[1][:n_te])
        self.rest_cache = [x_rest, y_rest]

        # get all loader
        self.all_cache[0] = np.concatenate([self.all_cache[0], x_first], axis=0)
        self.all_cache[1] = np.concatenate([self.all_cache[1], y_first], axis=0)
        self.all_loader = self.data_loader(self.all_cache[0], self.all_cache[1])

        # get new train
        x_first_tr, x_first_va, y_first_tr, y_first_va = train_test_split(x_first, y_first, test_size=val_frac,
                                                                          random_state=seed, shuffle=True,
                                                                          stratify=y_first)
        x_train = np.concatenate([self.exp_cache[0], x_first_tr], axis=0)
        y_train = np.concatenate([self.exp_cache[1], y_first_tr], axis=0)
        self.train_loader = self.data_loader(x_train, y_train)

        # renew exemplar
        sub_size = int(num_exemplar / self.t)
        _, x_train_old_exp, _, y_train_old_exp = train_test_split(x_first_tr, y_first_tr,
                                                                  test_size=sub_size,
                                                                  random_state=seed, shuffle=True)
        x_exp, _, y_exp, _ = train_test_split(self.exp_cache[0], self.exp_cache[1], test_size=sub_size,
                                              random_state=seed, shuffle=True, stratify=self.exp_cache[1])
        # shuffle(self.exp_cache[0], self.exp_cache[1], random_state=seed)

        x_exp = np.concatenate([x_exp, x_train_old_exp], axis=0)
        y_exp = np.concatenate([y_exp, y_train_old_exp], axis=0)
        self.exp_cache[0], self.exp_cache[1] = x_exp, y_exp

        # add validation
        self.val_cache[0] = np.concatenate([self.val_cache[0], x_first_va], axis=0)
        self.val_cache[1] = np.concatenate([self.val_cache[1], y_first_va], axis=0)
        self.val_loader = self.data_loader(self.val_cache[0], self.val_cache[1])

        return self.train_loader, self.val_loader, self.all_loader, self.test_loader

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
    dataLoaderGenerator = ContinuousDataLoaders(num_first_classes)

    arguments.num_classes = 2
    encoder_matching_train(dataLoaderGenerator.train_loader, arguments)
    teacher_classifier = copy.deepcopy(arguments.classifier)
    res0 = encoder_matching_test_continuous(dataLoaderGenerator.test_loader, arguments, binary=False)
    fs = np.hstack([res0[0] for _ in range(num_methods)])
    results = np.hstack([res0[:3], np.full(times - 1, np.nan), res0[-3:], np.full(times - 1, np.nan)])
    results = np.hstack([results for _ in range(num_methods)])
    time_spans = np.array([0, 0, 0, 0], 'float32')

    for t in range(4):
        train_loader, val_loader, all_loader, test_loader = dataLoaderGenerator.next_loader()
        arguments.num_classes = t + 3

        # Re-train
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V, train_loader,
                                                   arguments)
        time_spans[0] += time.time() - s
        res_rt = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        # Distillation
        s = time.time()
        # lam = np.power((i + num_first_classes)/(i + num_first_classes + 1), 1/2)
        lam = (t + num_first_classes) / (t + num_first_classes + 1)
        arguments.classifier = classifier_distillation_training_CI(teacher_classifier, t + num_first_classes,
                                                                   train_loader,
                                                                   arguments, lam=lam)
        time_spans[1] += time.time() - s
        res_ds = encoder_matching_test_continuous(test_loader, arguments, binary=False)
        teacher_classifier = copy.deepcopy(arguments.classifier)

        # BiC
        s = time.time()
        arguments.bic = bic_training(arguments.classifier, t + 1, val_loader, arguments)
        res_bic = encoder_matching_test_with_bic(test_loader, arguments, binary=False)
        time_spans[2] += time.time() - s

        # Using All
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V, all_loader,
                                                   arguments)
        time_spans[3] += time.time() - s
        res_aio = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        res_list = [res_rt, res_ds, res_bic, res_aio]
        fs = np.vstack([fs, np.hstack([res[0] for res in res_list])])
        for i in range(num_methods):
            res = res_list[i]
            new_res = np.hstack(
                [res[:4 + t], np.full(times - 2 - t, np.nan), res[-(4 + t):], np.full(times - 2 - t, np.nan)])
            res_list[i] = new_res
        results = np.vstack([results, np.hstack([res for res in res_list])])

    fs = np.array(fs).T
    time_end = time.time()
    time_spans /= (times - 1)
    print("Average Time Used:" + str(time_spans))

    np.savetxt('{}continuous_incremental/CI_seed{}_rs{}_lam{}.csv'.format(arguments.rec_path, seed, rs, lam), fs,
               fmt='%.3f', delimiter=',')
    np.savetxt('{}continuous_incremental/CI_seed{}_rs{}_lam{}_detail.csv'.format(arguments.rec_path, seed, rs, lam),
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
np.savetxt('{}continuous_incremental/CI_aver_seed{}_rs{}_lam{}.csv'.format(arguments.rec_path, seed, rs, lam), aver_fs,
           fmt='%.3f', delimiter=',')
np.savetxt('{}continuous_incremental/CI_aver_seed{}_rs{}_lam{}_detail.csv'.format(arguments.rec_path, seed, rs, lam),
           aver_results,
           fmt='%.3f', delimiter=',')

plot_multi_CI(aver_fs, 'HAR', None)

print(aver_fs)
