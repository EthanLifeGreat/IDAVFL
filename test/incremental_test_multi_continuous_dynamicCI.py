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
from plot_continuous_incremental import plot_multi_dCI, plot_multi_sCI

from args import HAR_Arguments
from sklearn.utils import shuffle


def setup_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True


bo = 2
seed = bo
rs = bo
lam = .95
num_exemplar = 300
num_first_classes = 4
num_all_classes = 6
setup_seed(seed)
val_frac = 0.05

arguments = HAR_Arguments()
arguments.display_test_rec = False
arguments.display_cr = False
num_methods = 4  # Re-train, distill, distill+BiC, Joint-training
arguments.num_epochs *= 10


class ContinuousDataLoaders:
    def __init__(self, ic1, ic2=None):
        """

        :param ic1: the number of the first incremental class
        :param ic2:  the number of the second incremental class, and must be greater than ic1
        """
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

        if ic2 is not None:
            # pick Incremental Class(es), named as ic1, ic2
            assert ic1 < ic2
            if ic1 != 4:
                data_id_equals_4 = (np.where(y_train == 4), np.where(y_test == 4))
                data_id_equals_ic1 = (np.where(y_train == ic1), np.where(y_test == ic1))
                y_train[data_id_equals_4[0]] = ic1
                y_test[data_id_equals_4[1]] = ic1
                y_train[data_id_equals_ic1[0]] = 4
                y_test[data_id_equals_ic1[1]] = 4
            if ic2 != 5:
                data_id_equals_5 = (np.where(y_train == 5), np.where(y_test == 5))
                data_id_equals_ic2 = (np.where(y_train == ic2), np.where(y_test == ic2))
                y_train[data_id_equals_5[0]] = ic2
                y_test[data_id_equals_5[1]] = ic2
                y_train[data_id_equals_ic2[0]] = 5
                y_test[data_id_equals_ic2[1]] = 5

            idc = np.hstack((np.where(y_train == 4)[0], np.where(y_train == 5)[0]))
            select_arr = np.zeros(n_tr, dtype=bool)
            for j in idc:
                select_arr[j] = True
            x_tr_first, y_tr_first = x_train[~select_arr], y_train[~select_arr]
            x_tr_rest, y_tr_rest = x_train[select_arr], y_train[select_arr]
            x_first_tr, x_first_va, y_first_tr, y_first_va = train_test_split(x_tr_first, y_tr_first, test_size=val_frac,
                                                                              random_state=seed, shuffle=True,
                                                                              stratify=y_tr_first)
            _, x_train_old_exp, _, y_train_old_exp = train_test_split(x_first_tr, y_first_tr,
                                                                      test_size=num_exemplar, stratify=y_first_tr,
                                                                      random_state=seed, shuffle=True)

            idc = np.hstack((np.where(y_test == 4)[0], np.where(y_test == 5)[0]))
            select_arr = np.zeros(n_te, dtype=bool)
            for j in idc:
                select_arr[j] = True
            x_te_first, y_te_first = x_test[~select_arr], y_test[~select_arr]
            self.exp_cache = [x_train_old_exp, y_train_old_exp]
            self.val_cache = [x_first_va, y_first_va]
            self.all_cache = None
            self.test_cache = [x_test, y_test]
            self.rest_cache = [x_tr_rest, y_tr_rest]
            self.dynamic_train = [x_first_tr, y_first_tr]
            self.test_loader = self.data_loader(x_te_first, y_te_first)
            self.all_loader = None
            self.val_loader = None
            self.train_loader_with_val = None
        else:
            # pick Incremental Class, named as ic1
            ic2 = ic1
            if ic2 != 5:
                data_id_equals_5 = (np.where(y_train == 5), np.where(y_test == 5))
                data_id_equals_ic2 = (np.where(y_train == ic2), np.where(y_test == ic2))
                y_train[data_id_equals_5[0]] = ic2
                y_test[data_id_equals_5[1]] = ic2
                y_train[data_id_equals_ic2[0]] = 5
                y_test[data_id_equals_ic2[1]] = 5

            idc = np.where(y_train == 5)[0]
            select_arr = np.zeros(n_tr, dtype=bool)
            for j in idc:
                select_arr[j] = True
            x_tr_first, y_tr_first = x_train[~select_arr], y_train[~select_arr]
            x_tr_rest, y_tr_rest = x_train[select_arr], y_train[select_arr]
            x_first_tr, x_first_va, y_first_tr, y_first_va = train_test_split(x_tr_first, y_tr_first, test_size=val_frac,
                                                                              random_state=seed, shuffle=True,
                                                                              stratify=y_tr_first)
            _, x_train_old_exp, _, y_train_old_exp = train_test_split(x_first_tr, y_first_tr,
                                                                      test_size=num_exemplar, stratify=y_first_tr,
                                                                      random_state=seed, shuffle=True)

            idc = np.where(y_test == 5)[0]
            select_arr = np.zeros(n_te, dtype=bool)
            for j in idc:
                select_arr[j] = True
            x_te_first, y_te_first = x_test[~select_arr], y_test[~select_arr]
            self.exp_cache = [x_train_old_exp, y_train_old_exp]
            self.val_cache = [x_first_va, y_first_va]
            self.all_cache = [x_tr_first, y_tr_first]
            self.test_cache = [x_test, y_test]
            self.rest_cache = [x_tr_rest, y_tr_rest]
            self.dynamic_train = [x_first_tr, y_first_tr]
            self.test_loader = self.data_loader(x_te_first, y_te_first)
            self.all_loader = None
            self.val_loader = None
            self.train_loader_with_val = None

        self.t = 0
        self.dynamic_rest = []

    def dynamic_loader(self):
        give_frac = [[3, 3, 3, 3],  # time_stamp = 0
                     [4, 1, 1, 1],  # time_stamp = 1
                     [1, 4, 1, 1],  # time_stamp = 2
                     [1, 1, 4, 1],  # time_stamp = 3
                     [1, 1, 1, 4]]  # time_stamp = 4
        give_frac = np.array(give_frac)
        x_to_give, y_to_give = [], []
        for give_id in range(4):
            if self.t < 3:
                give_frac_this_id = give_frac[self.t][give_id] / np.sum(give_frac[self.t:, give_id])
                x_of_this_id, self.dynamic_train[0], y_of_this_id, self.dynamic_train[1] = \
                    train_test_split(self.dynamic_train[0], self.dynamic_train[1], train_size=give_frac_this_id,
                                     random_state=seed, shuffle=True, stratify=self.dynamic_train[1])
            elif self.t == 3:
                x_of_this_id, y_of_this_id = self.dynamic_train[0], self.dynamic_train[1]
            else:
                raise ValueError
            x_to_give.append(x_of_this_id)
            y_to_give.append(y_of_this_id)
        x_to_give = np.concatenate(x_to_give, axis=0)
        y_to_give = np.concatenate(y_to_give, axis=0)
        if self.all_cache is None:
            self.all_cache = [x_to_give, y_to_give]
        else:
            self.all_cache[0] = np.concatenate([self.all_cache[0], x_to_give], axis=0)
            self.all_cache[1] = np.concatenate([self.all_cache[1], y_to_give], axis=0)
        all_ldr = self.data_loader(self.all_cache[0], self.all_cache[1])
        train_ldr = self.data_loader(x_to_give, y_to_give)

        self.t += 1
        return train_ldr, all_ldr

    def incremental_loader(self):
        # Class Increment at the Last Timestamp
        x_first, x_rest, y_first, y_rest = self.rest_cache[0], None, self.rest_cache[1], None
        self.test_loader = self.data_loader(self.test_cache[0], self.test_cache[1])

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
        train_ldr = self.data_loader(x_train, y_train)

        x_train = np.concatenate([self.exp_cache[0], x_first], axis=0)
        y_train = np.concatenate([self.exp_cache[1], y_first], axis=0)
        self.train_loader_with_val = self.data_loader(x_train, y_train)

        # add validation
        self.val_cache[0] = np.concatenate([self.val_cache[0], x_first_va], axis=0)
        self.val_cache[1] = np.concatenate([self.val_cache[1], y_first_va], axis=0)
        self.val_loader = self.data_loader(self.val_cache[0], self.val_cache[1])

        return train_ldr, self.val_loader, self.train_loader_with_val, self.all_loader, self.test_loader

    @staticmethod
    def data_loader(x_data, y_data):
        x_data, y_data = shuffle(x_data, y_data, random_state=rs)
        x_data = torch.from_numpy(x_data).to(torch.float32)
        y_data = torch.from_numpy(y_data).type(torch.LongTensor)
        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=arguments.batch_size,
                                 shuffle=True)
        return data_loader


time_start = time.time()

time_spans = np.array([0, 0, 0, 0], 'float32')
count = 0
fss = []
assert num_first_classes == 4

fs_CIs = []
for m in range(5):
    for n in range(m+1, 6):
        count += 1
        dataLoaderGenerator = ContinuousDataLoaders(m, n)
        arguments.num_classes = num_first_classes
        train_ldr, all_ldr = dataLoaderGenerator.dynamic_loader()
        encoder_matching_train(train_ldr, arguments)
        teacher_classifier = copy.deepcopy(arguments.classifier)
        res0 = encoder_matching_test_continuous(dataLoaderGenerator.test_loader, arguments, binary=False)
        fs = np.hstack([res0[0] for _ in range(num_methods)])
        for t in range(1, 4):
            train_ldr, prev_ldr = dataLoaderGenerator.dynamic_loader()
            # Re-train
            arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V,
                                                       train_ldr, arguments)
            res_rt = encoder_matching_test_continuous(dataLoaderGenerator.test_loader, arguments, binary=False)
            # Distill
            arguments.classifier = classifier_distillation_training(teacher_classifier, num_first_classes, train_ldr
                                                                    , arguments, lam=lam)
            teacher_classifier = copy.deepcopy(arguments.classifier)
            results_dis = encoder_matching_test_continuous(dataLoaderGenerator.test_loader, arguments, binary=False)
            # All in once
            arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V,
                                                       prev_ldr, arguments)
            results_aio = encoder_matching_test_continuous(dataLoaderGenerator.test_loader, arguments, binary=False)

            fs = np.vstack([fs, np.hstack([res[0] for res in [res_rt, results_dis, results_dis, results_aio]])])

        train_loader, val_loader, train_loader_with_val, all_loader, test_loader = \
            dataLoaderGenerator.incremental_loader()
        arguments.num_classes = num_all_classes


        # Re-train
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V,
                                                   train_loader_with_val, arguments)
        time_spans[0] += time.time() - s
        res_rt = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        # Distillation
        s = time.time()
        # lam = np.power((i + num_first_classes)/(i + num_first_classes + 1), 1/2)
        lam = num_first_classes / num_all_classes
        arguments.classifier = classifier_distillation_training_CI(teacher_classifier, num_first_classes,
                                                                   train_loader_with_val,
                                                                   arguments, lam=lam)
        time_1 = time.time() - s
        time_spans[1] += time_1
        res_ds = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        # BiC
        arguments.classifier = classifier_distillation_training_CI(teacher_classifier, num_first_classes,
                                                                   train_loader,
                                                                   arguments, lam=lam)
        s = time.time()
        arguments.bic = bic_training(arguments.classifier, 2, val_loader, arguments)
        time_spans[2] += time.time() - s + time_1
        res_bic = encoder_matching_test_with_bic(test_loader, arguments, binary=False)

        # Using All
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V, all_loader,
                                                   arguments)
        time_spans[3] += time.time() - s
        res_aio = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        fs_CI = np.hstack([res[0] for res in [res_rt, res_ds, res_bic, res_aio]])
        fs_CIs.append(fs_CI)
        fs = np.vstack([fs, fs_CI])

        fs = fs.T  # dim:0, method; dim:1, times
        fss.append(fs)

fs_CIs = np.vstack(fs_CIs)
fs_CIs = np.vstack([fs_CIs, fs_CIs.mean(0)])
aver_fs = np.average(fss, axis=0)
fss = np.concatenate(fss, axis=1)
time_end = time.time()
time_spans /= count
print('Time Elapsed:{:.2f} min'.format((time_end - time_start) / 60))

print("Average Time Used:" + str(time_spans))
np.savetxt('{}continuous_incremental/dCI_{}CI_{}times_CI_result_lam{:.2f}.csv'.format(
    arguments.rec_path, num_all_classes - num_first_classes, count, lam), fs_CIs, fmt='%.3f', delimiter=',')
np.savetxt('{}continuous_incremental/dCI_{}CI_{}times_result_lam{:.2f}.csv'.format(
    arguments.rec_path, num_all_classes - num_first_classes, count, lam), fss, fmt='%.3f', delimiter=',')
np.savetxt('{}continuous_incremental/dCI_{}CI_aver_result_{}_times_lam{:.2f}.csv'.format(
    arguments.rec_path, num_all_classes - num_first_classes, count, lam), aver_fs, fmt='%.3f', delimiter=',')

plot_multi_dCI(aver_fs, 'Dynamic Incremental', None)
print(np.mean(fs, axis=1))
