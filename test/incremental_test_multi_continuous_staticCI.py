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
from plot_continuous_incremental import plot_multi_sCI

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
lam = None
num_exemplar = 300
num_first_classes = 4
num_all_classes = 6
setup_seed(seed)
val_frac = 0.05

arguments = HAR_Arguments()
arguments.display_test_rec = False
arguments.display_cr = False
fine_tune_scale = 1e-1
num_methods = 4  # Re-train, distill, distill+BiC, Joint-training


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
            self.all_cache = [x_train, y_train]
            self.test_cache = [x_test, y_test]
            self.rest_cache = [x_tr_rest, y_tr_rest]
            self.train_loader = self.data_loader(x_first_tr, y_first_tr)
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
            self.train_loader = self.data_loader(x_first_tr, y_first_tr)
            self.test_loader = self.data_loader(x_te_first, y_te_first)
            self.all_loader = None
            self.val_loader = None
            self.train_loader_with_val = None

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
        self.train_loader = self.data_loader(x_train, y_train)

        x_train = np.concatenate([self.exp_cache[0], x_first], axis=0)
        y_train = np.concatenate([self.exp_cache[1], y_first], axis=0)
        self.train_loader_with_val = self.data_loader(x_train, y_train)

        # add validation
        self.val_cache[0] = np.concatenate([self.val_cache[0], x_first_va], axis=0)
        self.val_cache[1] = np.concatenate([self.val_cache[1], y_first_va], axis=0)
        self.val_loader = self.data_loader(self.val_cache[0], self.val_cache[1])

        return self.train_loader, self.val_loader, self.train_loader_with_val, self.all_loader, self.test_loader

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

fss = None
results = None

time_spans = np.array([0, 0, 0, 0], 'float32')
count = 0
if num_first_classes == 4:
    for m in range(5):
        for n in range(m+1, 6):
            count += 1
            dataLoaderGenerator = ContinuousDataLoaders(m, n)

            arguments.num_classes = num_first_classes
            encoder_matching_train(dataLoaderGenerator.train_loader, arguments)
            teacher_classifier = copy.deepcopy(arguments.classifier)
            res0 = encoder_matching_test_continuous(dataLoaderGenerator.test_loader, arguments, binary=False)

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
            time_spans[1] += time.time() - s
            res_ds = encoder_matching_test_continuous(test_loader, arguments, binary=False)

            # BiC
            s = time.time()
            arguments.classifier = classifier_distillation_training_CI(teacher_classifier, num_first_classes,
                                                                       train_loader,
                                                                       arguments, lam=lam)
            arguments.bic = bic_training(arguments.classifier, 2, val_loader, arguments)
            time_spans[2] += time.time() - s
            res_bic = encoder_matching_test_with_bic(test_loader, arguments, binary=False)

            # Using All
            s = time.time()
            arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V, all_loader,
                                                       arguments)
            time_spans[3] += time.time() - s
            res_aio = encoder_matching_test_continuous(test_loader, arguments, binary=False)

            res_list = [res_rt, res_ds, res_bic, res_aio]
            if fss is None:
                fss = np.hstack([res[0] for res in res_list])
                results = np.hstack(res_list)
            else:
                fss = np.vstack((fss, np.hstack([res[0] for res in res_list])))
                results = np.hstack([results, np.hstack(res_list)])

elif num_first_classes == 5:
    for n in range(0, 6):
        count += 1
        dataLoaderGenerator = ContinuousDataLoaders(n)
        arguments.num_classes = num_first_classes
        encoder_matching_train(dataLoaderGenerator.train_loader, arguments)
        teacher_classifier = copy.deepcopy(arguments.classifier)
        res0 = encoder_matching_test_continuous(dataLoaderGenerator.test_loader, arguments, binary=False)

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
        time_spans[1] += time.time() - s
        res_ds = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        # BiC
        s = time.time()
        arguments.classifier = classifier_distillation_training_CI(teacher_classifier, num_first_classes,
                                                                   train_loader,
                                                                   arguments, lam=lam)
        arguments.bic = bic_training(arguments.classifier, 1, val_loader, arguments)
        time_spans[2] += time.time() - s
        res_bic = encoder_matching_test_with_bic(test_loader, arguments, binary=False)

        # Using All
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V, all_loader,
                                                   arguments)
        time_spans[3] += time.time() - s
        res_aio = encoder_matching_test_continuous(test_loader, arguments, binary=False)

        res_list = [res_rt, res_ds, res_bic, res_aio]
        if fss is None:
            fss = np.hstack([res[0] for res in res_list])
            results = np.hstack(res_list)
        else:
            fss = np.vstack((fss, np.hstack([res[0] for res in res_list])))
            results = np.hstack([results, np.hstack(res_list)])

fs = np.array(fss)
time_end = time.time()
print("Average Time Used:" + str(time_spans / count))

print('Time Elapsed:{:.2f} min'.format((time_end - time_start) / 60))
aver_fs = np.mean(fs, axis=0)
aver_results = np.mean(results, axis=0)

np.savetxt('{}continuous_incremental/sCI_{}CI_lam{:.2f}.csv'.format(
    arguments.rec_path, num_all_classes - num_first_classes, lam), fs, fmt='%.3f', delimiter=',')
np.savetxt('{}continuous_incremental/sCI_{}CI_lam{:.2f}_detail.csv'.format(
    arguments.rec_path, num_all_classes - num_first_classes, lam), results, fmt='%.3f', delimiter=',')

print(aver_fs)
plot_multi_sCI(fs, 'HAR', None)
