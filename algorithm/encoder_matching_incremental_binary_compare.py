import copy

from sklearn.model_selection import train_test_split
from algorithm.encoder_matching_baseline import *
from algorithm.encoder_matching_incremental import classifier_distillation_training, bic_training, \
    encoder_matching_test_with_bic, classifier_fine_tune
from utils.utils import tensor_scale
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

use_unbalance = True
first_frac = 0.5
test_frac = 0.3
val_frac = 0.1
first_pos_frac = 0.8
lam = 0.9
fine_tune_scale = 1e-2

n_tr2 = None
n_tr1 = None
n_te = None


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        # self.arguments = arguments
        self.lf = nn.Linear(1, 1)

    def forward(self, x):
        assert x.size() == 2
        z = torch.zeros_like(x)
        z[0] = x[0]
        z[1] = self.lf(x[1])
        return z


def print_incremental_set():
    print()
    print("training set 1 size:" + str(n_tr1))
    print("training set 2 size:" + str(n_tr2))
    print("test set size:" + str(n_te))


def incremental_data_loader(arguments):
    global n_te, n_tr1, n_tr2
    n = arguments.data.shape[0]
    n_te = np.math.ceil(n * arguments.test_frac)
    n_tr = n - n_te
    n_tr1 = np.math.ceil(n_tr * first_frac)
    n_tr2 = n_tr - n_tr1
    n_tr1_pos = np.math.ceil(n_tr1 * first_pos_frac)
    n_tr1_neg = n_tr1 - n_tr1_pos
    n_tr2_pos = np.math.ceil(n_tr2 * (1 - first_pos_frac))
    n_tr2_neg = n_tr2 - n_tr2_pos
    data_copy = arguments.data.copy()
    if arguments.data_shuffle:
        np.random.shuffle(data_copy)
    y = data_copy[:, -1]
    if use_unbalance:
        idc = np.argsort(y)
    else:
        idc = np.random.permutation(len(y))
    data_copy = data_copy[idc]

    train1_x_data = np.vstack([data_copy[:n_tr1_neg, 0:-1], data_copy[-n_tr1_pos:, 0:-1]])
    train2_x_data = np.vstack([
        data_copy[n_tr1_neg:n_tr1_neg + n_tr2_neg, 0:-1],
        data_copy[-n_tr1_pos - n_tr2_pos:-n_tr1_pos, 0:-1]])
    test_x_data = data_copy[n_tr1_neg + n_tr2_neg:-n_tr1_pos - n_tr2_pos, 0:-1]

    train1_y_data = np.hstack([data_copy[:n_tr1_neg, -1], data_copy[-n_tr1_pos:, -1]])
    train2_y_data = np.hstack([data_copy[n_tr1_neg:n_tr1_neg + n_tr2_neg, -1],
                               data_copy[-n_tr1_pos - n_tr2_pos:-n_tr1_pos, -1]])

    train1_x_data, train1_y_data = shuffle(train1_x_data, train1_y_data, random_state=None)
    train2_x_data, train2_y_data = shuffle(train2_x_data, train2_y_data, random_state=None)

    train1_x_data = torch.from_numpy(train1_x_data)
    train2_x_data = torch.from_numpy(train2_x_data)
    test_x_data = torch.from_numpy(test_x_data)

    if arguments.data_scale:
        train1_x_data[:, arguments.train_idc] = \
            tensor_scale(train1_x_data[:, arguments.train_idc], arguments.data_scale_method)
        train2_x_data[:, arguments.train_idc] = \
            tensor_scale(train2_x_data[:, arguments.train_idc], arguments.data_scale_method)
        test_x_data[:, arguments.train_idc] = \
            tensor_scale(test_x_data[:, arguments.train_idc], arguments.data_scale_method)

    test_y_data = data_copy[n_tr1_neg + n_tr2_neg:-n_tr1_pos - n_tr2_pos, -1]
    train1_y_data = torch.from_numpy(train1_y_data).type(torch.LongTensor)
    train2_y_data = torch.from_numpy(train2_y_data).type(torch.LongTensor)
    test_y_data = torch.from_numpy(test_y_data).type(torch.LongTensor)

    train_x_data = torch.cat([train1_x_data, train2_x_data], 0)
    train_y_data = torch.cat([train1_y_data, train2_y_data], 0)

    _, val_x_data, _, val_y_data = train_test_split(train_x_data, train_y_data,
                                                    test_size=val_frac,
                                                    random_state=None, shuffle=True)

    train1_dataset = torch.utils.data.TensorDataset(train1_x_data, train1_y_data)
    train2_dataset = torch.utils.data.TensorDataset(train2_x_data, train2_y_data)
    train_dataset = torch.utils.data.TensorDataset(train_x_data, train_y_data)
    val_dataset = torch.utils.data.TensorDataset(val_x_data, val_y_data)
    test_dataset = torch.utils.data.TensorDataset(test_x_data, test_y_data)

    train1_loader = DataLoader(dataset=train1_dataset,
                               batch_size=arguments.batch_size,
                               shuffle=arguments.train_loader_shuffle)
    train2_loader = DataLoader(dataset=train2_dataset,
                               batch_size=arguments.batch_size,
                               shuffle=arguments.train_loader_shuffle)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=arguments.batch_size,
                              shuffle=arguments.train_loader_shuffle)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=arguments.batch_size,
                            shuffle=arguments.train_loader_shuffle)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=len(test_dataset),
                             shuffle=False)

    return train1_loader, train2_loader, train_loader, val_loader, test_loader


def encoder_matching_incremental(arguments):
    train1_loader, train2_loader, train_loader, val_loader, test_loader = incremental_data_loader(arguments)

    # prepare first turn model
    encoder_matching_train(train1_loader, arguments)
    teacher_classifier = copy.deepcopy(arguments.classifier)

    # Baseline: Fine-Tune
    arguments.classifier = classifier_fine_tune(arguments.classifier, train2_loader, arguments, scale=fine_tune_scale)
    ft_te = encoder_matching_test(test_loader, arguments)
    ft_tr = encoder_matching_test(train_loader, arguments)

    # Ablation: Distillation without BiC
    arguments.classifier = classifier_distillation_training(teacher_classifier, 2, train2_loader, arguments, lam=lam)
    dis = encoder_matching_test(test_loader, arguments)
    dis_tr = encoder_matching_test(train_loader, arguments)

    # Target: Distillation with BiC
    arguments.bic = bic_training(arguments.classifier, 1, val_loader, arguments)
    bic = encoder_matching_test_with_bic(test_loader, arguments)
    bic_tr = encoder_matching_test_with_bic(train_loader, arguments)

    # Baseline-High: Train all in once
    arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V, train_loader,
                                               arguments)
    all_in_once_te = encoder_matching_test(test_loader, arguments)
    all_in_once_tr = encoder_matching_test(train_loader, arguments)

    print(
        "                        Test set           Train set")
    print(
        "                        F-score  Accuracy  F-score  Accuracy")
    print(
        "Fine-Tune (Baseline):   {:.3f}   {:.3f}    {:.3f}   {:.3f}".format(ft_te[0], ft_te[1], ft_tr[0], ft_tr[1]))
    print(
        "Distillation (Ablation):{:.3f}   {:.3f}    {:.3f}   {:.3f}".format(dis[0], dis[1], dis_tr[0], dis_tr[1]))
    print(
        "Distillation + BiC:     {:.3f}   {:.3f}    {:.3f}   {:.3f}".format(bic[0], bic[1], bic_tr[0], bic_tr[1]))
    print(
        "Using all data once:    {:.3f}   {:.3f}    {:.3f}   {:.3f}".format(all_in_once_te[0], all_in_once_te[1],
                                                                            all_in_once_tr[0],
                                                                            all_in_once_tr[1]))

    first = (ft_te[0], ft_te[1], ft_tr[0], ft_tr[1])
    second = (dis[0], dis[1], dis_tr[0], dis_tr[1])
    twice = (bic[0], bic[1], bic_tr[0], bic_tr[1])
    all_in_once = (all_in_once_te[0], all_in_once_te[1], all_in_once_tr[0], all_in_once_tr[1])
    return np.array([first, second, twice, all_in_once], dtype=np.float)
