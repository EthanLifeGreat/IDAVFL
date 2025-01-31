import copy

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from algorithm.encoder_matching import *
from algorithm.encoder_matching_incremental import classifier_distillation_training, bic_training, \
    encoder_matching_test_with_bic, classifier_fine_tune
from utils.utils import tensor_scale
import numpy as np
from torch.utils.data import DataLoader

class_sort = True
first_frac = 0.5
exemplar_frac = 0.1
val_frac = exemplar_frac
num_first_classes = 3
n_tr1, n_tr2, n_te = 0, 0, 0
fine_tune_scale = 1e-2


def print_incremental_set():
    print()
    print("training set 1 size:" + str(n_tr1))
    print("training set 2 size:" + str(n_tr2))
    print("test set size:" + str(n_te))


def incremental_data_loader(arguments):
    global n_tr1, n_tr2, n_te

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

    n_tr1 = int(np.ceil(first_frac * n_tr))

    idc = np.arange(n_tr)
    if class_sort:
        idc = np.argsort(y_train)
        x_train = x_train[idc]
        y_train = y_train[idc]
        n_tr1 = np.where(y_train == num_first_classes)[0][0]

    elif arguments.data_shuffle:
        idc = np.random.permutation(idc)
        x_train = x_train[idc]
        y_train = y_train[idc]

    x_train_old, x_train_new, y_train_old, y_train_new = train_test_split(x_train, y_train, train_size=int(n_tr1),
                                                                          random_state=None, shuffle=False)
    x_train_old, x_val_old, y_train_old, y_val_old = train_test_split(x_train_old, y_train_old, test_size=val_frac,
                                                                      random_state=None, shuffle=True,
                                                                      stratify=y_train_old)
    x_train_new, x_val_new, y_train_new, y_val_new = train_test_split(x_train_new, y_train_new, test_size=val_frac,
                                                                      random_state=None, shuffle=True,
                                                                      stratify=y_train_new)
    _, x_train_old_exp, _, y_train_old_exp = train_test_split(x_train_old, y_train_old,
                                                              test_size=exemplar_frac,
                                                              random_state=None, shuffle=True)

    train1_x_data, train1_y_data = x_train_old, y_train_old
    train1_x_data = torch.from_numpy(train1_x_data).to(torch.float32)
    train1_y_data = torch.from_numpy(train1_y_data).type(torch.LongTensor)

    train2_x_data = np.concatenate([x_train_old_exp, x_train_new], axis=0)
    train2_y_data = np.concatenate([y_train_old_exp, y_train_new], axis=0)
    n_tr2 = len(train2_y_data)
    train2_x_data, train2_y_data = shuffle(train2_x_data, train2_y_data, random_state=None)
    train2_x_data = torch.from_numpy(train2_x_data).to(torch.float32)
    train2_y_data = torch.from_numpy(train2_y_data).type(torch.LongTensor)

    x_train, y_train = shuffle(x_train, y_train, random_state=None)
    train_x_data = torch.from_numpy(x_train).to(torch.float32)
    train_y_data = torch.from_numpy(y_train).type(torch.LongTensor)

    val_x_data = np.concatenate([x_val_old, x_val_new], axis=0)
    val_y_data = np.concatenate([y_val_old, y_val_new], axis=0)
    val_x_data, val_y_data = shuffle(val_x_data, val_y_data, random_state=None)
    val_x_data = torch.from_numpy(val_x_data).to(torch.float32)
    val_y_data = torch.from_numpy(val_y_data).type(torch.LongTensor)

    test_x_data = torch.from_numpy(x_test).to(torch.float32)
    test_y_data = torch.from_numpy(y_test).type(torch.LongTensor)

    train1_dataset = torch.utils.data.TensorDataset(train1_x_data, train1_y_data)
    train2_dataset = torch.utils.data.TensorDataset(train2_x_data, train2_y_data)
    train_dataset = torch.utils.data.TensorDataset(train_x_data, train_y_data)
    val_dataset = torch.utils.data.TensorDataset(val_x_data, val_y_data)
    test_dataset = torch.utils.data.TensorDataset(test_x_data, test_y_data)

    # Train_1 = Train_old
    # Train_2 = Train_old.exemplar + Train_new
    # Val = Val_old + Val_new
    # Train = Train_old + Train_new
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
    arguments.classifier = classifier_distillation_training(teacher_classifier, num_first_classes, train2_loader,
                                                            arguments)
    dis = encoder_matching_test(test_loader, arguments)
    dis_tr = encoder_matching_test(train_loader, arguments)

    # Target: Distillation with BiC
    arguments.bic = bic_training(arguments.classifier, num_first_classes, val_loader, arguments)
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
