from sklearn.metrics import f1_score, accuracy_score, classification_report, recall_score, precision_score
import torch
import numpy as np

beta = 1


def datasets(arguments):
    n = arguments.data.shape[0]
    n_te = np.math.ceil(n * arguments.test_frac)
    n_tr = n - n_te
    data_copy = arguments.data.copy()
    if arguments.data_shuffle:
        np.random.shuffle(data_copy)
    train_x_data = torch.from_numpy(data_copy[0:n_tr, 0:-1])
    test_x_data = torch.from_numpy(data_copy[n_tr:n, 0:-1])
    if arguments.data_scale:
        train_x_data[:, arguments.train_idc] = \
            tensor_scale(train_x_data[:, arguments.train_idc], arguments.data_scale_method)
        test_x_data[:, arguments.train_idc] = \
            tensor_scale(test_x_data[:, arguments.train_idc], arguments.data_scale_method)
    train_y_data = torch.from_numpy(data_copy[0:n_tr, -1]).type(torch.LongTensor)
    test_y_data = torch.from_numpy(data_copy[n_tr:n, -1]).type(torch.LongTensor)

    train_dataset = torch.utils.data.TensorDataset(train_x_data, train_y_data)
    test_dataset = torch.utils.data.TensorDataset(test_x_data, test_y_data)

    return train_dataset, test_dataset


def standard_scale(data):
    # Standardize features by removing the mean and scaling to unit variance. The standard score of a sample x is
    # calculated as:
    #       z = (x - u) / s
    # , where u is the mean of the training samples, and s is the standard deviation of the training samples.
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    data = torch.sub(data, mean)
    ret = torch.div(data, std)

    return ret


def min_max_scale(data):
    # Transforms features by scaling each feature to a given range,e.g.between minimum and maximum.
    # The transformation is given by:
    #       X_scale = (X - X.min) / (X.max - X.min)
    # , where X.min is the minimum value of feature, and X.max is the maximum
    col_min, _ = torch.min(data, dim=0)
    col_max, _ = torch.max(data, dim=0)
    dividend = torch.sub(data, col_min)
    divider = torch.sub(col_max, col_min)
    ret = torch.div(dividend, divider)

    return ret


def tensor_scale(data, data_scale_method):
    if data_scale_method != 'min_max':
        ret = standard_scale(data)
    else:
        ret = min_max_scale(data)

    return ret


def model_evaluation(label_pairs, call_name, arguments):
    label_pairs = np.array(label_pairs)
    total = len(label_pairs)

    predicted = label_pairs[:, 0]
    labels = label_pairs[:, 1]
    true_positive = ((predicted == 1) & (labels == 1)).sum()
    true_negative = ((predicted == 0) & (labels == 0)).sum()
    false_positive = ((predicted == 1) & (labels == 0)).sum()
    false_negative = ((predicted == 0) & (labels == 1)).sum()
    # correct = true_positive + true_negative

    fs = f1_score(labels, predicted, average='macro')
    acc = accuracy_score(labels, predicted)
    r = recall_score(labels, predicted, average='macro')
    p = precision_score(labels, predicted, average='macro')
    # ret = np.concatenate([fs, r])
    ret = np.stack([fs, p, r, acc])

    if arguments.display_test_rec:
        print("\n----{}----".format(call_name))
        print('\nTesting model:\nAccuracy of the network on the {} test samples: {} %'
              .format(total, 100 * acc))
        print('Confusion Matrix:\n'
              '\t\t\t\t   YES\t\t    NO\n'
              'positive\t\t{:>6d}\t\t{:>6d}\n'
              'negative\t\t{:>6d}\t\t{:>6d} '
              .format(true_positive, false_positive, false_negative, true_negative))
        print('F{}-score:\t{}'.format(beta, fs))

    if arguments.display_cr:
        cr = classification_report(labels, predicted, target_names=arguments.label_names.values(), digits=4)
        print(cr)

    return ret


def model_evaluation_multi(label_pairs, call_name, arguments):
    label_pairs = np.array(label_pairs)
    total = len(label_pairs)

    predicted = label_pairs[:, 0]
    labels = label_pairs[:, 1]
    # correct = true_positive + true_negative

    # fs = f_score(true_positive, true_negative, false_positive, false_negative)
    fs = f1_score(labels, predicted, average='macro')
    acc = accuracy_score(labels, predicted)
    r = recall_score(labels, predicted, average='macro')
    p = precision_score(labels, predicted, average='macro')

    if arguments.display_test_rec:
        print("\n----{}----".format(call_name))
        print('\nTesting model:\nAccuracy of the network on the {} test samples: {} %'
              .format(total, 100 * acc))
        print('F{}-score:\t{}'.format(beta, fs))

    if arguments.display_cr:
        cr = classification_report(labels, predicted, digits=4)
        print(cr)

    return fs, p, r


def model_evaluation_continuous(label_pairs, call_name, arguments):
    label_pairs = np.array(label_pairs)
    total = len(label_pairs)

    predicted = label_pairs[:, 0]
    labels = label_pairs[:, 1]
    true_positive = ((predicted == 1) & (labels == 1)).sum()
    true_negative = ((predicted == 0) & (labels == 0)).sum()
    false_positive = ((predicted == 1) & (labels == 0)).sum()
    false_negative = ((predicted == 0) & (labels == 1)).sum()
    # correct = true_positive + true_negative

    fsm = f1_score(labels, predicted, average='macro')
    acc = accuracy_score(labels, predicted)
    fsn, fsp = f1_score(labels, predicted, average=None)
    prn, prp = precision_score(labels, predicted, average=None)
    rcn, rcp = recall_score(labels, predicted, average=None)

    if arguments.display_test_rec:
        print("\n----{}----".format(call_name))
        print('\nTesting model:\nAccuracy of the network on the {} test samples: {} %'
              .format(total, 100 * acc))
        print('Confusion Matrix:\n'
              '\t\t\t\t   YES\t\t    NO\n'
              'positive\t\t{:>6d}\t\t{:>6d}\n'
              'negative\t\t{:>6d}\t\t{:>6d} '
              .format(true_positive, false_positive, false_negative, true_negative))
        print('F{}-score:\t{}'.format(beta, fsm))

    if arguments.display_cr:
        cr = classification_report(labels, predicted, target_names=arguments.label_names.values(), digits=4)
        print(cr)

    ret = np.array([fsm, prp, rcp, fsp, prn, rcn, fsn])
    return ret


def model_evaluation_continuous_multi(label_pairs, call_name, arguments):
    label_pairs = np.array(label_pairs)
    total = len(label_pairs)

    predicted = label_pairs[:, 0]
    labels = label_pairs[:, 1]
    # correct = true_positive + true_negative

    # fs = f_score(true_positive, true_negative, false_positive, false_negative)
    fs = f1_score(labels, predicted, average='macro')
    fss = f1_score(labels, predicted, average=None)
    acc = accuracy_score(labels, predicted)
    precisions = precision_score(labels, predicted, average=None)

    if arguments.display_test_rec:
        print("\n----{}----".format(call_name))
        print('\nTesting model:\nAccuracy of the network on the {} test samples: {} %'
              .format(total, 100 * acc))
        print('F{}-score:\t{}'.format(beta, fs))

    if arguments.display_cr:
        cr = classification_report(labels, predicted, digits=4)
        print(cr)

    ret = np.hstack([fs, fss, acc, precisions])
    return ret


def model_evaluation_rcr(label_pairs, call_name, arguments):
    total = len(label_pairs)
    correct = 0
    true_positive, true_negative = 0, 0
    false_positive, false_negative = 0, 0

    tp_id, tn_id, fp_id, fn_id = [], [], [], []
    for i in range(total):
        no = label_pairs[i, 0]
        predicted = label_pairs[i, 1]
        label = label_pairs[i, 2]
        if predicted == label == 1:
            true_positive += 1
            correct += 1
            tp_id.append(no)
        elif predicted == label == 0:
            true_negative += 1
            correct += 1
            tn_id.append(no)
        elif predicted == 1 and label == 0:
            false_positive += 1
            fp_id.append(no)
        elif predicted == 0 and label == 1:
            false_negative += 1
            fn_id.append(no)
        else:
            raise ValueError("Labels value error!")

    rcr(call_name, tp_id, tn_id, fp_id, fn_id, arguments)

    fs = f_score(true_positive, true_negative, false_positive, false_negative)
    acc = correct / total

    if arguments.display_test_rec:
        print("\n----{}----".format(call_name))
        print('\nTesting model:\nAccuracy of the network on the {} test samples: {} %'
              .format(total, 100 * acc))
        print('Confusion Matrix:\n'
              '\t\t\t\t   YES\t\t    NO\n'
              'positive\t\t{:>6d}\t\t{:>6d}\n'
              'negative\t\t{:>6d}\t\t{:>6d} '
              .format(true_positive, false_positive, false_negative, true_negative))
        print('F{}-score:\t{}'.format(beta, fs))

    return fs, acc


def f_score(true_positive, true_negative, false_positive, false_negative):
    reverse_pn = False
    if reverse_pn:
        true_negative, true_positive = true_positive, true_negative
        false_positive, false_negative = false_negative, false_positive
    if true_positive == 0:
        precision, recall = 0, 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    if precision + recall == 0:
        fs = 0
    else:
        fs = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    return fs


# record_classification_result
def rcr(call_name, tp_id, tn_id, fp_id, fn_id, arguments):
    if call_name not in arguments.data_frame.columns:
        arguments.data_frame[call_name] = ''
    arguments.data_frame.loc[tp_id, call_name] = "TP"
    arguments.data_frame.loc[tn_id, call_name] = "TN"
    arguments.data_frame.loc[fp_id, call_name] = "FP"
    arguments.data_frame.loc[fn_id, call_name] = "FN"


def confusion_matrix_analysis(true_positive, true_negative, false_positive, false_negative):
    predicted = []
    labels = []
    for i in range(true_positive):
        predicted.append([1])
        labels.append([1])
    for i in range(true_negative):
        predicted.append([0])
        labels.append([0])
    for i in range(false_positive):
        predicted.append([1])
        labels.append([0])
    for i in range(false_negative):
        predicted.append([0])
        labels.append([1])
    predicted = np.array(predicted)
    labels = np.array(labels)

    cr = classification_report(labels, predicted, digits=4)
    print(cr)
    fs = f1_score(labels, predicted, average='macro')
    acc = accuracy_score(labels, predicted)
    r = recall_score(labels, predicted, average='macro')
    p = precision_score(labels, predicted, average='macro')
    print('macro-F1', fs)
    print('macro-p', p)
    print('macro-r', r)
