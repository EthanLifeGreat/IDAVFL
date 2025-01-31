from utils.utils import tensor_scale
from algorithm.encoder_matching import *
import sys
import numpy as np
import os
import pandas as pd


def record_cv_arg(arg_name, arguments, train_func, test_func, arg_list, times=1):
    folder_name = "argument_" + arg_name + "_temp_result/"

    f0 = sys.stdout
    if not os.path.isdir(arguments.rec_path + folder_name):
        os.mkdir(arguments.rec_path + folder_name)
    f = open(arguments.rec_path + folder_name + arg_name + "_performance.rec.txt", 'w')
    sys.stdout = f
    print("argument " + arg_name + " performance test:\n\n")

    for arg_val in arg_list:
        arguments.set_arg(arg_name, arg_val)
        rec_name = folder_name + "{}.rec.txt".format(arg_val)
        print("running with argument value:{}".format(arg_val), file=sys.__stdout__)
        score = record_cv(rec_name, arguments, train_func, test_func, times)
        print("value:{}\t\tscore:{}\n".format(arg_val, score))
    f.close()
    sys.stdout = f0


def record_cv(rec_name, arguments, train_func, test_func, times=1):
    rec_name += ".rec.txt"
    f0 = sys.stdout
    f = open(arguments.rec_path + rec_name, 'w')
    print("Logging output in file: {}".format(arguments.rec_path+rec_name))
    sys.stdout = f
    cv = CV(arguments.k_fold, arguments)
    if times == 1:
        print("running once time CV mode\n")
        aver_score = cv.cross_validation(arguments.data.copy(), train_func, test_func, arguments)
        print("\n\nTotal CV score:\n{}".format(aver_score))
    else:
        aver_score = 0
        print("running {} times encoder_matching_compare CV mode\n".format(times))
        for i in range(times):
            print("\n-------THE {}th TIME-------\n".format(i))
            arguments.display_train_rec = False
            score = cv.cross_validation(arguments.data.copy(), train_func, test_func, arguments)
            print("\n\nTotal CV score:{}".format(score))
            aver_score += score
        aver_score /= times
        print("\n\nCV_compare_{}\n Total CV score:\n{}".format(times, aver_score))
    f.close()
    sys.stdout = f0
    print("Save evaluation data in file: {}".format(arguments.rec_path+"metrics.csv"))
    pd.DataFrame.to_csv(pd.DataFrame(aver_score.reshape(aver_score.shape[0], -1)), arguments.rec_path+"metrics.csv")
    print("Done!\n\n")
    return aver_score


class CV:
    def __init__(self, k_fold, arguments):
        self.k_fold = k_fold
        self.scale = arguments.data_scale
        self.scale_method = arguments.data_scale_method

    def cross_validation(self, data, train_func, test_func, arguments):
        n = data.shape[0]
        k = self.k_fold
        s = n // k
        if arguments.data_shuffle:
            np.random.shuffle(data)
        scores = []
        for j in range(k):
            # Train and Test j-th fold
            print("\n\n--------Fold {}--------".format(j))
            test_idc = range(j * s, min((j + 1) * s, n))
            train_x = np.delete(data[:, 0:-1], test_idc, axis=0)
            train_y = np.delete(data[:, -1], test_idc, axis=0)
            test_x = data[test_idc, 0:-1]
            test_y = data[test_idc, -1]
            train_x_data = torch.from_numpy(train_x)
            test_x_data = torch.from_numpy(test_x)
            if arguments.data_scale:
                train_x_data[:, arguments.train_idc] = \
                    tensor_scale(train_x_data[:, arguments.train_idc], arguments.data_scale_method)
                test_x_data[:, arguments.train_idc] = \
                    tensor_scale(test_x_data[:, arguments.train_idc], arguments.data_scale_method)
            train_dataset = torch.utils.data.TensorDataset(train_x_data,
                                                           torch.from_numpy(train_y).type(torch.LongTensor))
            test_dataset = torch.utils.data.TensorDataset(test_x_data,
                                                          torch.from_numpy(test_y).type(torch.LongTensor))
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=arguments.batch_size,
                                                       shuffle=arguments.train_loader_shuffle)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=len(test_dataset),
                                                      shuffle=False)
            train_func(train_loader, arguments)
            score = test_func(test_loader, arguments)
            scores.append(score)
        scores = np.stack(scores)
        # score = np.divide(scores, k)

        return scores
