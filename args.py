import numpy as np
import pandas as pd
import torch
import os

os.chdir(os.path.abspath(os.path.dirname(__file__)))

class Arguments(object):
    # Horizontal Data Partition
    test_frac = .3

    # Data Pre-processing Settings
    data_scale = False
    data_scale_method = 'min_max'
    data_shuffle = True
    train_loader_shuffle = True

    # Display Settings
    display_train_rec = False
    display_test_rec = True
    display_cr = False

    # Test using same data as training
    test_using_train_data = False

    # Record Classification result
    record_classification = False

    # k-fold argument
    k_fold = 10

    # Plot Correlation Heatmap
    plot_heatmap = True

    # Model Saving Path
    model_path = "./model/model_ckpt/"

    # Record Saving Path
    rec_path = "./record/"

    # Computing Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    def set_arg(self, arg_name, arg_val):
        exec("self.{} = {}".format(arg_name, arg_val))

    label_names = {
        0: 'Negative',
        1: 'Positive',
    }

    classifier = None
    sae_A2B = None
    sae_A = None
    sae_B2V = None


class EPS5_Arguments(Arguments):
    def __init__(self):
        super(EPS5_Arguments, self).__init__()

    data_scale = True
    data_scale_method = 'std'

    # Breast Cancer Wisconsin (Diagnostic) Data Set
    file_path = "./data/epsilon_5k.csv"
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=np.float32)
    data_x = np.delete(data, 1, axis=1)
    data_y = data[:, [1]]
    data = np.hstack((data_x, data_y))
    num_features = 100
    num_classes = 2
    train_idc = range(1, num_features + 1)
    data_frame = pd.read_csv(file_path, index_col='id')
    # Cross Entropy Weight (of Negative)
    neg_weight = .5
    weight_arr = None
    # weight_arr = torch.tensor([1 - neg_weight, neg_weight]).to(Arguments.device)

    __vdp_type = 1
    pivot = 50
    if __vdp_type == 1:
        # ---------------------- even+odd
        input_A_size = num_features // 2
        input_B_size = num_features - input_A_size
        party_A_idc = range(1, num_features, 2)
        party_B_idc = range(0, num_features, 2)
    elif __vdp_type == 2:
        # ---------------------- odd+even
        if num_features % 2 == 1:
            pivot = num_features // 2 + 1
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(0, num_features, 2)
        party_B_idc = range(1, num_features, 2)
    elif __vdp_type == 3:
        # ---------------------- post-3+pre-3
        input_B_size = pivot
        input_A_size = num_features - input_B_size
        party_B_idc = range(input_B_size)
        party_A_idc = range(input_B_size, num_features)
    elif __vdp_type == 4:
        # ---------------------- pre-3+post-3
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(input_A_size)
        party_B_idc = range(input_A_size, num_features)
    elif __vdp_type == 5:
        # ---------------------- 23 - baseline
        run_type = 'CV_baseline'
        input_A_size = 23
        input_B_size = num_features - input_A_size
        party_A_idc = range(input_A_size)
        party_B_idc = range(input_B_size, num_features)
    else:
        raise ValueError('__vdp_type value error!')

    all_idc = range(0, num_features)

    # Hyper-parameters: NN structure
    encoding_size = 200
    hidden_A_size = 100
    hidden_B_size = 100
    hidden_F_size = 40
    hidden_F_depth = 4
    classifier_hidden_size = 100
    classifier_depth = 1

    # Hyper-parameters: Training
    num_epochs_pre = 50
    num_epochs = 30
    batch_size = 128
    learning_rate_pre = 1e-3
    learning_rate = 5e-3
    weight_decay = 0

    # Disturbing Coefficient
    eps = 0.6

    # Record Saving Path
    rec_path = "./record/EPS5/"


class BCW_Arguments(Arguments):
    def __init__(self):
        super(BCW_Arguments, self).__init__()

    def cancer_type(s):
        it = {b'B': 0, b'M': 1}
        return it[s]

    # Breast Cancer Wisconsin (Diagnostic) Data Set
    file_path = "./data/bcw.csv"
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=np.float32)
    data_x = np.delete(data, 1, axis=1)
    data_y = data[:, [1]]
    data = np.hstack((data_x, data_y))
    num_features = 30
    num_classes = 2
    train_idc = range(1, num_features + 1)
    data_frame = pd.read_csv(file_path, index_col='id')

    # Cross Entropy Weight (of Negative)
    neg_weight = .25
    # weight_arr = None
    weight_arr = torch.tensor([1 - neg_weight, neg_weight]).to(Arguments.device)

    # data_scale = True
    # data_scale_method = 'std'

    __vdp_type = 3
    pivot = 20
    if __vdp_type == 1:
        # ---------------------- even+odd
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(1, num_features, 2)
        party_B_idc = range(0, num_features, 2)
    elif __vdp_type == 2:
        # ---------------------- odd+even
        if num_features % 2 == 1:
            pivot = pivot + 1
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(0, num_features, 2)
        party_B_idc = range(1, num_features, 2)
    elif __vdp_type == 3:
        # ---------------------- post-3+pre-3
        input_B_size = pivot
        input_A_size = num_features - input_B_size
        party_B_idc = range(input_B_size)
        party_A_idc = range(input_B_size, num_features)
    elif __vdp_type == 4:
        # ---------------------- pre-3+post-3
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(input_A_size)
        party_B_idc = range(input_A_size, num_features)
    elif __vdp_type == 5:
        # ---------------------- 23 - baseline
        run_type = 'CV_baseline'
        input_A_size = 23
        input_B_size = num_features - input_A_size
        party_A_idc = range(input_A_size)
        party_B_idc = range(input_B_size, num_features)
    else:
        raise ValueError('__vdp_type value error!')

    all_idc = range(0, num_features)

    # Hyperparameters: NN structure
    encoding_size = 200
    hidden_A_size = 500
    hidden_B_size = 500
    hidden_F_size = 40
    hidden_F_depth = 4
    classifier_hidden_size = 100
    classifier_depth = 1

    # Hyper-parameters: Training
    num_epochs_pre = 5
    num_epochs = 5
    batch_size = 128
    learning_rate_pre = 5e-3
    learning_rate = 5e-3
    weight_decay = 0

    # Disturbing Coefficient
    eps = 1

    # k-fold argument
    k_fold = 10

    # Record Saving Path
    rec_path = "./record/BCW/"


class DCC_Arguments(Arguments):

    # UCI-default-credit-cards dataset
    file_path = "./data/default_credit.csv"
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=np.float32)
    # data_x = data[:, :-1]
    # data_y = data[:, -1]
    data_x = np.delete(data, 1, axis=1)
    data_y = data[:, [1]]
    data = np.hstack((data_x, data_y))
    num_features = 23
    num_classes = 2
    train_idc = range(1, num_features + 1)
    data_frame = pd.read_csv(file_path, index_col='id')

    # Cross Entropy Weight (of Negative)
    neg_weight = .65
    weight_arr = torch.tensor([1 - neg_weight, neg_weight]).to(Arguments.device)

    __vdp_type = 1
    pivot = num_features // 2
    if __vdp_type == 1:
        # ---------------------- even+odd
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(1, num_features, 2)
        party_B_idc = range(0, num_features, 2)
    elif __vdp_type == 2:
        # ---------------------- odd+even
        if num_features % 2 == 1:
            pivot = pivot + 1
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(0, num_features, 2)
        party_B_idc = range(1, num_features, 2)
    elif __vdp_type == 3:
        # ----------------------
        input_B_size = 5
        input_A_size = num_features - input_B_size
        party_B_idc = range(input_B_size)
        party_A_idc = range(input_B_size, num_features)
    elif __vdp_type == 4:
        # ----------------------
        input_A_size = 5
        input_B_size = num_features - input_A_size
        party_A_idc = range(input_A_size)
        party_B_idc = range(input_A_size, num_features)
    elif __vdp_type == 5:
        # ---------------------- 23 - baseline
        run_type = 'CV_baseline'
        input_A_size = 23
        input_B_size = num_features - input_A_size
        party_A_idc = range(input_A_size)
        party_B_idc = range(input_B_size, num_features)
    else:
        raise ValueError('__vdp_type value error!')

    all_idc = range(0, num_features)

    # Hyperparameters: NN structure
    encoding_size = 200
    hidden_A_size = 100
    hidden_B_size = 100
    hidden_F_size = 50
    hidden_F_depth = 3
    classifier_hidden_size = 100
    classifier_depth = 1

    # Hyper-parameters: Training
    num_epochs_pre = 1
    num_epochs = 1
    batch_size = 128
    learning_rate_pre = 1e-3
    learning_rate = 5e-3
    weight_decay = 0

    # Disturbing Coefficient
    eps = 0.6

    # k-fold argument
    k_fold = 10

    # Record Saving Path
    rec_path = "./record/DCC/"


class HAR_Arguments(Arguments):
    # Human Activity Recognition dataset
    file_path = "./data/UCI_HAR_Dataset.txt"
    # x_train = np.loadtxt("data\\UCI HAR Dataset\\train\\X_train.txt", dtype=np.float32)
    # y_train = np.loadtxt("data\\UCI HAR Dataset\\train\\y_train.txt", dtype=np.int8)
    # x_test = np.loadtxt("data\\UCI HAR Dataset\\test\\X_test.txt", dtype=np.float32)
    # y_test = np.loadtxt("data\\UCI HAR Dataset\\test\\y_test.txt", dtype=np.int8)
    data = np.loadtxt(file_path, dtype=np.float32)
    # data_x = data[:, :-1]
    # data_y = data[:, -1]
    data_x = np.delete(data, -1, axis=1)
    data_y = data[:, [-1]] - 1
    data = np.hstack((data_x, data_y))
    num_features = 561
    num_classes = 6
    train_idc = range(1, num_features + 1)
    # pd.DataFrame.to_csv(pd.DataFrame(data), "./data/UCI_HAR_Dataset.csv")
    # data_frame = pd.read_csv(file_path, index_col='id')

    # Cross Entropy Weight (of Negative)
    weight_arr = None

    __vdp_type = 4
    pivot = num_features // 2
    if __vdp_type == 1:
        # ---------------------- even+odd
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(1, num_features, 2)
        party_B_idc = range(0, num_features, 2)
    elif __vdp_type == 2:
        # ---------------------- odd+even
        if num_features % 2 == 1:
            pivot = pivot + 1
        input_A_size = pivot
        input_B_size = num_features - input_A_size
        party_A_idc = range(0, num_features, 2)
        party_B_idc = range(1, num_features, 2)
    elif __vdp_type == 3:
        # ----------------------
        input_B_size = 50
        input_A_size = num_features - input_B_size
        party_B_idc = range(input_B_size)
        party_A_idc = range(input_B_size, num_features)
    elif __vdp_type == 4:
        # ----------------------
        input_A_size = 30
        input_B_size = num_features - input_A_size
        party_A_idc = range(input_A_size)
        party_B_idc = range(input_A_size, num_features)
    elif __vdp_type == 5:
        # ---------------------- 23 - baseline
        run_type = 'CV_baseline'
        input_A_size = 23
        input_B_size = num_features - input_A_size
        party_A_idc = range(input_A_size)
        party_B_idc = range(input_B_size, num_features)
    else:
        raise ValueError('__vdp_type value error!')

    all_idc = range(0, num_features)

    # Hyper-parameters: NN structure
    encoding_size = 200
    hidden_A_size = 500
    hidden_B_size = 500
    hidden_F_size = 50
    hidden_F_depth = 3
    classifier_hidden_size = 100
    classifier_depth = 1

    # Hyper-parameters: Training
    num_epochs_pre = 5
    num_epochs = 3
    batch_size = 128
    learning_rate_pre = 1e-3
    learning_rate = 1e-3
    weight_decay = 0

    # Disturbing Coefficient
    eps = 0.3

    # k-fold argument
    k_fold = 10

    # Record Saving Path
    rec_path = "./record/HAR/"

    display_test_rec = True
    display_cr = False

    label_names = {
        0: 'Walking',
        1: 'Walking Upstairs',
        2: 'Walking Downstairs',
        3: 'Sitting',
        4: 'Standing',
        5: 'Laying'
    }

