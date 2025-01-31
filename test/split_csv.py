import pandas as pd
from sklearn.utils import shuffle

folder_path = "../data/"
file_name = "epsilon_5k"
suffix = ".csv"
test_frac = 0.2
val_frac = 0.1

data_frame = pd.read_csv(folder_path+file_name+suffix, index_col='id')
num_features = data_frame.shape[1] - 1
num_samples = data_frame.shape[0]
num_test_samples = round(num_samples * test_frac)
num_train_samples = num_samples - num_test_samples
num_val_samples = int(num_samples * val_frac)

data_frame = shuffle(data_frame, random_state=2)

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

data_frame_y = data_frame.iloc[:, 0]
data_frame_x = data_frame.drop(['y'], axis=1)
data_frame_x_guest = data_frame_x.iloc[:, party_A_idc]
data_frame_host = data_frame_x.iloc[:, party_B_idc]
data_frame_guest = pd.concat([data_frame_y, data_frame_x_guest], axis=1)
data_frame_host_train = data_frame_host.iloc[num_val_samples:num_train_samples, :]
data_frame_host_validate = data_frame_host.iloc[:num_val_samples, :]
data_frame_host_test = data_frame_host.iloc[num_train_samples:, :]
data_frame_guest_train = data_frame_guest.iloc[num_val_samples:num_train_samples, :]
data_frame_guest_validate = data_frame_guest.iloc[:num_val_samples, :]
data_frame_guest_test = data_frame_guest.iloc[num_train_samples:, :]

data_frame_host_train.to_csv(folder_path+'my_'+file_name+'_host_train'+suffix, index=True, index_label='id')
data_frame_host_validate.to_csv(folder_path+'my_'+file_name+'_host_validate'+suffix, index=True, index_label='id')
data_frame_host_test.to_csv(folder_path+'my_'+file_name+'_host_test'+suffix, index=True, index_label='id')
data_frame_guest_train.to_csv(folder_path+'my_'+file_name+'_guest_train'+suffix, index=True, index_label='id')
data_frame_guest_validate.to_csv(folder_path+'my_'+file_name+'_guest_validate'+suffix, index=True, index_label='id')
data_frame_guest_test.to_csv(folder_path+'my_'+file_name+'_guest_test'+suffix, index=True, index_label='id')


