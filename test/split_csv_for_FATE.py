from args import *

# args = BCW_Arguments()
# args = DCC_Arguments()
args = EPS5_Arguments()
# args = HAR_Arguments()


out_path = "./data/split_csv/"

data = args.data.copy()
n = data.shape[0]
k = args.k_fold
s = n // k
# for j in range(k):
for j in range(1):
    # Train and Test j-th fold
    print("\n\n--------Fold {}--------".format(j))
    test_idc = range(j * s, min((j + 1) * s, n))
    train_x = np.delete(data[:, 0:-1], test_idc, axis=0)
    train_y = np.delete(data[:, [-1]], test_idc, axis=0)
    test_x = data[test_idc, 0:-1]
    test_y = data[test_idc, -1].reshape(len(test_idc), 1)
    train_x_host = train_x[:, args.party_B_idc]
    test_x_host = test_x[:, args.party_B_idc]
    train_x_guest = train_x[:, args.party_A_idc]
    test_x_guest = test_x[:, args.party_A_idc]
    train_host = train_x_host
    train_ids = np.array(range(1, train_y.shape[0]+1)).reshape([train_y.shape[0], 1]).astype(np.int16)
    test_ids = np.array(range(1, test_y.shape[0] + 1)).reshape([test_y.shape[0], 1]).astype(np.int16)

    # guest has y
    train_guest_np = np.concatenate((train_ids, train_y, train_x_guest), 1)
    test_guest_np = np.concatenate((test_ids, test_y, test_x_guest), 1)
    train_host_np = np.concatenate((train_ids, train_x_host), 1)
    test_host_np = np.concatenate((test_ids, test_x_host), 1)
    col_names_host = ['id'] + ['x{}'.format(i) for i in range(train_x_host.shape[1])]
    col_names_guest = ['id', 'y'] + ['x{}'.format(i) for i in range(train_x_guest.shape[1])]
    train_guest_df = pd.DataFrame(train_guest_np, columns=col_names_guest)
    test_guest_df = pd.DataFrame(test_guest_np, columns=col_names_guest)
    train_host_df = pd.DataFrame(train_host_np, columns=col_names_host)
    test_host_df = pd.DataFrame(test_host_np, columns=col_names_host)

    train_guest_df.to_csv(out_path + "train_guest.csv", index=False, header=True)
    test_guest_df.to_csv(out_path + "test_guest.csv", index=False, header=True)
    train_host_df.to_csv(out_path + "train_host.csv", index=False, header=True)
    test_host_df.to_csv(out_path + "test_host.csv", index=False, header=True)
