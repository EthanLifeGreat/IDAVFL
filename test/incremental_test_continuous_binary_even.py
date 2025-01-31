import copy
import random
import time
from algorithm.encoder_matching import *
from sklearn.model_selection import train_test_split
from algorithm.encoder_matching_incremental import classifier_distillation_training, bic_training, \
    encoder_matching_test_with_bic, classifier_fine_tune
from utils.utils import tensor_scale
from torch.utils.data import DataLoader
from plot_continuous_incremental import plot

from args import *
from sklearn.utils import shuffle


def setup_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True


ta = 5

bo = 0
seed = bo
rs = bo
times = 6
lam = 0.75  # 0.9
ki = False
setup_seed(seed)

# arguments = BCW_Arguments()
arguments = DCC_Arguments()
# arguments = EPS5_Arguments()


class ContinuousDataLoaders:
    def __init__(self, data_copy, t, test=True, kick_imbalance=False):
        self.times = t
        if kick_imbalance:
            np.random.shuffle(data_copy)
            pos_idc = (data_copy[:, -1] == 1)
            pos_data = data_copy[pos_idc]
            neg_data = data_copy[~pos_idc]
            n = min(pos_data.shape[0], neg_data.shape[0])
            pos_data = pos_data[:n]
            neg_data = neg_data[:n]
            data_copy = np.vstack([pos_data, neg_data])
        if test:
            self.data, self.test_data = train_test_split(data_copy, test_size=0.2, shuffle=True, random_state=rs)
        else:
            self.data = data_copy
            self.test_data = None
        self.test_loader = self.test_data_loader()

        pos_idc = (self.data[:, -1] == 1)
        self.pos_data = self.data[pos_idc]
        self.neg_data = self.data[~pos_idc]
        self.n_pos = self.pos_data.shape[0]
        self.n_neg = self.neg_data.shape[0]
        self.pos_start = 0
        self.neg_start = 0
        self.t = 0
        self.pos_end = round(0.25 * self.n_pos)
        self.neg_end = round(0.25 * self.n_neg)

    def next_loader(self):
        self.t += 1
        if self.t > self.times:
            return None
        self.pos_start, self.neg_start = self.pos_end, self.neg_end
        self.pos_end += round(0.15 * self.n_pos)
        self.neg_end += round(0.15 * self.n_neg)
        if self.t == self.times:
            self.pos_end = self.n_pos
            self.neg_end = self.n_neg
        return self.data_loader()

    def data_loader(self):
        pos_start = self.pos_start
        neg_start = self.neg_start
        ret = [None, None]
        for j in range(2):
            pos_x = self.pos_data[pos_start:self.pos_end, :-1]
            pos_y = self.pos_data[pos_start:self.pos_end, -1]
            neg_x = self.neg_data[neg_start:self.neg_end, :-1]
            neg_y = self.neg_data[neg_start:self.neg_end, -1]
            x_data = np.vstack([pos_x, neg_x])
            y_data = np.hstack([pos_y, neg_y])

            x_data, y_data = shuffle(x_data, y_data, random_state=rs)
            x_data = torch.from_numpy(x_data)
            y_data = torch.from_numpy(y_data).type(torch.LongTensor)
            if arguments.data_scale:
                x_data[:, arguments.train_idc] = \
                    tensor_scale(x_data[:, arguments.train_idc], arguments.data_scale_method)

            dataset = torch.utils.data.TensorDataset(x_data, y_data)
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=arguments.batch_size,
                                     shuffle=False)
            ret[j] = data_loader
            pos_start, neg_start = 0, 0

        return ret[0], ret[1]   # new_loader, all_loader

    def test_data_loader(self):
        if self.test_data is None:
            return None
        x_data = torch.from_numpy(self.test_data[:, :-1])
        y_data = torch.from_numpy(self.test_data[:, -1]).type(torch.LongTensor)
        if arguments.data_scale:
            x_data[:, arguments.train_idc] = \
                tensor_scale(x_data[:, arguments.train_idc], arguments.data_scale_method)
        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=arguments.batch_size,
                                 shuffle=False)
        return data_loader


time_start = time.time()

arguments.display_test_rec = False
arguments.display_cr = False
fine_tune_scale = 1e-1

data = arguments.data.copy()
aver_fs = 0
for k in range(ta):
    dataLoaderGenerator = ContinuousDataLoaders(data, t=times, kick_imbalance=ki)
    loader, _ = dataLoaderGenerator.data_loader()
    test_loader = dataLoaderGenerator.test_loader

    fs = []
    encoder_matching_train(loader, arguments)
    teacher_classifier = copy.deepcopy(arguments.classifier)
    ft_classifier = copy.deepcopy(arguments.classifier)
    fsm, fsn, fsp, acc = encoder_matching_test_continuous(test_loader, arguments)
    fs.append([fsm, fsm, fsm, fsm])
    results = np.array((fsm, fsm, fsm, fsm,
                        fsn, fsn, fsn, fsn,
                        fsp, fsp, fsp, fsp,
                        acc, acc, acc, acc)).reshape([1, 16])

    time_spans = np.array([0, 0, 0, 0], 'float32')

    for i in range(times - 1):
        new_loader, previous_loader = dataLoaderGenerator.next_loader()

        # Fine-tuning
        s = time.time()
        arguments.classifier = classifier_fine_tune(ft_classifier, new_loader, arguments, scale=fine_tune_scale)
        time_spans[1] += time.time() - s
        fsm_ft, fsn_ft, fsp_ft, acc_ft = encoder_matching_test_continuous(test_loader, arguments)
        ft_classifier = copy.deepcopy(arguments.classifier)

        # Distillation: Ours
        s = time.time()
        arguments.classifier = classifier_distillation_training(teacher_classifier, 2, new_loader, arguments, lam=lam)

        fsm_ds, fsn_ds, fsp_ds, acc_ds = encoder_matching_test_continuous(test_loader, arguments)
        time_spans[2] += time.time() - s
        # experimental comment:
        teacher_classifier = copy.deepcopy(arguments.classifier)

        # Re-train
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V, new_loader,
                                                   arguments)
        time_spans[0] += time.time() - s
        fsm_rt, fsn_rt, fsp_rt, acc_rt = encoder_matching_test_continuous(test_loader, arguments)


        # All in once
        s = time.time()
        arguments.classifier = classifier_training(arguments.sae_A2B, arguments.sae_A, arguments.sae_B2V,
                                                   previous_loader,
                                                   arguments)
        time_spans[3] += time.time() - s
        fsm_aio, fsn_aio, fsp_aio, acc_aio = encoder_matching_test_continuous(test_loader, arguments)

        fs.append([fsm_rt, fsm_ft, fsm_ds, fsm_aio])
        results = np.vstack([results,
                             np.array([fsm_rt, fsm_ft, fsm_ds, fsm_aio, fsn_rt, fsn_ft, fsn_ds, fsn_aio,
                                       fsp_rt, fsp_ft, fsp_ds, fsp_aio, acc_rt, acc_ft, acc_ds, acc_aio]).reshape(
                                 [1, 16])])

    fs = np.array(fs).T
    results = np.array(results).reshape([times, -1])
    time_end = time.time()
    time_spans /= (times - 1)
    print("Average Time Used:" + str(time_spans))

    np.savetxt('{}continuous_incremental/even/seed{}_rs{}_lam{}{}.txt'.format(arguments.rec_path, seed, rs, lam,
                                                                         '_ki' if ki else ''), fs)
    np.savetxt('{}continuous_incremental/even/seed{}_rs{}_lam{}{}_mpna.txt'.format(arguments.rec_path, seed, rs, lam,
                                                                         '_ki' if ki else ''), results, fmt='%.3f',
               delimiter='\t')
    print('Time Elapsed:{:.2f} min'.format((time_end - time_start) / 60))
    aver_fs += fs
    seed += 1
    rs += 1
    setup_seed(seed)

aver_fs /= ta
plot(aver_fs, 'Even', None)
np.savetxt('{}continuous_incremental/even/aver.txt'.format(arguments.rec_path), aver_fs, fmt='%.3f', delimiter='\t')
print(aver_fs)
