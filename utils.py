import pickle
import csv
from pathlib import Path
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
import copy
import numpy as np
import torch
import torch.nn as nn


######################################################################
# dataset processing
######################################################################
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        :param shuffle:
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data_len[-int(data_len * (val_ratio + test_ratio)): -int(data_len * test_ratio)]
    train_data = data[: -int(data_len * (val_ratio + test_ratio))]

    return train_data, val_data, test_data


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    """
    data format for seq2seq task or seq to single value task.
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :param single:
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:  # 预测一个值
        while index < end_index:
            X.append(data[index: index + window])
            Y.append(data[index + window + horizon - 1: index + window + horizon])
            index += 1
    else:  # 预测下一个序列
        while index < end_index:
            X.append(data[index: index + window])
            Y.append(data[index + window: index + window + horizon])
            index += 1
    X = np.array(X).astype('float32')
    Y = np.array(Y).astype('float32')

    return X, Y


def load_dataset(data_dir, batch_size, test_batch_size=None, **kwargs):
    """
    generate dataset
    :param data_dir:
    :param batch_size:
    :param test_batch_size:
    :param kwargs:
    :return:
    """
    data = {}
    if 'pollution' not in data_dir and 'weather' not in data_dir:
        for category in ['train', 'val', 'test']:
            cat_data = np.load(Path().joinpath(data_dir, category + '.npz'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']
            # data['x_' + category] = cat_data['x'].astype('float32')
            # data['y_' + category] = cat_data['y'].astype('float32')

        scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:  # norm?
            data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

        train_len = len(data['x_train'])
        permutation = np.random.permutation(train_len)
        data['x_train_1'] = data['x_train'][permutation][:int(train_len / 2)]
        data['y_train_1'] = data['y_train'][permutation][:int(train_len / 2)]
        data['x_train_2'] = data['x_train'][permutation][int(train_len / 2):]
        data['y_train_2'] = data['y_train'][permutation][int(train_len / 2):]
        data['x_train_3'] = copy.deepcopy(data['x_train_2'])
        data['y_train_3'] = copy.deepcopy(data['y_train_2'])
        data['train_loader_1'] = DataLoader(data['x_train_1'], data['y_train_1'], batch_size)
        data['train_loader_2'] = DataLoader(data['x_train_2'], data['y_train_2'], batch_size)
        data['train_loader_3'] = DataLoader(data['x_train_3'], data['y_train_3'], batch_size)

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = scalar

        return data
    else:
        dataset = np.load(data_dir, allow_pickle=True)
        data_train, data_val, data_test = split_data_by_ratio(dataset, 0.1, 0.2)
        x_tr, y_tr = Add_Window_Horizon(data_train, 12, 12, False)
        x_tr_orig = x_tr.copy()
        x_val, y_val = Add_Window_Horizon(data_val, 12, 12, False)
        x_test, y_test = Add_Window_Horizon(data_test, 12, 12, False)
        data['x_train'] = x_tr
        data['y_train'] = y_tr
        data['x_val'] = x_val
        data['y_val'] = y_val
        data['x_test'] = x_test
        data['y_test'] = y_test

        real_scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:
            for i in range(x_tr.shape[-1]):
                scaler = StandardScaler(mean=x_tr_orig[..., i].mean(), std=x_tr_orig[..., i].std())
                data['x_' + category][..., i] = scaler.transform(data['x_' + category][..., i])
            print('x_' + category, data['x_' + category].shape)

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = real_scaler

        return data


def load_adj(pkl_filename):
    """
    为什么gw的邻接矩阵要做对称归一化，而dcrnn的不做？其实做了，在不同的地方，是为了执行双向随机游走算法。
    所以K-order GCN需要什么样的邻接矩阵？
    这个应该参考ASTGCN，原始邻接矩阵呢？参考dcrnn
    为什么ASTGCN不采用对称归一化的拉普拉斯矩阵？
    :param pkl_filename: adj_mx.pkl
    :return:
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    return sensor_ids, sensor_id_to_ind, adj_mx
    # return sensor_ids, sensor_id_to_ind, adj_mx.astype('float32')


def load_pickle(pkl_filename):
    try:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pkl_filename, ':', e)
        raise

    return pkl_data


######################################################################
# generating chebyshev polynomials
######################################################################
def scaled_Laplacian(W):
    """
    compute \tilde{L}
    :param W: adj_mx
    :return: scaled laplacian matrix
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real  # k largest real part of eigenvalues

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T{K-1}
    :param L_tilde: scaled laplacian matrix
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i-1] - cheb_polynomials[i-2])

    return cheb_polynomials


######################################################################
# generating diffusion convolution adj
######################################################################
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


######################################################################
# metrics
######################################################################
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


######################################################################
# Exponential annealing for softmax temperature
######################################################################
class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.05, last_epoch=-1):
        self.total_epochs = total_epochs
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        # if self.curr_temp < self.temp_min:
        #     self.curr_temp = self.temp_min

        self.curr_temp = max(self.base_temp * 0.90 ** self.last_epoch, self.temp_min)

        return self.curr_temp


######################################################################
# PEMS03 ~ PEMS08 dataset
######################################################################
def generate_data(graph_signal_matrix_name, batch_size, test_batch_size=None, transformer=None):
    """shape=[num_of_samples, 12, num_of_vertices, 1]"""

    origin_data = np.load(graph_signal_matrix_name)  # shape=[17856, 170, 3]
    keys = origin_data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        data = generate_from_train_val_test(origin_data, transformer)

    elif 'data' in keys:
        length = origin_data['data'].shape[0]
        data = generate_from_data(origin_data, length, transformer)

    else:
        raise KeyError("neither data nor train, val, test is in the data")

    scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        # print(data['x_train'].shape)
        # print(data['x_val'].shape)
        # print(data['x_test'].shape)
        data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

    train_len = len(data['x_train'])
    permutation = np.random.permutation(train_len)
    data['x_train_1'] = data['x_train'][permutation][:int(train_len / 2)]
    data['y_train_1'] = data['y_train'][permutation][:int(train_len / 2)]
    data['x_train_2'] = data['x_train'][permutation][int(train_len / 2):]
    data['y_train_2'] = data['y_train'][permutation][int(train_len / 2):]
    data['x_train_3'] = copy.deepcopy(data['x_train_2'])
    data['y_train_3'] = copy.deepcopy(data['y_train_2'])
    data['train_loader_1'] = DataLoader(data['x_train_1'], data['y_train_1'], batch_size)
    data['train_loader_2'] = DataLoader(data['x_train_2'], data['y_train_2'], batch_size)
    data['train_loader_3'] = DataLoader(data['x_train_3'], data['y_train_3'], batch_size)

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar

    return data


def generate_from_train_val_test(origin_data, transformer):
    data = {}
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(origin_data[key], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')


    return data


def generate_from_data(origin_data, length, transformer):
    """origin_data shape: [17856, 170, 3]"""
    data = {}
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):

        x, y = generate_seq(origin_data['data'][line1: line2], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')


    return data


def generate_seq(data, train_length, pred_length):
    # split data to generate x and y
    # print(data.shape)
    # aa = np.expand_dims(data[0: 0 + train_length + pred_length], 0)
    # print(aa.shape)
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    # print(seq.shape)
    return np.split(seq, 2, axis=1)


def get_adj_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be connectivity or distance!")

    return A


######################################################################
# MLP for spatial attention
######################################################################
class MLP(nn.Module):
    def __init__(self, hiddens, input_size, activation_function, out_act, dropout_ratio=0.):
        super(MLP, self).__init__()
        # dropout_ratio = 0.2
        # layers = [nn.Dropout(dropout_ratio)]
        layers = []  # 包含线性层和相应的激活函数

        previous_h = input_size
        for i, h in enumerate(hiddens):
            # out_act为false的时候，输出层不加激活
            activation = None if i == len(hiddens) - 1 and not out_act else activation_function
            layers.append(nn.Linear(previous_h, h))

            if activation is not None:
                layers.append(activation)

            # layers.append(nn.Dropout(dropout_ratio))
            previous_h = h
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


######################################################################
# Darts utils
######################################################################
if __name__ == '__main__':
    adj = get_adj_matrix('data/pems/PEMS08/PEMS08.csv', 170)
    print(adj)
    # dataloader = load_dataset('data/METR-LA', 64, 64)
    # train_iterator = dataloader['train_loader_1'].get_iterator()
    # val_iterator = dataloader['train_loader_2'].get_iterator()
    # train_val = dataloader['train_loader'].get_iterator()
    # print(len(list(train_iterator)))
