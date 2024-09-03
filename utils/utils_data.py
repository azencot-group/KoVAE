"""
data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine datasets
(2) real_data_loading: Load and preprocess real datasets
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import os
import torch
import controldiffeq
import pathlib


PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent


def to_tensor(data):
    return torch.from_numpy(data).float()


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - datasets: original datasets

    Returns:
      - norm_data: normalized datasets
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def pendulum_nonlinear(num_points, noise, theta=2.4):
    from matplotlib import pylab as plt
    from scipy.special import ellipj, ellipk
    np.random.seed(1)

    def sol(t, theta0):
        S = np.sin(0.5 * (theta0))
        K_S = ellipk(S ** 2)
        omega_0 = np.sqrt(9.81)
        sn, cn, dn, ph = ellipj(K_S - omega_0 * t, S ** 2)
        theta = 2.0 * np.arcsin(S * sn)
        d_sn_du = cn * dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0 * S * d_sn_dt / np.sqrt(1.0 - (S * sn) ** 2)
        return np.stack([theta, d_theta_dt], axis=1)

    anal_ts = np.arange(0, 170 * 0.1, 0.1)
    # Generate random angles in radians
    # angles = np.random.uniform(1, 3.5, num_points)
    angles = np.random.uniform(.5, 2.7, num_points)
    X = []
    for theta in angles:
     X.append(sol(anal_ts, theta))

    # X = X.T
    # Xclean = X.copy()
    X = np.array(X)
    X += np.random.standard_normal(X.shape) * noise


    X = MinMaxScaler(X)

    return X

def sine_data_generation(no, seq_len, dim):
    """Sine datasets generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - datasets: generated datasets
    """
    # Initialize the output
    data = list()

    # Generate sine datasets
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated datasets
        data.append(temp)

    return data


def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - datasets: preprocessed datasets.
    """
    assert data_name in ['stock', 'energy', 'metro']

    if data_name == 'stock':
        ori_data = np.loadtxt('./datasets/stock_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('./datasets/energy_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'metro':
        ori_data = np.loadtxt('./datasets/metro_data.csv', delimiter=",", skiprows=1)

    # Flip the datasets to make chronological datasets
    ori_data = ori_data[::-1]
    # Normalize the datasets
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the datasets
    temp_data = []
    # Cut datasets by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


class TimeDataset_irregular(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, missing_rate=0.0):
        SEED = 56789
        base_loc = PROJECT_DIR / 'datasets'
        loc = PROJECT_DIR / 'datasets' / (data_name + str(missing_rate))
        # if data is in cache
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)
        else:  # preprocess data according to missing rate
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)
            if not os.path.exists(loc):
                os.mkdir(loc)

            if data_name in ['stock', 'energy']:
                data = np.loadtxt(f'./datasets/{data_name}_data.csv', delimiter=",", skiprows=1)
                data = data[::-1]
                norm_data = MinMaxScaler(data)
                total_length = len(norm_data)
                time = np.array(range(total_length)).reshape(-1, 1)
            elif data_name == 'mujoco':
                tensors = load_data(loc)
                time = tensors['train_X'][:, :, :1].cpu().numpy()
                data = tensors['train_X'][:, :, 1:].reshape(-1, 14).cpu().numpy()
                norm_data = MinMaxScaler(data)
                norm_data = norm_data.reshape(4620, seq_len, 14)
            elif data_name == 'sine':
                norm_data = sine_data_generation(no=10000, seq_len=24, dim=5)

            self.original_sample = []
            ori_seq_data = []

            for i in range(len(norm_data) - seq_len + 1):
                x = norm_data[i: i + seq_len].copy()
                ori_seq_data.append(x)
            idx = torch.randperm(len(ori_seq_data))
            for i in range(len(ori_seq_data)):
                self.original_sample.append(ori_seq_data[idx[i]])
            orig_samples_np = np.array(self.original_sample)
            self.X_mean = np.mean(orig_samples_np, axis=0).reshape(1, orig_samples_np.shape[1], orig_samples_np.shape[2])

            generator = torch.Generator().manual_seed(SEED)
            removed_points = torch.randperm(norm_data.shape[0], generator=generator)[
                             :int(norm_data.shape[0] * missing_rate)].sort().values
            norm_data[removed_points] = float('nan')
            norm_data = np.concatenate((norm_data, time), axis=1)
            seq_data = []
            for i in range(len(norm_data) - seq_len + 1):
                x = norm_data[i: i + seq_len]
                seq_data.append(x)
            self.samples = []
            for i in range(len(seq_data)):
                self.samples.append(seq_data[idx[i]])

            self.samples = np.array(self.samples)

            norm_data_tensor = torch.Tensor(self.samples[:, :, :-1]).float().cuda()

            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.samples[:, :, -1][:, -1]).float()
            self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)
            self.original_sample = torch.tensor(self.original_sample)
            self.samples = torch.tensor(self.samples)

            save_data(loc, data=self.samples,
                      original_data=self.original_sample,
                      train_a=self.train_coeffs[0],
                      train_b=self.train_coeffs[1],
                      train_c=self.train_coeffs[2],
                      train_d=self.train_coeffs[3],
                      )

            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

    def __getitem__(self, index):
        batch_coeff = (self.train_coeffs[0][index].float(),
                       self.train_coeffs[1][index].float(),
                       self.train_coeffs[2][index].float(),
                       self.train_coeffs[3][index].float())

        self.sample = {'data': self.samples[index], 'inter': batch_coeff, 'original_data': self.original_sample[index]}

        return self.sample  # self.samples[index]

    def __len__(self):
        return len(self.samples)


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')