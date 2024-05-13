import numpy as np
import os
import logging
import torch
import tensorflow as tf


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test datasets for both original and synthetic datasets.

    Args:
      - data_x: original datasets
      - data_x_hat: generated datasets
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training datasets from the original datasets
    """
    # Divide train/test index (original datasets)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic datasets)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - datasets: original datasets

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - datasets: time-series datasets
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series datasets in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def save_checkpoint(ckpt_dir, state):
  import torch
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
  }
  torch.save(saved_state, ckpt_dir)

def restore_checkpoint(ckpt_dir, state, device='cuda:0'):
  if not os.path.exists(ckpt_dir):
      os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
      logging.warning(f"No checkpoint found at {ckpt_dir}. "
                      f"Returned the same state as input")
      return state
  else:
      loaded_state = torch.load(ckpt_dir, map_location=device)
      # state['optimizer'].load_state_dict(loaded_state['optimizer'])
      state['model'].load_state_dict(loaded_state['model'], strict=False)
      return state


def t_to_np(x):
    return x.detach().cpu().numpy()


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('cuda is available')
    else:
        device = torch.device("cpu")
    return device

def agg_losses(LOSSES, losses):
    if not LOSSES:
        LOSSES = [[] for _ in range(len(losses))]
    for jj, loss in enumerate(losses):
        LOSSES[jj].append(loss.item())
    return LOSSES

def log_losses(epoch, losses_tr, names):
    losses_avg_tr = []

    for loss in losses_tr:
        losses_avg_tr.append(np.mean(loss))

    loss_str_tr = 'Epoch {}, TRAIN: '.format(epoch + 1)
    for jj, loss in enumerate(losses_avg_tr):
        loss_str_tr += '{}={:.3e}, \t'.format(names[jj], loss)
    logging.info(loss_str_tr)

    logging.info('#'*30)
    return losses_avg_tr[0]
