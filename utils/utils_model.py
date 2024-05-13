import torch

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device


def which_device(model):
    return next(model.parameters()).device


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]),
                                         torch.Tensor([std / n_units]))
    A_init = sampler.sample((n_units, n_units))[..., 0]
    return A_init


def exp_lr_scheduler(epoch,
                     optimizer,
                     strategy='normal',
                     decay_eff=0.1,
                     decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""

    if strategy == 'normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
            print('New learning rate is: ', param_group['lr'])
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')

    return optimizer