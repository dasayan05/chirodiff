import os
import typing
import math
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from wandb.util import generate_id

import matplotlib.pyplot as plt


def positionalencoding1d(d_model, length, N=10000, dtype=None, device=None):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model, dtype=dtype, device=device)
    position = torch.arange(0, length, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=dtype, device=device) *
                         -(math.log(N) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def random_fourier_encoding_dyn(ts, W, scale=4.):
    proj = (W * scale) @ ts
    emb = torch.cat([torch.sin(2 * torch.pi * proj), torch.cos(2 * torch.pi * proj)], 0)
    return emb.T


def make_pad_mask_for_transformer(lens, total_length=None, device=None):
    total_length = total_length or max(lens)
    pad = torch.zeros(len(lens), total_length + 1, device=device)
    for b, l in enumerate(lens):
        pad[b, l] = 1.
    pad = torch.cumsum(pad, 1)
    return (pad[:, :-1] == 1.)


def nonunif_timestep_selector(T, infer_T, gamma=2.):
    ui = np.linspace(1., 0., infer_T) # uniform index
    return np.unique(np.clip(
            # sample using gamma curves (y = x^gamma)
            np.floor((ui ** gamma) * T), 1., T
        ))[::-1].astype(np.int64)


def openai_cosine_schedule(T, *args, s=0.008):
    # # explicitely defined $\bar{\alpha_t}$ and cosine function;
    # # beta and alpha derived thereafter; suggested by "Improved Denoising ..
    # # .. Diffusion Probabilistic Models" by OpenAI

    def f(t): return math.cos((t/T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = np.array([f(t) / f(0) for t in range(T + 1)], dtype=np.float32)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    sqrt_one_min_alpha_bar = np.sqrt(1. - alpha_bar)
    betas = np.clip(1. - alpha_bar[1:] / alpha_bar[:-1], 0., 0.999)
    alphas = 1. - betas
    beta_tilde = (1. - alpha_bar[:-1]) / (1. - alpha_bar[1:]) * betas

    return betas, alphas, alpha_bar[1:], \
        sqrt_alpha_bar[1:], sqrt_one_min_alpha_bar[1:], beta_tilde


def linear_schedule(T, low_noise, high_noise):
    # standard linear schedule defined in terms of $\beta_t$
    betas = np.linspace(low_noise, high_noise, T, dtype=np.float32)
    alphas = 1. - betas
    alpha_bar = np.cumprod(alphas, 0)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    sqrt_one_min_alpha_bar = np.sqrt(1. - alpha_bar)
    beta_tilde_wo_first_term = ((sqrt_one_min_alpha_bar[:-1] / sqrt_one_min_alpha_bar[1:])**2 * betas[1:])
    beta_tilde = np.array([
        beta_tilde_wo_first_term[0],
        *beta_tilde_wo_first_term
    ])

    return betas, alphas, alpha_bar, \
        sqrt_alpha_bar, sqrt_one_min_alpha_bar, beta_tilde


def cg_subtracted_noise(noise, lens):
    mask = torch.cumprod(1. - F.one_hot(lens, num_classes=noise.size(1) + 1)[:, :-1, None].float(), 1)
    # make sure the padding doesn't interfere in CoM calculation
    com = (mask * noise).sum(1, keepdim=True) / lens[:, None, None]
    return noise - com


class CustomWandbLogger(WandbLogger):

    def __init__(self,
                 name: typing.Optional[str],
                 save_dir: typing.Optional[str] = 'logs',
                 group: typing.Optional[str] = 'common',
                 project: typing.Optional[str] = 'diffset',
                 log_model: typing.Optional[bool] = True,
                 offline: bool = False,
                 entity: typing.Optional[str] = 'dasayan05'):
        rid = generate_id()
        name_rid = '-'.join([name, rid])
        super().__init__(name=name_rid, id=rid, offline=offline,
                         save_dir=os.path.join(save_dir, name_rid), project=project,
                         log_model=log_model, group=group, entity=entity)


class CustomViz(object):

    def __init__(self, test_n_sample_viz: int, n_viz: int, compact_mode: bool = True, subfig_slack: float = 0.) -> None:
        super().__init__()

        self.test_n_sample_viz = test_n_sample_viz
        self.n_viz = n_viz
        self.compact_mode = compact_mode

        if self.compact_mode:
            self.fig, self.ax = plt.subplots(
                self.test_n_sample_viz,
                self.n_viz,
                figsize=(self.n_viz, self.test_n_sample_viz),
                gridspec_kw = {'wspace': subfig_slack, 'hspace': subfig_slack})
        else:
            self.figs = [
                [
                    plt.subplots(1, 1, figsize=(1, 1)) \
                        for j in range(self.n_viz)
                ] for i in range(self.test_n_sample_viz)
            ]
    
    def __getitem__(self, pos: tuple):
        i, j = pos
        if self.compact_mode:
            return self.ax[i, j]
        else:
            _, ax = self.figs[i][j]
            return ax
    
    @property
    def shape(self):
        return self.test_n_sample_viz, self.n_viz
    
    def savefig(self, path: str, **kwargs):
        if self.compact_mode:
            self.fig.savefig(path, **kwargs)
        else:
            *rest, ext = path.split('.')
            rest = '.'.join(rest)
            os.makedirs(rest, exist_ok=False)
            for i in range(self.test_n_sample_viz):
                for j in range(self.n_viz):
                    path = os.path.join(rest, f'{i}_{j}.' + ext)
                    fig, _ = self.figs[i][j]
                    fig.savefig(path, **kwargs)