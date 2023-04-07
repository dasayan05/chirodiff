import os
import sys
import typing
import contextlib
import numpy as np
import matplotlib
from matplotlib.cm import get_cmap
matplotlib.rcParams['axes.edgecolor'] = '#aaaaaa'
from enum import Enum

import torch
from torch_ema import ExponentialMovingAverage as EMA
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    TQDMProgressBar,
    ModelCheckpoint
)

from data.dm import ReprType, GenericDM
from data.sketch import Sketch
from models.score import (
    ScoreFunc,
    TransformerSetFeature,
    BiRNNEncoderFeature,
    ClassEmbedding
)
from utils import (
    positionalencoding1d,
    random_fourier_encoding_dyn,
    make_pad_mask_for_transformer,
    openai_cosine_schedule,
    linear_schedule,
    CustomViz,
)


class SketchDiffusion(pl.LightningModule):

    class ModelType(Enum):
        birnn = "birnn"
        transformer = "transformer"
    
    class SamplingAlgo(Enum):
        ddpm = "ddpm"
        ddim = "ddim"
        fddim = "fddim" # only for private use
    
    class NoiseSchedule(Enum):
        linear = "linear"
        cosine = "cosine"
    
    class TimeEmbedding(Enum):
        sinusoidal = "sinusoidal"
        randomfourier = "randomfourier"
    
    class VizProcess(Enum):
        forward = "forward"
        backward = "backward"
        both = "both"
    
    class Parameterization(Enum):
        mu = "mu"
        eps = "eps"

    def __init__(self,
                 repr: ReprType = ReprType.threeseqdel,
                 modeltype: ModelType = ModelType.transformer,
                 time_embedding: TimeEmbedding = TimeEmbedding.sinusoidal,
                 vae_weight: float = 0.,
                 vae_kl_anneal_start: int = 200_000,
                 vae_kl_anneal_end: int = 400_000,
                 num_classes: typing.Optional[int] = None,
                 optim_ema: bool = True,
                 optim_sched: str = 'steplr',
                 optim_lr: float = 1.e-4,
                 optim_decay: float = 1.e-2,
                 optim_gamma: float = 0.9995,
                 optim_warmup: int = 3000,
                 optim_interval: str = 'step',
                 optim_div_factor: int = 3,
                 arch_head: int = 4,
                 arch_layer: int = 4,
                 arch_internal: int = 64,
                 arch_layer_cond: typing.Optional[int] = None,
                 arch_internal_cond: typing.Optional[int] = None,
                 arch_pe_dim: int = 2,
                 arch_n_cond_latent: int = 32,
                 arch_causal: bool = False,
                 arch_dropout: float = 0.1,
                 arch_parameterization: Parameterization = Parameterization.eps, # unused
                 noise_low_noise: float = 1e-4,
                 noise_high_noise: float = 2e-2,
                 noise_schedule: NoiseSchedule = NoiseSchedule.linear,
                 noise_T: int = 1000,
                 test_variance_strength: float = 0.5,
                 test_sampling_algo: SamplingAlgo = SamplingAlgo.ddpm,
                 test_partial_T: typing.Optional[int] = None,
                 test_recon: bool = True,
                 test_interp: bool = False,
                 test_n_viz: int = 10,
                 test_n_sample_viz: int = 10,
                 test_viz_fig_compact: bool = True,
                 text_viz_process: VizProcess = VizProcess.both,
                 test_save_everything: bool = True
                ) -> None:
        """
        Diffusion Model for Sketches (both set and sequential representation)

        Args:
            repr: POINTCLOUD for sets and THREEPOINT for sequence
            arch: architecture params of transformer/RNN (head, layer, inp_n_emb, ff_dim, pe_dim)
            noise: noise parameters (number of scales, low and high noise variance, T)
            test: which test to do (reconstruction, interpolation etc)
        """

        super().__init__()
        self.save_hyperparameters()
        self.hp = self.hparams

        self.cond = self.hp.repr in [
            ReprType.threeseqdel_pointcloudcond,
            ReprType.threeseqdel_classcond,
            ReprType.threeseqabs_classcond,
            ReprType.threeseqabs_pointcloudcond,
            ReprType.threeseqabs_threeseqabscond
        ]

        if self.hp.vae_weight != 0.:
            assert self.hp.repr.value.endswith('pointcloudcond') or self.hp.repr.value.endswith('threeseqabscond'), \
                "VAE only allowed in bottlenecked conditional models"

        self.elem_dim = 3

        self.pe_dim = self.hp.arch_pe_dim
        
        n_cond_dim = 0
        if self.cond:
            n_cond_dim = self.hp.arch_n_cond_latent

        self.seq_pe_dim = self.pe_dim if self.hp.modeltype == self.ModelType.transformer else 0

        if self.cond:
            if self.hp.repr.value.endswith('pointcloudcond'):
                self.encoder = TransformerSetFeature(
                    self.hp.arch_internal_cond or self.hp.arch_internal,
                    self.hp.arch_layer_cond or self.hp.arch_layer,
                    self.hp.arch_head,
                    n_cond_dim,
                    dropout=self.hp.arch_dropout,
                    vae_weight=self.hp.vae_weight
                )
            elif self.hp.repr == ReprType.threeseqabs_threeseqabscond:
                self.encoder = BiRNNEncoderFeature(
                    self.hp.arch_internal_cond or self.hp.arch_internal,
                    self.hp.arch_layer_cond or self.hp.arch_layer,
                    n_cond_dim,
                    dropout=self.hp.arch_dropout,
                    vae_weight=self.hp.vae_weight
                )
            elif self.hp.repr == ReprType.threeseqdel_classcond or self.hp.repr == ReprType.threeseqabs_classcond:
                assert self.hp.num_classes is not None, "class conditional model but num_classes == 0"
                self.encoder = ClassEmbedding(self.hp.num_classes, n_cond_dim)
            else:
                raise NotImplementedError('unknown conditioning type')

        self.scorefn = ScoreFunc(
            self.hp.modeltype.value,
            # kwargs go here onwards
            inp_n_features=self.elem_dim * 2 - 1, # concat complementary repr too
            out_n_features=self.elem_dim,
            time_pe_features=self.pe_dim,
            seq_pe_features=self.seq_pe_dim,
            n_cond_features=n_cond_dim,
            n_internal=self.hp.arch_internal,
            n_head=self.hp.arch_head,
            n_layer=self.hp.arch_layer,
            causal=self.hp.arch_causal,
            dropout=self.hp.arch_dropout
        )
        if self.hp.optim_ema:
            self.ema = EMA([
                *self.scorefn.parameters(),
                *(self.encoder.parameters() if self.cond else [])
            ], decay=0.9999)

        self.register_buffer("pe_proj_W",
            torch.randn(self.pe_dim // 2, 1, requires_grad=False), persistent=True
        )
        if self.seq_pe_dim > 0:
            self.register_buffer("seq_proj_W",
                torch.randn(self.seq_pe_dim // 2, 1, requires_grad=False), persistent=True
            )

        # pre-computing all betas and alphas
        schedule_generator = {
            SketchDiffusion.NoiseSchedule.linear: linear_schedule,
            SketchDiffusion.NoiseSchedule.cosine: openai_cosine_schedule
        }[self.hp.noise_schedule]
        betas, alphas, alpha_bar, sqrt_alpha_bar, sqrt_one_min_alpha_bar, beta_tilde = \
            schedule_generator(
                self.hp.noise_T,
                self.hp.noise_low_noise * 1000 / self.hp.noise_T,
                self.hp.noise_high_noise * 1000 / self.hp.noise_T,
            )
        self.register_buffer("betas", torch.from_numpy(betas), persistent=False)
        self.register_buffer("alphas", torch.from_numpy(alphas), persistent=False)
        self.register_buffer("alpha_bar", torch.from_numpy(alpha_bar), persistent=False)
        self.register_buffer("sqrt_alpha_bar", torch.from_numpy(sqrt_alpha_bar), persistent=False)
        self.register_buffer("sqrt_one_min_alpha_bar", torch.from_numpy(sqrt_one_min_alpha_bar), persistent=False)
        self.register_buffer("beta_tilde", torch.from_numpy(beta_tilde), persistent=False)

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        if self.device.index == 0 and self.hp.optim_ema:
            self.ema.to(self.device)
        return ret

    def on_fit_start(self) -> None:
        self.on_test_start() # needed for testing while training

    def on_before_zero_grad(self, optimizer) -> None:
        if self.device.index == 0 and self.hp.optim_ema:
            self.ema.update([
                *self.scorefn.parameters(),
                *(self.encoder.parameters() if self.cond else [])
            ])

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.hp.optim_ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint) -> None:
        if self.hp.optim_ema:
            self.ema.load_state_dict(checkpoint["ema"])
    
    @contextlib.contextmanager
    def ema_average(self, activate=True):
        if activate:
            with self.ema.average_parameters() as ctx:
                yield ctx
        else:
            with contextlib.nullcontext() as ctx:
                yield ctx

    def stdg_noise_seeded(self, *dims, seed: typing.Optional[int] = None):
        if seed is not None:
            _rngstate = torch.get_rng_state()
            torch.manual_seed(seed)
        _tmp = torch.randn(*dims, device=self.device)
        if seed is not None:
            torch.set_rng_state(_rngstate)
        return _tmp
    
    def create_batch_with_utilities(self, padded_seq, lens, seed=None):
        # padded_seq: (BxTxF) shape
        # lens: (B,) shaped long tensor to denote original length of each sample
        batch_size, = lens.shape
        padded_seq, timestamps = padded_seq[..., :self.elem_dim], padded_seq[..., self.elem_dim:]

        batch = {}  # Keys: noise_target, timestamps, lens, noise_t, noisy_points, t

        # different 't's for different sample in the batch
        t = torch.randint(1, self.hp.noise_T + 1, size=(batch_size, ))

        g_noise = self.stdg_noise_seeded(*padded_seq.shape, seed=seed)

        batch['timestamps'] = timestamps
        batch['lens'] = lens
        batch['noise_t'] = self.pe[t - 1, :]
        batch['t'] = t - 1
        batch['noisy_points'] = padded_seq * self.sqrt_alpha_bar[t - 1, None, None] \
            + g_noise * self.sqrt_one_min_alpha_bar[t - 1, None, None]
        batch['target'] = g_noise

        return batch

    def ncsn_loss(self, score, noise_target, lens, t):
        pad_mask = make_pad_mask_for_transformer(lens, total_length=score.shape[1], device=lens.device)
        unreduced_loss = (score - noise_target).pow(2).mean(-1)
        masked_loss = (unreduced_loss * (~pad_mask).float()) / lens.unsqueeze(-1)
        per_sample_loss = masked_loss.sum(-1)  # sum along length since already divided by lengths
        return per_sample_loss.mean()
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(),
                lr=self.hp.optim_lr,
                weight_decay=self.hp.optim_decay)
        if self.hp.optim_sched == 'steplr':
            sched = torch.optim.lr_scheduler.StepLR(optim,
                step_size=1,
                gamma=self.hp.optim_gamma
            )
        elif self.hp.optim_sched == 'onecyclelr':
            steps_per_epoch = len(self.trainer.datamodule.train_dataset) \
                // self.trainer.datamodule.batch_size
            total_epochs = self.trainer.max_epochs
            total_steps = steps_per_epoch * total_epochs
            total = total_epochs if self.hp.optim_interval == 'epoch' else total_steps
            warmup_fraction = self.hp.optim_warmup / total
            sched = torch.optim.lr_scheduler.OneCycleLR(optim,
                max_lr=self.hp.optim_lr,
                total_steps=total,
                anneal_strategy='linear',
                cycle_momentum=True,
                pct_start=warmup_fraction,
                div_factor=self.hp.optim_div_factor,
                final_div_factor=1000
            )
        else:
            raise NotImplementedError('scheduler not known/implemented')
        
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': sched,
                'frequency': 1,
                'interval': self.hp.optim_interval
            }
        }
    
    def create_posvel_aug_input(self, points):
        if self.hp.repr.value.startswith('threeseqdel'):
            points_vel = points
            points_pos = torch.cumsum(points[..., :-1], dim=1)
        elif self.hp.repr.value.startswith('threeseqabs'):
            points_vel = torch.cat([
                points[:, 0, None, :-1],
                (points[:, 1:, :-1] - points[:, :-1, :-1])
            ], 1)
            points_pos = points
        else:
            raise NotImplementedError('ReprType not implemented')
        
        return points_pos, points_vel

    def forward(self, noisy_points, seq_pe, lens, noise_t, cond_latent):
        noisy_points_pos, noisy_points_vel = self.create_posvel_aug_input(noisy_points)
        
        if self.hp.modeltype == SketchDiffusion.ModelType.transformer:
            origin = torch.zeros(noisy_points.size(0), 1, 3, dtype=self.dtype, device=self.device, requires_grad=False)
            noisy_points_pos = torch.cat([origin[..., :noisy_points_pos.shape[-1]], noisy_points_pos], 1)
            noisy_points_vel = torch.cat([origin[..., :noisy_points_vel.shape[-1]], noisy_points_vel], 1)
            seq_pe = torch.cat([self._create_seq_embeddings(origin[..., :1]), seq_pe], dim=1) # add origin timestamp
            lens = lens + 1 # due an added origin

        with self.ema_average(not self.training and self.hp.optim_ema):
            out = self.scorefn((noisy_points_pos, noisy_points_vel), seq_pe, lens, noise_t, cond_latent)
        
        return out
    
    def _create_seq_embeddings(self, timestamps):
        if self.seq_pe_dim > 0:
            batch_size, max_len, _ = timestamps.shape
            timestamps = timestamps.permute(2, 0, 1)
            temb = random_fourier_encoding_dyn(timestamps.view(1, batch_size * max_len), self.seq_proj_W, scale=4.)
            return temb.view(batch_size, max_len, self.seq_pe_dim)
        else:
            return None
    
    def encode(self, *args):
        if self.cond:
            with self.ema_average(not self.training and self.hp.optim_ema):
                return self.encoder(*args)
        else:
            return None, 0.

    def training_step(self, batch, batch_idx):
        cond_batch, batch = batch

        batch = self.create_batch_with_utilities(*batch)
        cond_latent, kl_loss = self.encode(cond_batch)
        score = self(batch['noisy_points'], self._create_seq_embeddings(batch['timestamps']),
                    batch['lens'], batch['noise_t'], cond_latent)
        loss = self.ncsn_loss(score, batch['target'], batch['lens'], batch['t'])
        self.log('train/loss', loss, prog_bar=True)
        if self.hp.vae_weight != 0.:
            kl_loss = kl_loss.mean()
            self.log('train/kl', kl_loss, prog_bar=False)
            kl_annealing_factor = min(max(self.global_step - self.hp.vae_kl_anneal_start, 0.) / \
                                                    (self.hp.vae_kl_anneal_end - self.hp.vae_kl_anneal_start), 1.)
            self.log('train/kl_factor', kl_annealing_factor, prog_bar=False)
        else:
            kl_annealing_factor = 0.
        return loss + \
            self.hp.vae_weight * kl_annealing_factor * kl_loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)

        # on-the-fly testing while training
        if batch_idx == 0 and (self.current_epoch + 0) % 300 == 0 and self.device.index == 0:
            save_file_path = os.path.join(self.trainer.log_dir,
                f"ddpm1.pdf")
            ret_dict = self.reconstruction(batch, SketchDiffusion.SamplingAlgo.ddpm, langevin_strength=1.)
            self.fig.savefig(save_file_path, bbox_inches='tight')
            self.cache_reverse_process(ret_dict["all"], -1, ret_dict["lens"], idx=batch_idx, prefix='ddpm1')
            
            save_file_path = os.path.join(self.trainer.log_dir,
                f"ddpm.5.pdf")
            ret_dict = self.reconstruction(batch, SketchDiffusion.SamplingAlgo.ddpm, langevin_strength=0.5)
            self.fig.savefig(save_file_path, bbox_inches='tight')
            self.cache_reverse_process(ret_dict["all"], -1, ret_dict["lens"], idx=batch_idx, prefix='ddpm.5')

            save_file_path = os.path.join(self.trainer.log_dir,
                f"ddim_reco.pdf")
            ret_dict = self.reconstruction(batch, SketchDiffusion.SamplingAlgo.ddim, langevin_strength=0.)
            self.fig.savefig(save_file_path, bbox_inches='tight')
            self.cache_reverse_process(ret_dict["all"], -1, ret_dict["lens"], idx=batch_idx, prefix='ddim_reco')

            save_file_path = os.path.join(self.trainer.log_dir,
                f"ddim_gen.pdf")
            ret_dict = self.reconstruction(batch, SketchDiffusion.SamplingAlgo.ddim, langevin_strength=0., generation=True)
            self.fig.savefig(save_file_path, bbox_inches='tight')
            self.cache_reverse_process(ret_dict["all"], -1, ret_dict["lens"], idx=batch_idx, prefix='ddim_gen')

        return loss
    
    def validation_epoch_end(self, losses_for_batches) -> None:
        valid_loss = sum(losses_for_batches) / len(losses_for_batches)
        self.log('valid/loss', valid_loss, prog_bar=True)

    def on_test_start(self) -> None:
        ts = torch.linspace(1, self.hp.noise_T, self.hp.noise_T,
                            dtype=self.dtype, device=self.device) / self.hp.noise_T
        self.pe = random_fourier_encoding_dyn(ts[None, ...], self.pe_proj_W, scale=4.) \
            if self.hp.time_embedding == SketchDiffusion.TimeEmbedding.randomfourier else \
                positionalencoding1d(self.pe_dim, self.hp.noise_T, N=self.hp.noise_T,
                                        dtype=self.dtype, device=self.device)

        n_viz = self.hp.test_n_viz * 2 if self.hp.text_viz_process == SketchDiffusion.VizProcess.both else self.hp.test_n_viz
        cviz = CustomViz(self.hp.test_n_sample_viz, n_viz, compact_mode=self.hp.test_viz_fig_compact)
        self.fig, self.ax = cviz, cviz

    def cache_reverse_process(self, all_points_t, t, lens, idx, prefix='gen'):
        # npz_save_path = os.path.join(self.trainer.log_dir, f'{prefix}_rev_{idx}.npz')
        # with open(npz_save_path, 'wb') as f:
        #     np.savez(f, reverse=all_points_t.cpu().numpy(), lens=lens.cpu().numpy())
        samples = all_points_t[t, ...]
        samples = torch.split(samples, self.ax.shape[0], dim=0)
        lens = torch.split(lens, self.ax.shape[0], dim=0)
        for j in range(self.ax.shape[1]):
            try:
                self.draw_on_seq(samples[j], lens[j], j)
            except:
                for i in range(self.ax.shape[0]):
                    self.ax[i, j].cla()
                    self.ax[i, j].axis('off')
        save_file_path = os.path.join(self.trainer.log_dir, f'{prefix}_{idx}.svg')
        self.fig.savefig(save_file_path, bbox_inches='tight')

    def test_step(self, batch, batch_idx):
        if self.hp.test_recon:
            save_file_path = os.path.join(self.trainer.log_dir, f'diff_{batch_idx}.svg')
            rev_dict = self.reconstruction(batch, self.hp.test_sampling_algo, self.hp.test_variance_strength,
                        generation=True, partial_t=self.hp.test_partial_T)
            self.fig.savefig(save_file_path, bbox_inches='tight')
            if self.hp.test_save_everything:
                _, (vels, lens) = batch
                vels, ts = vels[..., :self.elem_dim], vels[..., self.elem_dim:]
                orig, orig_len = self.velocity_to_position(vels, lens)
                # self.cache_reverse_process(orig[None, ...], -1, orig_len, idx=batch_idx, prefix='orig')
                self.cache_reverse_process(rev_dict["all"], -1, rev_dict["lens"], idx=batch_idx, prefix=f'gen')

        if self.hp.test_interp:
            save_file_path = os.path.join(self.trainer.log_dir, f'interp_{batch_idx}.svg')
            _ = self.interpolation(batch, self.hp.test_sampling_algo, langevin_strength=0.)
            self.fig.savefig(save_file_path, bbox_inches='tight')
    
    def velocity_to_position(self, points, lens):
        B, _, _ = points.shape

        points = torch.cat([
            torch.zeros(B, 1, self.elem_dim, dtype=points.dtype, device=points.device),
            points
        ], dim=1)
        lens = lens + 1  # there is the extra initial point along length
        
        if self.hp.repr.value.startswith('threeseqdel'):            
            # last one is pen-up bit -- leave it as is
            points[..., :-1] = torch.cumsum(points[..., :-1], dim=1)
        else:
            # this incorporates THREESEQABS
            pass
        
        points[..., -1][points[..., -1] > 0.8] = 1.
        points[..., -1][points[..., -1] < 0.8] = 0.

        return points, lens

    def draw_on_seq(self, points, lens, t_):        
        points = points.detach().cpu().numpy()
        lens = lens.cpu().numpy()

        cm = get_cmap('copper') # I like this one
        for b in range(self.hp.test_n_sample_viz):
            sample_seq: Sketch = Sketch.from_threeseqabs(points[b, :lens[b], :])
            sample_seq.draw(self.ax[b, t_], color=cm, cla=True, scatter=False)
            
    def forward_diffusion(self, velocs, lens, draw=True, end_t=None):
        viz_t = np.linspace(0, end_t or self.hp.noise_T, self.hp.test_n_viz, dtype=np.int64)

        if draw: # the original sample
            points, points_len = self.velocity_to_position(velocs, lens)
            self.draw_on_seq(points, points_len, self.t_)
            self.t_ += 1
        
        for t in viz_t[1:]:
            g_noise = self.stdg_noise_seeded(*velocs.shape)

            velocs_t = velocs * self.sqrt_alpha_bar[t - 1, None, None] \
                + g_noise * self.sqrt_one_min_alpha_bar[t - 1, None, None]

            if draw:
                points_t, points_len = self.velocity_to_position(velocs_t, lens)
                self.draw_on_seq(points_t, points_len, self.t_)
                self.t_ += 1

        return velocs_t

    def reverse_purturb_DDPM(self, points, timestamps, t, lens, cond_latent, steps, noise_weight=1.):
        now, now_index = steps[t], steps[t] - 1

        score = self(points, timestamps, lens, self.pe[now_index, :].repeat(points.shape[0], 1), cond_latent)
        k1 = 1. / torch.sqrt(self.alphas[now_index])
        k2 = (1. - self.alphas[now_index]) / self.sqrt_one_min_alpha_bar[now_index]
        mean = k1 * (points - k2 * score)
        
        gen_noise = self.stdg_noise_seeded(*points.shape) * torch.sqrt(self.beta_tilde[now_index]) \
            if now > 1 else 0.
        
        points = mean + gen_noise * noise_weight
        return points
    
    def reverse_purturb_DDIM(self, points, timestamps, t, lens, cond_latent, steps, noise_weight=0.):
        now, now_index = steps[t], steps[t] - 1
        
        score = self(points, timestamps, lens, self.pe[now_index, :].repeat(points.shape[0], 1), cond_latent)
        x0_pred = (points - self.sqrt_one_min_alpha_bar[now_index] * score) \
            / self.sqrt_alpha_bar[now_index]

        if now > 1:
            prev, prev_index = steps[t + 1], steps[t + 1] - 1

            # generalized version of DDIM sampler, with explicit \sigma_t
            s1 = self.sqrt_one_min_alpha_bar[prev_index] / self.sqrt_one_min_alpha_bar[now_index]
            s2 = torch.sqrt(1. - self.alpha_bar[now_index] / self.alpha_bar[prev_index])
            sigma = (s1 * s2) * noise_weight # additional control for the noise

            gen_noise = self.stdg_noise_seeded(*points.shape)

            points = self.sqrt_alpha_bar[prev_index] * x0_pred \
                + torch.sqrt(1. - self.alpha_bar[prev_index] - sigma**2) * score \
                + gen_noise * sigma
        else:
            points = x0_pred

        return points
    
    def forward_purturb_DDIM(self, points, timestamps, t, lens, cond_latent, steps, noise_weight=1.):
        # DDIM's reverse of the reverse process -- integrating the ODE backwards
        now, now_index = steps[t], steps[t] - 1
        prev, prev_index = steps[t] - 1, steps[t] - 2
        
        score = self(points, timestamps, lens, self.pe[prev_index, :].repeat(points.shape[0], 1), cond_latent) \
            if prev != 0 else 0.

        xT_pred = (points - self.sqrt_one_min_alpha_bar[prev_index] * score) \
            / (self.sqrt_alpha_bar[prev_index] if prev != 0 else 1.)

        points = self.sqrt_alpha_bar[now_index] * xT_pred + self.sqrt_one_min_alpha_bar[now_index] * score
        return points

    def reverse_diffusion(self, points, timestamps, lens, cond_latent, sampling_algo, langevin_strength, draw=True, start_t=None):
        veloc_t = points

        if start_t is not None:
            assert sampling_algo == SketchDiffusion.SamplingAlgo.ddpm, \
                'partially stopping diffusion makes sense only for stochastic sampler'
            assert start_t <= self.hp.noise_T, f"partial stopping time must be less that T={self.hp.noise_T}"

        inference_steps, sampling_fn = {
            SketchDiffusion.SamplingAlgo.ddpm: (
                np.linspace(start_t or self.hp.noise_T, 1, start_t or self.hp.noise_T, dtype=np.int64),
                SketchDiffusion.reverse_purturb_DDPM
            ),
            SketchDiffusion.SamplingAlgo.ddim: (
                np.linspace(self.hp.noise_T, 1, self.hp.noise_T, dtype=np.int64),
                SketchDiffusion.reverse_purturb_DDIM
            ),
            SketchDiffusion.SamplingAlgo.fddim: (
                np.linspace(1, self.hp.noise_T, self.hp.noise_T, dtype=np.int64),
                SketchDiffusion.forward_purturb_DDIM
            )
        }[sampling_algo]

        viz_t = np.linspace(self.hp.noise_T, 1, self.hp.test_n_viz, dtype=np.int64)

        points_t_all_steps = []
        for t in range(inference_steps.shape[0]):
            veloc_t = sampling_fn(self, veloc_t, timestamps, t, lens, cond_latent,
                                    inference_steps, noise_weight=langevin_strength)
            points_t, points_len = self.velocity_to_position(veloc_t, lens)
            if inference_steps[t] in viz_t:
                if draw:
                    self.draw_on_seq(points_t, points_len, self.t_)
                    self.t_ += 1
            
            if self.hp.test_save_everything:
                points_t_all_steps.append(points_t)

        return {
            "orig_last": veloc_t,
            "last": points_t,
            "all": torch.stack(points_t_all_steps, 0) if self.hp.test_save_everything else [ ],
            "lens": points_len
        }

    def reconstruction(self, batch, sampling_algo, langevin_strength, generation=False, partial_t=None):
        assert sampling_algo != SketchDiffusion.SamplingAlgo.fddim, "FDDIM is not to be used by public API"
        
        self.t_ = 0
        cond_batch, (points, lens) = batch

        cond_latent, _ = self.encode(cond_batch)
        points, timestamps = points[..., :self.elem_dim], points[..., self.elem_dim:]
        
        if sampling_algo != SketchDiffusion.SamplingAlgo.ddim:
            diffused = self.forward_diffusion(points, lens,
                        draw=self.hp.text_viz_process == SketchDiffusion.VizProcess.forward \
                            or self.hp.text_viz_process == SketchDiffusion.VizProcess.both,
                            end_t=partial_t)

            if partial_t is None:
                perm = torch.randperm(lens.size(0))
                lens = lens[perm] # reset lengths
                diffused = torch.randn_like(diffused)
        else:
            # execute forward DDIM (feature extraction)
            diffused = self.reverse_diffusion(points, self._create_seq_embeddings(timestamps), lens, cond_latent,
                SketchDiffusion.SamplingAlgo.fddim, langevin_strength,
                draw=self.hp.text_viz_process == SketchDiffusion.VizProcess.forward \
                    or self.hp.text_viz_process == SketchDiffusion.VizProcess.both)
            diffused = diffused["orig_last"]
            if generation:
                diffused = torch.randn_like(diffused)
        
        rev_dict = self.reverse_diffusion(diffused, self._create_seq_embeddings(timestamps), lens, cond_latent,
                sampling_algo, langevin_strength,
                draw=self.hp.text_viz_process == SketchDiffusion.VizProcess.backward \
                    or self.hp.text_viz_process == SketchDiffusion.VizProcess.both, start_t=partial_t)
        return rev_dict

    def interpolation(self, batch, sampling_algo, langevin_strength=0.):
        assert sampling_algo != SketchDiffusion.SamplingAlgo.fddim, "FDDIM is not to be used by public API"
        
        cond_batch1, (points1, lens1) = batch  # samples not really needed, only lens
        
        # random shuffle before executing generation
        perm = torch.randperm(points1.shape[0], device=points1.device)
        points2, lens2 = points1[perm, ...], lens1[perm]

        cond_latent1, _ = self.encode(cond_batch1)
        cond_latent2 = cond_latent1[perm, ...] if self.cond else None
        
        points1, timestamps1 = points1[..., :self.elem_dim], points1[..., self.elem_dim:]
        points2, timestamps2 = points2[..., :self.elem_dim], points2[..., self.elem_dim:]

        prior1 = torch.randn_like(points1)
        prior2 = torch.randn_like(points2)

        for a_, alpha in enumerate(np.linspace(0., 1., self.ax.shape[1])):
            if not self.cond:
                prior = prior1 * (1. - alpha) + prior2 * alpha
                lens = lens1
                cond_latent = None
            else:
                prior = prior1
                lens = lens1
                cond_latent = cond_latent1 * (1. - alpha) + cond_latent2 * alpha

            if self.hp.modeltype == SketchDiffusion.ModelType.transformer:
                raise NotImplementedError('interpolation with transformer model not yet implemented')
            
            recon_dict = self.reverse_diffusion(prior, None, lens, cond_latent,
                        sampling_algo, langevin_strength=0., draw=False)
            self.draw_on_seq(recon_dict["last"], recon_dict["lens"], a_)


if __name__ == '__main__':
    cli = LightningCLI(SketchDiffusion, GenericDM, run=True,
                       subclass_mode_data=True,
                       parser_kwargs={"parser_mode": "omegaconf"},
                       trainer_defaults={
                           'callbacks': [
                               LearningRateMonitor(logging_interval='step'),
                               ModelCheckpoint(monitor='valid/loss', filename='model', save_last=True),
                               TQDMProgressBar(refresh_rate=1 if sys.stdin.isatty() else 0)
                           ]
                       })
