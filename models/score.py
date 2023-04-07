import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import make_pad_mask_for_transformer


class ConditionalTransformerEncoder(nn.Module):

    def __init__(self, n_input, n_internal, n_layers, n_head, causal=False, dropout=0.) -> None:
        super().__init__()

        self.n_input = n_input
        self.n_internal = n_internal
        self.n_layers = n_layers
        self.n_head = n_head
        self.causal = causal
        self.dropout = dropout

        self.embedder = nn.Linear(self.n_input, self.n_internal)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.n_internal,
                self.n_head,
                dim_feedforward=self.n_internal * 2,
                batch_first=True, dropout=self.dropout, activation=F.silu
            ),
            num_layers=self.n_layers
        )

    def forward(self, noisy, lens):
        _, max_len, _ = noisy.shape
        len_padd_mask = make_pad_mask_for_transformer(lens, max_len, noisy.device)

        if self.causal:
            I = torch.eye(max_len, dtype=noisy.dtype, device=noisy.device)
            attn_mask = (torch.cumsum(I, -1) - I) == 1.
        else:
            attn_mask = None

        input_emb = self.embedder(noisy)
        output = self.transformer(input_emb, mask=attn_mask, src_key_padding_mask=len_padd_mask)

        return output


class ConditionalBiRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0., causal=False) -> None:
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.causal = causal

        self.rnn = nn.GRU(self.n_input, self.n_hidden, self.n_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=not self.causal)

        directionality = 2 if not self.causal else 1
        self.out_proj = nn.Linear(
            self.n_hidden * directionality,
            self.n_hidden
        )

    def forward(self, noisy, lens):
        noisy_packed = pack_padded_sequence(noisy, lens.cpu(), batch_first=True, enforce_sorted=False)
        hid, _ = self.rnn(noisy_packed)
        out_unpacked, _ = pad_packed_sequence(hid, batch_first=True)

        return self.out_proj(out_unpacked)


class ScoreFunc(nn.Module):

    def __init__(self, modeltype, *, inp_n_features=5, out_n_features=3, time_pe_features=2, seq_pe_features=2,
                 n_cond_features=0, n_head=4, n_layer=4, n_internal=64, causal=False, dropout=0.) -> None:
        super().__init__()

        self.modeltype = modeltype
        self.inp_n_features = inp_n_features
        self.out_n_features = out_n_features
        self.time_pe_features = time_pe_features  # for diffusion steps
        self.seq_pe_features = seq_pe_features  # for sequence time-stamps
        self.n_cond_features = n_cond_features # for conditioning
        self.n_internal = n_internal
        self.n_head = n_head
        self.n_layer = n_layer
        self.causal = causal
        self.dropout = dropout

        self.n_additionals = self.time_pe_features + self.seq_pe_features + self.n_cond_features
        self.n_total_features = self.inp_n_features + self.n_additionals

        if self.modeltype == 'birnn':
            self.model = ConditionalBiRNN(self.n_total_features, self.n_internal, self.n_layer,
                                        dropout=self.dropout, causal=self.causal)
        elif self.modeltype == 'transformer':
            self.model = ConditionalTransformerEncoder(self.n_total_features,
                            self.n_internal, self.n_layer, self.n_head, causal=self.causal, dropout=self.dropout)
        else:
            raise NotImplementedError(f"Unknown model type {self.modeltype.value}")

        self.final_proj = nn.Sequential(
            nn.Linear(self.n_internal * (2 if self.modeltype == 'transformer' else 1) \
                + self.n_additionals - self.seq_pe_features, self.out_n_features),
        )

    def forward(self, noisy, seq_pe, lens, time_pe, cond=None):
        noisy_pos, noisy_vel = noisy
        noisy = torch.cat([noisy_pos, noisy_vel], -1)

        if isinstance(cond, tuple):
            # This is 'threeseqabs_threeseqabseqsampledcond' repr.
            # But not a good way to check (TODO: better API)
            cond = torch.cat(cond, -1)

        batch_size, max_len, _ = noisy.shape

        time_pe = time_pe.unsqueeze(1).repeat(1, max_len, 1)
        
        if cond is not None:
            assert self.n_cond_features != 0, "conditioning is being done but no dimension allocated"
            if len(cond.shape) == 2:
                cond = cond.unsqueeze(1).repeat(1, max_len, 1)
            time_cond = torch.cat([time_pe, cond], -1)
        else:
            time_cond = time_pe
        
        if self.seq_pe_features > 0:
            additionals = torch.cat([seq_pe, time_cond], -1)
        else:
            additionals = time_cond
        
        output = self.model(
            torch.cat([noisy, additionals], -1),
            lens
        )

        if self.modeltype == 'birnn':
            return self.final_proj(torch.cat([output, time_cond], -1))
        else:
            conseq_cat_output = torch.cat([output[:, :-1, :], output[:, 1:, ]], -1)
            return self.final_proj(torch.cat([conseq_cat_output, time_cond[:, 1:, :]], -1))


class TransformerSetFeature(ConditionalTransformerEncoder):

    def __init__(self, n_internal, n_layers, n_head, n_latent, dropout=0., vae_weight=0.) -> None:
        # '+1' is for the extra feature for denoting feature extractor token
        super().__init__(2 + 1, n_internal, n_layers, n_head, causal=False, dropout=dropout)
        self.n_latent = n_latent
        self.vae_weight = vae_weight

        if self.vae_weight == 0.:
            self.latent_proj = nn.Sequential(
                nn.Linear(n_internal, self.n_latent),
                nn.Tanh()
            )
        else:
            self.latent_proj_mean = nn.Sequential(nn.Linear(n_internal, self.n_latent))
            self.latent_proj_logvar = nn.Sequential(nn.Linear(n_internal, self.n_latent))
    
    def forward(self, cond_batch):
        set_input, lens = cond_batch
        B, L, _ = set_input.shape
        # creating an extra feature extractor token
        pad_token = torch.zeros(B, L, 1, device=set_input.device, dtype=set_input.dtype)
        feat_token = torch.tensor([0., 0., 1.], device=set_input.device, dtype=set_input.dtype)
        feat_token = feat_token[None, None, :].repeat(B, 1, 1)
        set_input = torch.cat([set_input, pad_token], -1)
        set_input = torch.cat([feat_token, set_input], 1)
        lens = lens + 1 # extra token for feature extraction

        trans_out = super().forward(set_input, lens)

        if self.vae_weight == 0.:
            return self.latent_proj(trans_out[:, 0]), 0.
        else:
            mu = self.latent_proj_mean(trans_out[:, 0])
            logvar = self.latent_proj_logvar(trans_out[:, 0])
            posterior = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
            prior = torch.distributions.Normal(
                torch.zeros_like(mu),
                torch.ones_like(logvar)
            )
            return posterior.rsample(), torch.distributions.kl_divergence(posterior, prior)


class BiRNNEncoderFeature(ConditionalBiRNN):

    def __init__(self, n_hidden, n_layers, n_latent, dropout=0., vae_weight=0.) -> None:
        super().__init__(3, n_hidden, n_layers, dropout)
        self.out_proj = nn.Identity()
        self.vae_weight = vae_weight

        self.n_latent = n_latent

        if self.vae_weight == 0.:
            self.latent_proj = nn.Sequential(
                nn.Linear(2 * self.n_hidden, self.n_latent),
                nn.Tanh()
            )
        else:
            self.latent_proj_mean = nn.Sequential(nn.Linear(self.n_hidden, self.n_latent))
            self.latent_proj_logvar = nn.Sequential(nn.Linear(self.n_hidden, self.n_latent))
    
    def forward(self, cond_batch):
        batch, lens = cond_batch
        batch_size, max_len, _ = batch.shape
        batch = batch[..., :-1] # exclude the timestamps
        out = super().forward(batch, lens).view(batch_size, max_len, 2, self.n_hidden)
        out_fwd, out_bwd = out[:, :, 0, :], out[:, :, 1, :]
        fwd_feat = torch.gather(
            out_fwd,
            1,
            lens[:, None, None].repeat(1, 1, self.n_hidden) - 1
        ).squeeze()
        bwd_feat = out_bwd[:, 0, :]
        
        if self.vae_weight == 0.:
            return self.latent_proj(torch.cat([fwd_feat, bwd_feat], -1)), 0.
        else:
            mu = self.latent_proj_mean(torch.cat([fwd_feat, bwd_feat], -1))
            logvar = self.latent_proj_logvar(torch.cat([fwd_feat, bwd_feat], -1))
            posterior = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
            prior = torch.distributions.Normal(
                torch.zeros_like(mu),
                torch.ones_like(logvar)
            )
            return posterior.rsample(), torch.distributions.kl_divergence(posterior, prior)


class Lambda(nn.Module):

    def __init__(self, fn: typing.Callable) -> None:
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        # the extra zero is to make it compatible with other encoder
        return self.fn(x), 0.


class ClassEmbedding(nn.Module):

    def __init__(self, num_classes, emb_dim) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(self.num_classes, self.emb_dim)
    
    def forward(self, x):
        return self.emb(x), 0.