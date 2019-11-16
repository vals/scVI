# -*- coding: utf-8 -*-
"""Main module."""
from typing import Dict, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.models.modules import Encoder, DecoderSCVI, DecoderCAVI
from scvi.models.log_likelihood import (
    log_zinb_positive,
    log_nb_positive,
    log_mixture_nb,
)

# VAE model
class CAVI(nn.Module):
    r'''Latent Cell Assignment using Variational Inference
    '''

    def __init__(
        self,
        n_input_genes: int,
        n_labels: int,
        marker_gene_matrix: np.array,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.2,
        gene_dispersion: str = 'gene',
        log_variational: bool = True,
        reconstruction_loss: str = 'nb'
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes

        if self.gene_dispersion == 'gene':
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        else:  # gene-cell
            pass

        # z encoder goes from the n_input_genes-dimensional data to an
        # n_latent-dimensional latent space reprsentation
        self.z_encoder = Encoder(
            self.n_input_genes,
            self.n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate
        )

        # l encoder goes from n_input-dimensional data to 1-dimensional library size
        self.l_encoder = Encoder(
            self.n_input_genes,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate
        )

        # decoder goes from n_latent-dimensional space to n_labels probabilites
        self.decoder = DecoderCAVI(
            self.n_latent,
            self.n_labels,
            n_layers=n_layers,
            n_hidden=n_hidden
        )

        # Marker gene matrix
        self.rho = torch.from_numpy(marker_gene_matrix)

        # Marker gene DE matrix
        self.delta = torch.nn.Parameter(torch.randn(self.rho.shape))

        # Base expression
        self.beta = torch.nn.Parameter(torch.randn(n_input_genes))

    def inference(self, x, n_samples=1):

        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)
        
        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library = self.l_encoder(x_)

        if n_samples > 1:
            1  # Let's just start with one sample

        ppi_scale = self.decoder(z)

        px_r = self.px_r
        px_r = torch.exp(px_r)

        deltarho = F.softplus(self.delta) * self.rho

        px_scale = F.softmax(deltarho + self.beta[:, None], dim=0)
        px_rate = library[:, None, None] * px_scale[None, :, :]

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            ppi_scale=ppi_scale,
            qz_m=qz_m,
            qz_v=qz_v,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library
        )

    def forward(self, x, local_l_mean, local_l_var):
        r''' Return reconstruction loss and KL divergences
        '''
        outputs = self.inference(x)
        qz_m = outputs['qz_m']
        qz_v = outputs['qz_v']
        ql_m = outputs['ql_m']
        ql_v = outputs['ql_v']
        ppi_scale = outputs['ppi_scale']
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )
        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)
        kl_divergence = kl_divergence_z

        nll = -log_nb_positive(x[:, :, None], px_rate, px_r[None, :, None])
        reconst_loss = (ppi_scale[:, None, :] * nll).sum()

        return reconst_loss + kl_divergence_l, kl_divergence, 0.0
