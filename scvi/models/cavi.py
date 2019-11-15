# -*- coding: utf-8 -*-
"""Main module."""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.models.modules import Encoder, DecoderSCVI
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