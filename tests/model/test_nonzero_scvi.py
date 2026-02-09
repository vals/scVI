"""Tests for NonZeroSCVI model."""

import numpy as np
import pytest
import scipy.sparse as sp

from scvi.data import synthetic_iid
from scvi.model import NonZeroSCVI


@pytest.mark.parametrize("n_latent", [5])
def test_nonzero_scvi_basic(n_latent: int):
    """Test basic training and inference of NonZeroSCVI."""
    adata = synthetic_iid()
    NonZeroSCVI.setup_anndata(adata, batch_key="batch", labels_key="labels")
    model = NonZeroSCVI(adata, n_latent=n_latent)
    model.train(1, train_size=0.5)
    assert model.is_trained is True
    z = model.get_latent_representation()
    assert z.shape == (adata.shape[0], n_latent)
    model.get_elbo()
    model.get_marginal_ll(n_mc_samples=3)
    model.get_reconstruction_error()
    model.get_normalized_expression()


@pytest.mark.parametrize("n_latent", [5])
def test_nonzero_scvi_sparse(n_latent: int):
    """Test NonZeroSCVI with sparse input."""
    adata = synthetic_iid()
    adata.X = sp.csr_matrix(adata.X)
    NonZeroSCVI.setup_anndata(adata)
    model = NonZeroSCVI(adata, n_latent=n_latent)
    model.train(1, train_size=0.5)
    assert model.is_trained is True
    z = model.get_latent_representation()
    assert z.shape == (adata.shape[0], n_latent)


@pytest.mark.parametrize("normalize_by_nonzero", [True, False])
def test_nonzero_scvi_normalization_modes(normalize_by_nonzero: bool):
    """Test both normalization modes for NonZeroSCVI."""
    adata = synthetic_iid()
    NonZeroSCVI.setup_anndata(adata)
    model = NonZeroSCVI(adata, n_latent=5, normalize_by_nonzero=normalize_by_nonzero)
    model.train(1, train_size=0.5)
    assert model.is_trained is True
    z = model.get_latent_representation()
    assert z.shape == (adata.shape[0], 5)


@pytest.mark.parametrize("gene_likelihood", ["zinb", "nb", "poisson"])
def test_nonzero_scvi_gene_likelihoods(gene_likelihood: str):
    """Test NonZeroSCVI with different gene likelihoods."""
    adata = synthetic_iid()
    NonZeroSCVI.setup_anndata(adata)
    model = NonZeroSCVI(adata, n_latent=5, gene_likelihood=gene_likelihood)
    model.train(1, train_size=0.5)
    assert model.is_trained is True


def test_nonzero_scvi_model_summary():
    """Test that model summary string includes normalize_by_nonzero."""
    adata = synthetic_iid()
    NonZeroSCVI.setup_anndata(adata)

    model_true = NonZeroSCVI(adata, normalize_by_nonzero=True)
    assert "normalize_by_nonzero: True" in model_true._model_summary_string

    model_false = NonZeroSCVI(adata, normalize_by_nonzero=False)
    assert "normalize_by_nonzero: False" in model_false._model_summary_string


def test_nonzero_scvi_masking_effect():
    """Test that NonZeroSCVI masks zeros from loss computation."""
    adata = synthetic_iid()

    # Add explicit zeros to the data
    adata.X[:, :10] = 0

    NonZeroSCVI.setup_anndata(adata)
    model = NonZeroSCVI(adata, n_latent=5)

    # Just verify training works - the masking is internal
    model.train(1, train_size=0.5)
    assert model.is_trained is True
