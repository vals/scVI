from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scvi import REGISTRY_KEYS
from scvi.module.base import EmbeddingModuleMixin

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


class EmbeddingMixin:
    """``EXPERIMENTAL`` Mixin class for initializing and using embeddings in a model.

    Must be used with a module that inherits from :class:`~scvi.module.base.EmbeddingModuleMixin`.

    Notes
    -----
    Lifecycle: experimental in v1.2.
    """

    @torch.inference_mode()
    def get_batch_representation(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Get the batch representation for a given set of indices."""
        if getattr(self.module, "batch_representation", None) == "variational":
            from scvi.dataloaders._variational_batch_dataloader import VariationalBatchDataLoader
            from scvi.data import make_pseudobulk_batches
            
            # For variational batch representation, handle unseen batches by extending categories
            if adata is None:
                adata = self.adata
            
            # Check if we need to transfer fields (new data) and allow category extension
            adata_manager = self.get_anndata_manager(adata)
            if adata_manager is None:
                # Transfer fields with extend_categories=True for unseen batches
                self._register_manager_for_instance(
                    self.adata_manager.transfer_fields(adata, extend_categories=True)
                )
            adata = self._validate_anndata(adata)
            
            # Get batch key from registry
            batch_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)["original_key"]
            
            # Create pseudobulk for the current data (may include unseen batches)
            subset_adata = adata[indices] if indices is not None else adata
            pseudobulk_adata = make_pseudobulk_batches(subset_adata, batch_key)
            
            # Use variational batch dataloader
            dataloader = VariationalBatchDataLoader(
                self.adata_manager,
                pseudobulk_adata=pseudobulk_adata,
                batch_key=batch_key,
                indices=indices,
                batch_size=128 if batch_size is None else batch_size,
                shuffle=False,
            )
            
            # Collect batch representations from each minibatch
            reps = []
            for tensors in dataloader:
                pseudobulk_data = tensors["pseudobulk_counts"]
                unique_batch_indices = tensors["unique_batch_indices"]
                batch_indices = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1)
                
                # Encode pseudobulk data to get batch representations
                pseudobulk_data = pseudobulk_data.to(self.module.device)
                qb, _ = self.module.batch_encoder(pseudobulk_data)
                batch_latent = self.module.batch_encoder.z_transformation(qb.rsample())
                
                # Map each cell to its batch representation
                batch_mapping = torch.searchsorted(unique_batch_indices, batch_indices)
                cell_batch_reps = batch_latent[batch_mapping]
                reps.append(cell_batch_reps)
            
            return torch.cat(reps).detach().cpu().numpy()

        if isinstance(self.module, EmbeddingModuleMixin):
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )
            key = REGISTRY_KEYS.BATCH_KEY
            tensors = [self.module.compute_embedding(key, tensors[key]) for tensors in dataloader]
            return torch.cat(tensors).detach().cpu().numpy()

        raise ValueError(
            "The current `module` must inherit from `EmbeddingModuleMixin` or use variational "
            "batch representation."
        )
