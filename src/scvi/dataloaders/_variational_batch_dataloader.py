"""DataLoader for variational batch representation with pseudobulk support."""

import torch
import numpy as np
from typing import Dict, Any
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from ._ann_dataloader import AnnDataLoader


class VariationalBatchDataLoader(AnnDataLoader):
    """DataLoader that provides both single-cell data and batch pseudobulks.
    
    This dataloader extends AnnDataLoader to provide precomputed pseudobulk 
    representations for each batch alongside the regular single-cell data.
    This enables variational batch encoding where each batch can be encoded
    from its pseudobulk representation.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        pseudobulk_adata=None,
        batch_key: str | None = None,
        **kwargs,
    ):
        super().__init__(adata_manager, **kwargs)
        
        # Get batch key from registry if not provided
        if batch_key is None:
            batch_key = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)["original_key"]
        
        # Precompute pseudobulk if not provided
        if pseudobulk_adata is None:
            from scvi.data._utils import make_pseudobulk_batches
            # Use subset of data if indices are specified
            if hasattr(self, 'indices') and self.indices is not None:
                subset_adata = adata_manager.adata[self.indices]
            else:
                subset_adata = adata_manager.adata
            pseudobulk_adata = make_pseudobulk_batches(subset_adata, batch_key)
        
        # Store pseudobulk data for encoding
        self.pseudobulk_data = pseudobulk_adata
        self.batch_key = batch_key
        self.adata_manager = adata_manager
    
    def __iter__(self):
        """Iterate over batches, providing both single-cell data and pseudobulks."""
        for batch_data in super().__iter__():
            # Get batch indices for current minibatch
            batch_indices = batch_data[REGISTRY_KEYS.BATCH_KEY].squeeze(-1)
            unique_batches = torch.unique(batch_indices)
            
            # Get batch category names for unique batches in this minibatch
            batch_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
            batch_categories = batch_registry["categorical_mapping"]
            
            # Get pseudobulk data for the unique batches in this minibatch
            pseudobulk_counts = []
            for batch_idx in unique_batches:
                batch_name = batch_categories[batch_idx.item()]
                # Find this batch in pseudobulk data
                pb_mask = self.pseudobulk_data.obs[self.batch_key] == batch_name
                if pb_mask.any():
                    pb_counts = self.pseudobulk_data.X[pb_mask][0]  # Get first (should be only) match
                else:
                    # If batch not found in pseudobulk, create zero counts (shouldn't happen)
                    pb_counts = np.zeros(self.pseudobulk_data.n_vars)
                pseudobulk_counts.append(pb_counts)
            
            pseudobulk_batch = torch.tensor(np.stack(pseudobulk_counts), dtype=torch.float32)
            
            # Add pseudobulk data to batch
            batch_data["pseudobulk_counts"] = pseudobulk_batch
            batch_data["unique_batch_indices"] = unique_batches
            
            yield batch_data