"""DataSplitter for variational batch representation."""

from scvi.dataloaders import DataSplitter
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from scvi.dataloaders._variational_batch_dataloader import VariationalBatchDataLoader


class VariationalDataSplitter(DataSplitter):
    """DataSplitter that uses VariationalBatchDataLoader for variational batch representation.
    
    This class automatically selects the appropriate dataloader based on the model's
    batch representation setting.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract model reference if provided
        self.model = kwargs.pop('model', None)
        super().__init__(*args, **kwargs)
        
        # Override data_loader_cls based on model's batch representation
        if self.model is not None:
            batch_representation = getattr(self.model.module, 'batch_representation', None)
            if batch_representation == 'variational':
                self.data_loader_cls = VariationalBatchDataLoader
                # Store model-specific parameters for VariationalBatchDataLoader
                from scvi import REGISTRY_KEYS
                self._variational_params = {
                    'pseudobulk_adata': getattr(self.model, '_pseudobulk_adata', None),
                    'batch_key': self.model.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)["original_key"]
                }
            else:
                self.data_loader_cls = AnnDataLoader
    
    def train_dataloader(self):
        """Create train data loader with variational batch support."""
        if self.data_loader_cls == VariationalBatchDataLoader:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.train_idx,
                shuffle=True,
                drop_last=self.drop_last,
                load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                **self._variational_params,
                **self.data_loader_kwargs,
            )
        else:
            return super().train_dataloader()
    
    def val_dataloader(self):
        """Create validation data loader with variational batch support."""
        if len(self.val_idx) > 0:
            if self.data_loader_cls == VariationalBatchDataLoader:
                return self.data_loader_cls(
                    self.adata_manager,
                    indices=self.val_idx,
                    shuffle=False,
                    drop_last=self.drop_last,
                    load_sparse_tensor=self.load_sparse_tensor,
                    pin_memory=self.pin_memory,
                    **self._variational_params,
                    **self.data_loader_kwargs,
                )
            else:
                return super().val_dataloader()
        else:
            return None
    
    def test_dataloader(self):
        """Create test data loader with variational batch support."""
        if len(self.test_idx) > 0:
            if self.data_loader_cls == VariationalBatchDataLoader:
                return self.data_loader_cls(
                    self.adata_manager,
                    indices=self.test_idx,
                    shuffle=False,
                    drop_last=False,
                    load_sparse_tensor=self.load_sparse_tensor,
                    pin_memory=self.pin_memory,
                    **self._variational_params,
                    **self.data_loader_kwargs,
                )
            else:
                return super().test_dataloader()
        else:
            return None