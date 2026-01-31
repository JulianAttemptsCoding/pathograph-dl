"""
ST-MM-GNN Layer A Lightning Module.

Wraps STMMGraphWaveNet for training with masked loss and metrics.
"""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathograph.metrics import per_pathogen_metrics, macro_nanmean


class STMMPLModule(pl.LightningModule):
    """LightningModule for ST-MM-GNN Layer A training."""
    
    def __init__(self, model: nn.Module, lr: float = 0.001, weight_decay: float = 0.0, num_pathogens: int = 8):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_pathogens = num_pathogens
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Epoch-level accumulators for validation and test
        # These will store detached CPU tensors and be computed at epoch end
        self.val_probs_batches = []
        self.val_targets_batches = []
        self.val_mask_batches = []
        
        self.test_probs_batches = []
        self.test_targets_batches = []
        self.test_mask_batches = []
        
    def _masked_bce_with_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blueprint-required loss:
        1) masked BCE per element
        2) per pathogen p: mean over observed (B,N)
        3) average these means equally across pathogens with any observations

        Shapes: logits/targets/mask are (B, N, P)
        """
        # Per-element loss
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Ensure mask is float for safe multiplication/sums
        mask_f = mask.to(dtype=loss.dtype)

        # Apply mask
        loss = loss * mask_f

        # Per-pathogen sums over (B,N)
        loss_sum_p = loss.sum(dim=(0, 1))       # (P,)
        mask_sum_p = mask_f.sum(dim=(0, 1))     # (P,)

        valid = mask_sum_p > 0

        # Empty-mask guard: raise explicit error instead of silent NaN
        if not torch.any(valid):
            raise RuntimeError(
                f"[STMM Loss] All {len(mask_sum_p)} pathogens have zero observed labels in batch! "
                f"mask_sum_p={mask_sum_p.tolist()}. "
                f"This indicates dataset filtering failed or batch construction is broken. "
                f"Check: require_target_observed should be True in config."
            )

        mean_p = torch.zeros_like(loss_sum_p)
        mean_p[valid] = loss_sum_p[valid] / mask_sum_p[valid]

        # Equal weighting across pathogens (macro over pathogens with observations)
        return mean_p[valid].mean()

    
    def _masked_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked binary accuracy.
        
        Args:
            logits: (B, N, P)
            targets: (B, N, P)
            mask: (B, N, P)
        
        Returns:
            accuracy: scalar tensor (0-1)
        """
        # Threshold logits at 0
        preds = (logits > 0.0).float()
        
        # Boolean mask
        mask_bool = mask > 0.5
        
        # Compute accuracy only on observed elements
        if mask_bool.sum() == 0:
            return logits.sum() * 0.0  # Guard
        
        correct = (preds == targets)[mask_bool].float()
        accuracy = correct.mean()
        
        return accuracy
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(batch)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Forward pass
        logits = self.forward(batch)
        
        # Compute masked loss
        loss = self._masked_bce_with_logits(
            logits,
            batch['y_next'],
            batch['y_mask'],
        )
        
        # Log
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Optional: compute accuracy
        acc = self._masked_accuracy(logits, batch['y_next'], batch['y_mask'])
        self.log('train_acc', acc, prog_bar=False, on_step=False, on_epoch=True)
        
        return loss
    
        return loss
    
    def on_validation_epoch_start(self):
        """Clear epoch accumulators."""
        self.val_probs_batches = []
        self.val_targets_batches = []
        self.val_mask_batches = []
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with AUROC and AUPRC."""
        # Forward pass
        logits = self.forward(batch)
        
        # Compute masked loss
        loss = self._masked_bce_with_logits(
            logits,
            batch['y_next'],
            batch['y_mask'],
        )
        
        # Log loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Compute accuracy
        acc = self._masked_accuracy(logits, batch['y_next'], batch['y_mask'])
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        # Accumulate epoch data for metric computation
        probs = torch.sigmoid(logits)  # (B, N, P)
        targets = batch['y_next']      # (B, N, P)
        mask = batch['y_mask']         # (B, N, P)
        
        # Store detached CPU tensors
        self.val_probs_batches.append(probs.detach().cpu())
        self.val_targets_batches.append(targets.detach().cpu())
        self.val_mask_batches.append(mask.detach().cpu())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log epoch-level metrics using accumulated data."""
        if len(self.val_probs_batches) == 0:
            return  # No validation data
        
        # Concatenate all batches
        probs_all = torch.cat(self.val_probs_batches, dim=0)  # (B_total, N, P)
        targets_all = torch.cat(self.val_targets_batches, dim=0)
        mask_all = torch.cat(self.val_mask_batches, dim=0)
        
        # Compute per-pathogen metrics using new library
        import numpy as np
        result = per_pathogen_metrics(probs_all, targets_all, mask_all)
        
        # Extract metrics
        auroc_array = result['auroc']  # (P,) numpy array
        auprc_array = result['auprc']
        valid_array = result['valid']
        pos_array = result['pos']
        neg_array = result['neg']
        
        # Log per-pathogen metrics
        for p in range(self.num_pathogens):
            self.log(f'val_auroc_p{p}', float(auroc_array[p]), on_epoch=True)
            self.log(f'val_auprc_p{p}', float(auprc_array[p]), on_epoch=True)
            self.log(f'val_valid_p{p}', float(valid_array[p]), on_epoch=True)
            self.log(f'val_pos_p{p}', float(pos_array[p]), on_epoch=True)
            self.log(f'val_neg_p{p}', float(neg_array[p]), on_epoch=True)
        
        # Compute macro metrics (NaN-excluding)
        macro_auroc, n_auroc = macro_nanmean(auroc_array)
        macro_auprc, n_auprc = macro_nanmean(auprc_array)
        
        self.log('val_auroc_macro', float(macro_auroc), prog_bar=True, on_epoch=True)
        self.log('val_auprc_macro', float(macro_auprc), prog_bar=True, on_epoch=True)
        self.log('val_n_valid_auroc', float(n_auroc), on_epoch=True)
        self.log('val_n_valid_auprc', float(n_auprc), on_epoch=True)
        
        # Log totals
        self.log('val_pos_total', float(pos_array.sum()), on_epoch=True)
        self.log('val_valid_total', float(valid_array.sum()), on_epoch=True)
        
        # Clear accumulators
        self.val_probs_batches = []
        self.val_targets_batches = []
        self.val_mask_batches = []
    
    def on_test_epoch_start(self):
        """Clear epoch accumulators."""
        self.test_probs_batches = []
        self.test_targets_batches = []
        self.test_mask_batches = []

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step with AUROC and AUPRC."""
        # Forward pass
        logits = self.forward(batch)
        
        # Compute masked loss
        loss = self._masked_bce_with_logits(
            logits,
            batch['y_next'],
            batch['y_mask'],
        )
        
        # Log loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Compute accuracy
        acc = self._masked_accuracy(logits, batch['y_next'], batch['y_mask'])
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        # Accumulate epoch data for metric computation
        probs = torch.sigmoid(logits)  # (B, N, P)
        targets = batch['y_next']      # (B, N, P)
        mask = batch['y_mask']         # (B, N, P)
        
        # Store detached CPU tensors
        self.test_probs_batches.append(probs.detach().cpu())
        self.test_targets_batches.append(targets.detach().cpu())
        self.test_mask_batches.append(mask.detach().cpu())
        
        return loss
    
    def on_test_epoch_end(self):
        """Compute and log epoch-level test metrics using accumulated data."""
        if len(self.test_probs_batches) == 0:
            return  # No test data
        
        # Concatenate all batches
        probs_all = torch.cat(self.test_probs_batches, dim=0)  # (B_total, N, P)
        targets_all = torch.cat(self.test_targets_batches, dim=0)
        mask_all = torch.cat(self.test_mask_batches, dim=0)
        
        # Compute per-pathogen metrics using new library
        import numpy as np
        result = per_pathogen_metrics(probs_all, targets_all, mask_all)
        
        # Extract metrics
        auroc_array = result['auroc']  # (P,) numpy array
        auprc_array = result['auprc']
        valid_array = result['valid']
        pos_array = result['pos']
        neg_array = result['neg']
        
        # Log per-pathogen metrics
        for p in range(self.num_pathogens):
            self.log(f'test_auroc_p{p}', float(auroc_array[p]), on_epoch=True)
            self.log(f'test_auprc_p{p}', float(auprc_array[p]), on_epoch=True)
            self.log(f'test_valid_p{p}', float(valid_array[p]), on_epoch=True)
            self.log(f'test_pos_p{p}', float(pos_array[p]), on_epoch=True)
            self.log(f'test_neg_p{p}', float(neg_array[p]), on_epoch=True)
        
        # Compute macro metrics (NaN-excluding)
        macro_auroc, n_auroc = macro_nanmean(auroc_array)
        macro_auprc, n_auprc = macro_nanmean(auprc_array)
        
        self.log('test_auroc_macro', float(macro_auroc), on_epoch=True)
        self.log('test_auprc_macro', float(macro_auprc), on_epoch=True)
        self.log('test_n_valid_auroc', float(n_auroc), on_epoch=True)
        self.log('test_n_valid_auprc', float(n_auprc), on_epoch=True)
        
        # Log totals
        self.log('test_pos_total', float(pos_array.sum()), on_epoch=True)
        self.log('test_valid_total', float(valid_array.sum()), on_epoch=True)
        
        # Clear accumulators
        self.test_probs_batches = []
        self.test_targets_batches = []
        self.test_mask_batches = []

    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
