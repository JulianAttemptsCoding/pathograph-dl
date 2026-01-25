"""
ST-MM-GNN Layer A Lightning Module.

Wraps STMMGraphWaveNet for training with masked loss and metrics.
"""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class STMMPLModule(pl.LightningModule):
    """LightningModule for ST-MM-GNN Layer A training."""
    
    def __init__(self, model: nn.Module, lr: float = 0.001, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
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

        # Empty-mask guard: keep autograd connected to logits
        if not torch.any(valid):
            return logits.sum() * 0.0

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
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Forward pass
        logits = self.forward(batch)
        
        # Compute masked loss
        loss = self._masked_bce_with_logits(
            logits,
            batch['y_next'],
            batch['y_mask'],
        )
        
        # Log
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Compute accuracy
        acc = self._masked_accuracy(logits, batch['y_next'], batch['y_mask'])
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
