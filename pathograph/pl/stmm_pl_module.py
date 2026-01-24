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
        Compute masked binary cross-entropy with logits.
        
        Args:
            logits: (B, N, P) predicted logits
            targets: (B, N, P) ground truth (0/1)
            mask: (B, N, P) binary mask (1=observed, 0=missing)
        
        Returns:
            loss: scalar tensor
        """
        # Compute per-element loss
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply mask
        loss = loss * mask
        
        # Aggregate
        mask_sum = mask.sum()
        
        # Guard for empty mask (autograd-safe, device-safe)
        if mask_sum == 0:
            # Anchor to a trainable parameter to ensure requires_grad=True and correct device/dtype
            p0 = next(self.parameters())
            return p0.sum() * 0.0
        
        # Normalize by number of observed elements
        loss = loss.sum() / mask_sum
        
        return loss
    
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
