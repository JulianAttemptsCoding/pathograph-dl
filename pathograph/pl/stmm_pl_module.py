"""
ST-MM-GNN Layer A Lightning Module.

Wraps STMMGraphWaveNet for training with masked loss and metrics.
"""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


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
        
        # Per-pathogen AUROC and AUPRC for validation
        self.val_auroc = nn.ModuleList([
            BinaryAUROC() for _ in range(num_pathogens)
        ])
        self.val_auprc = nn.ModuleList([
            BinaryAveragePrecision() for _ in range(num_pathogens)
        ])
        
        # Per-pathogen AUROC and AUPRC for test
        self.test_auroc = nn.ModuleList([
            BinaryAUROC() for _ in range(num_pathogens)
        ])
        self.test_auprc = nn.ModuleList([
            BinaryAveragePrecision() for _ in range(num_pathogens)
        ])
        
        # Accumulators for counts
        self.val_pos_total = 0
        self.test_pos_total = 0
        
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
        """Reset counters."""
        self.val_pos_total = 0
        
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
        
        # Update per-pathogen AUROC and AUPRC with masked samples
        probs = torch.sigmoid(logits)  # (B, N, P)
        targets = batch['y_next']      # (B, N, P)
        mask = batch['y_mask']         # (B, N, P)
        
        # Update total positives count
        observed = mask > 0.5
        positives = (targets > 0.5) & observed
        self.val_pos_total += positives.sum().item()
        
        for p in range(self.num_pathogens):
            # Extract pathogen p
            probs_p = probs[:, :, p].flatten()      # (B*N,)
            targets_p = targets[:, :, p].flatten()  # (B*N,)
            mask_p = mask[:, :, p].flatten()        # (B*N,)
            
            # Filter to observed samples only
            observed_p = mask_p > 0.5
            if observed_p.sum() > 0:
                self.val_auroc[p].update(probs_p[observed_p], targets_p[observed_p].long())
                self.val_auprc[p].update(probs_p[observed_p], targets_p[observed_p].long())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log macro-averaged AUROC and AUPRC."""
        auroc_scores = []
        auprc_scores = []
        valid_pathogens = 0
        excluded_pathogens = 0
        
        for p in range(self.num_pathogens):
            auroc_p = float('nan')
            auprc_p = float('nan')
            
            try:
                # Compute might fail if 0 samples or degenerate (all 0 or all 1)
                # TorchMetrics usually raises ValueError or returns 0/1 depending on config
                # We enforce NaN for degeneracy
                auroc_val = self.val_auroc[p].compute()
                auprc_val = self.val_auprc[p].compute()
                
                # Check for validity (finite)
                if torch.isfinite(auroc_val) and torch.isfinite(auprc_val):
                    auroc_p = auroc_val
                    auprc_p = auprc_val
            except (ValueError, RuntimeError):
                # Degenerate case (e.g. no positive samples)
                pass
            
            # Only include if valid (not NaN)
            if not torch.isnan(torch.tensor(auroc_p)) and not torch.isnan(torch.tensor(auprc_p)):
                auroc_scores.append(auroc_p)
                auprc_scores.append(auprc_p)
                valid_pathogens += 1
            else:
                excluded_pathogens += 1
            
            # Log per-pathogen metrics (can be NaN)
            self.log(f'val_auroc_p{p}', auroc_p, on_epoch=True)
            self.log(f'val_auprc_p{p}', auprc_p, on_epoch=True)
            
            # Reset for next epoch
            self.val_auroc[p].reset()
            self.val_auprc[p].reset()
        
        # Macro average across pathogens
        if len(auroc_scores) > 0:
            macro_auroc = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in auroc_scores]).mean()
            self.log('val_auroc_macro', macro_auroc, prog_bar=True, on_epoch=True)
        
        if len(auprc_scores) > 0:
            macro_auprc = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in auprc_scores]).mean()
            self.log('val_auprc_macro', macro_auprc, prog_bar=True, on_epoch=True)
            
        # Log counts
        self.log('val_pos_total', float(self.val_pos_total), on_epoch=True)
        self.log('macro_valid_pathogens', float(valid_pathogens), on_epoch=True)
    
    def on_test_epoch_start(self):
        """Reset counters."""
        self.test_pos_total = 0

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
        
        # Update per-pathogen AUROC and AUPRC with masked samples
        probs = torch.sigmoid(logits)  # (B, N, P)
        targets = batch['y_next']      # (B, N, P)
        mask = batch['y_mask']         # (B, N, P)
        
        # Update total positives count
        observed = mask > 0.5
        positives = (targets > 0.5) & observed
        self.test_pos_total += positives.sum().item()
        
        for p in range(self.num_pathogens):
            # Extract pathogen p
            probs_p = probs[:, :, p].flatten()      # (B*N,)
            targets_p = targets[:, :, p].flatten()  # (B*N,)
            mask_p = mask[:, :, p].flatten()        # (B*N,)
            
            # Filter to observed samples only
            observed_p = mask_p > 0.5
            if observed_p.sum() > 0:
                self.test_auroc[p].update(probs_p[observed_p], targets_p[observed_p].long())
                self.test_auprc[p].update(probs_p[observed_p], targets_p[observed_p].long())
        
        return loss
    
    def on_test_epoch_end(self):
        """Compute and log macro-averaged AUROC and AUPRC for test."""
        auroc_scores = []
        auprc_scores = []
        valid_pathogens = 0
        excluded_pathogens = 0
        
        for p in range(self.num_pathogens):
            auroc_p = float('nan')
            auprc_p = float('nan')
            
            try:
                auroc_val = self.test_auroc[p].compute()
                auprc_val = self.test_auprc[p].compute()
                
                if torch.isfinite(auroc_val) and torch.isfinite(auprc_val):
                    auroc_p = auroc_val
                    auprc_p = auprc_val
            except (ValueError, RuntimeError):
                pass
            
            # Only include if valid (not NaN)
            if not torch.isnan(torch.tensor(auroc_p)) and not torch.isnan(torch.tensor(auprc_p)):
                auroc_scores.append(auroc_p)
                auprc_scores.append(auprc_p)
                valid_pathogens += 1
            else:
                excluded_pathogens += 1
            
            # Log per-pathogen metrics
            self.log(f'test_auroc_p{p}', auroc_p, on_epoch=True)
            self.log(f'test_auprc_p{p}', auprc_p, on_epoch=True)
            
            # Reset after test
            self.test_auroc[p].reset()
            self.test_auprc[p].reset()
        
        # Macro average across pathogens
        if len(auroc_scores) > 0:
            macro_auroc = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in auroc_scores]).mean()
            self.log('test_auroc_macro', macro_auroc, on_epoch=True)
        
        if len(auprc_scores) > 0:
            macro_auprc = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in auprc_scores]).mean()
            self.log('test_auprc_macro', macro_auprc, on_epoch=True)
            
        # Log counts
        self.log('test_pos_total', float(self.test_pos_total), on_epoch=True)
        self.log('macro_valid_pathogens', float(valid_pathogens), on_epoch=True)

    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
