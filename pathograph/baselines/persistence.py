"""
Persistence baseline for pathogen status prediction.

Predicts next month's status = current month's status (last observed value).
Evaluated under identical splits as ST-MM-GNN model.
"""

from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


class PersistenceBaseline(pl.LightningModule):
    """
    Persistence baseline: y_pred(t+1) = y_obs(t).
    
    For each (node, pathogen), predicts the last observed value in the input window.
    This is a strong baseline for rare event prediction with high autocorrelation.
    """
    
    def __init__(self, num_pathogens: int = 8):
        super().__init__()
        self.num_pathogens = num_pathogens
        
        # Metrics for validation and test
        self.val_auroc = nn.ModuleList([
            BinaryAUROC() for _ in range(num_pathogens)
        ])
        self.val_auprc = nn.ModuleList([
            BinaryAveragePrecision() for _ in range(num_pathogens)
        ])
        
        self.test_auroc = nn.ModuleList([
            BinaryAUROC() for _ in range(num_pathogens)
        ])
        self.test_auprc = nn.ModuleList([
            BinaryAveragePrecision() for _ in range(num_pathogens)
        ])
    
    def _predict_persistence(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate persistence predictions for pathogen status.
        
        Supports flexible batch formats:
        - With history: 'y_hist', 'y_history', 'y_window', 'y' (with time dim)
        - With masks: 'y_hist_mask', 'y_history_mask'
        - Fallback to target shape: 'y_next', 'targets'
        
        Args:
            batch: Dictionary with history and/or target tensors
        
        Returns:
            predictions: (N, P) or (B, N, P) binary predictions
        """
        # Try to find history tensor
        hist_keys = ['y_hist', 'y_history', 'y_window', 'y']
        y_hist = None
        for key in hist_keys:
            if key in batch:
                y_hist = batch[key]
                break
        
        # Try to find history mask
        mask_keys = ['y_hist_mask', 'y_history_mask']
        y_hist_mask = None
        for key in mask_keys:
            if key in batch:
                y_hist_mask = batch[key]
                break
        
        if y_hist is not None:
            # We have history - compute true persistence
            # Shape can be (B, L, N, P) or (L, N, P)
            has_batch = y_hist.ndim == 4
            
            if has_batch:
                B, L, N, P = y_hist.shape
                preds = torch.zeros(B, N, P, device=y_hist.device, dtype=y_hist.dtype)
                
                for b in range(B):
                    for n in range(N):
                        for p in range(P):
                            # Find last observed value
                            if y_hist_mask is not None:
                                mask_seq = y_hist_mask[b, :, n, p]
                                observed_times = torch.where(mask_seq > 0.5)[0]
                            else:
                                # No mask provided, assume all history is observed
                                observed_times = torch.arange(L, device=y_hist.device)
                            
                            if len(observed_times) > 0:
                                last_t = observed_times[-1]
                                preds[b, n, p] = y_hist[b, last_t, n, p]
                            # else: stays 0 (no observations)
            else:
                # Shape (L, N, P)
                L, N, P = y_hist.shape
                preds = torch.zeros(N, P, device=y_hist.device, dtype=y_hist.dtype)
                
                for n in range(N):
                    for p in range(P):
                        if y_hist_mask is not None:
                            mask_seq = y_hist_mask[:, n, p]
                            observed_times = torch.where(mask_seq > 0.5)[0]
                        else:
                            observed_times = torch.arange(L, device=y_hist.device)
                        
                        if len(observed_times) > 0:
                            last_t = observed_times[-1]
                            preds[n, p] = y_hist[last_t, n, p]
            
            return preds
        
        # No history available - infer shape from targets and predict all zeros
        target_keys = ['y_next', 'targets']
        y_next = None
        for key in target_keys:
            if key in batch:
                y_next = batch[key]
                break
        
        if y_next is None:
            available_keys = list(batch.keys())
            raise KeyError(
                f"Persistence baseline requires either history keys {hist_keys} "
                f"or target keys {target_keys}. Found: {available_keys}"
            )
        
        # Predict all zeros (conservative baseline)
        preds = torch.zeros_like(y_next)
        return preds
    
    def _masked_bce(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked BCE loss (same contract as STMM model).
        
        Args:
            predictions: (N, P) probabilities [0, 1]
            targets: (N, P) binary targets
            mask: (N, P) observation mask
        
        Returns:
            scalar loss
        """
        # Clip predictions to avoid log(0)
        preds_clipped = torch.clamp(predictions, 1e-7, 1 - 1e-7)
        
        # BCE: -[y*log(p) + (1-y)*log(1-p)]
        loss = -(targets * torch.log(preds_clipped) + 
                 (1 - targets) * torch.log(1 - preds_clipped))
        
        mask_f = mask.to(dtype=loss.dtype)
        loss = loss * mask_f
        
        # Per-pathogen mean
        loss_sum_p = loss.sum(dim=0)  # (P,)
        mask_sum_p = mask_f.sum(dim=0)  # (P,)
        
        valid = mask_sum_p > 0
        if not torch.any(valid):
            raise RuntimeError("Persistence baseline: all pathogens have zero observed labels")
        
        mean_p = torch.zeros_like(loss_sum_p)
        mean_p[valid] = loss_sum_p[valid] / mask_sum_p[valid]
        
        return mean_p[valid].mean()
    
    def _masked_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked binary accuracy for (N, P) tensors."""
        mask_bool = mask > 0.5
        if mask_bool.sum() == 0:
            return predictions.sum() * 0.0
        
        correct = (predictions == targets)[mask_bool].float()
        return correct.mean()
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step using TradeDataset batch format."""
        preds = self._predict_persistence(batch)
        targets = batch['y_next']  # (N, P)
        mask = batch['y_mask']  # (N, P)
        
        # Compute loss
        loss = self._masked_bce(preds, targets, mask)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Compute accuracy
        acc = self._masked_accuracy(preds, targets, mask)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        # Update AUROC/AUPRC (use predictions as "probabilities")
        for p in range(self.num_pathogens):
            probs_p = preds[:, p]  # (N,)
            targets_p = targets[:, p]  # (N,)
            mask_p = mask[:, p]  # (N,)
            
            observed = mask_p > 0.5
            if observed.sum() > 0:
                self.val_auroc[p].update(probs_p[observed], targets_p[observed].long())
                self.val_auprc[p].update(probs_p[observed], targets_p[observed].long())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log macro-averaged metrics."""
        auroc_scores = []
        auprc_scores = []
        
        for p in range(self.num_pathogens):
            try:
                auroc_p = self.val_auroc[p].compute()
                auprc_p = self.val_auprc[p].compute()
                
                if not torch.isnan(auroc_p):
                    auroc_scores.append(auroc_p)
                if not torch.isnan(auprc_p):
                    auprc_scores.append(auprc_p)
                
                self.log(f'val_auroc_p{p}', auroc_p, on_epoch=True)
                self.log(f'val_auprc_p{p}', auprc_p, on_epoch=True)
                
            except Exception:
                pass
            
            self.val_auroc[p].reset()
            self.val_auprc[p].reset()
        
        if len(auroc_scores) > 0:
            macro_auroc = torch.stack(auroc_scores).mean()
            self.log('val_auroc_macro', macro_auroc, prog_bar=True, on_epoch=True)
        
        if len(auprc_scores) > 0:
            macro_auprc = torch.stack(auprc_scores).mean()
            self.log('val_auprc_macro', macro_auprc, prog_bar=True, on_epoch=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step using TradeDataset batch format."""
        preds = self._predict_persistence(batch)
        targets = batch['y_next']  # (N, P)
        mask = batch['y_mask']  # (N, P)
        
        # Compute loss
        loss = self._masked_bce(preds, targets, mask)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Compute accuracy
        acc = self._masked_accuracy(preds, targets, mask)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        # Update AUROC/AUPRC
        for p in range(self.num_pathogens):
            probs_p = preds[:, p]  # (N,)
            targets_p = targets[:, p]  # (N,)
            mask_p = mask[:, p]  # (N,)
            
            observed = mask_p > 0.5
            if observed.sum() > 0:
                self.test_auroc[p].update(probs_p[observed], targets_p[observed].long())
                self.test_auprc[p].update(probs_p[observed], targets_p[observed].long())
        
        return loss
    
    def on_test_epoch_end(self):
        """Compute and log macro-averaged test metrics."""
        auroc_scores = []
        auprc_scores = []
        
        for p in range(self.num_pathogens):
            try:
                auroc_p = self.test_auroc[p].compute()
                auprc_p = self.test_auprc[p].compute()
                
                if not torch.isnan(auroc_p):
                    auroc_scores.append(auroc_p)
                if not torch.isnan(auprc_p):
                    auprc_scores.append(auprc_p)
                
                self.log(f'test_auroc_p{p}', auroc_p, on_epoch=True)
                self.log(f'test_auprc_p{p}', auprc_p, on_epoch=True)
                
            except Exception:
                pass
            
            self.test_auroc[p].reset()
            self.test_auprc[p].reset()
        
        if len(auroc_scores) > 0:
            macro_auroc = torch.stack(auroc_scores).mean()
            self.log('test_auroc_macro', macro_auroc, on_epoch=True)
        
        if len(auprc_scores) > 0:
            macro_auprc = torch.stack(auprc_scores).mean()
            self.log('test_auprc_macro', macro_auprc, on_epoch=True)
    
    def configure_optimizers(self):
        """No optimizer needed for persistence baseline."""
        return None
