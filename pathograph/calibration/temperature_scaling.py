"""
Temperature scaling for model calibration.

Fits a single scalar temperature parameter on validation set to calibrate probabilities.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class TemperatureScaling(nn.Module):
    """
    Temperature scaling calibration module.
    
    Fits a single temperature parameter T on validation logits to minimize NLL.
    After fitting, use model(logits) to get calibrated probabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: raw model logits, any shape
        
        Returns:
            calibrated probabilities (same shape as logits)
        """
        return torch.sigmoid(logits / self.temperature)
    
    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        max_iter: int = 50,
    ):
        """
        Fit temperature on validation set using masked BCE.
        
        Args:
            logits: (N_samples, P) or (B, N, P) validation logits
            targets: same shape, binary targets
            mask: same shape, observation mask
            max_iter: maximum LBFGS iterations
        """
        # Flatten to (N_total,)
        logits_flat = logits.flatten()
        targets_flat = targets.flatten()
        mask_flat = mask.flatten()
        
        # Filter to observed only
        observed = mask_flat > 0.5
        logits_obs = logits_flat[observed]
        targets_obs = targets_flat[observed]
        
        if logits_obs.numel() == 0:
            raise ValueError("No observed samples for temperature scaling fit")
        
        # Optimize temperature with LBFGS
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            # Compute loss
            scaled_logits = logits_obs / self.temperature
            loss = F.binary_cross_entropy_with_logits(
                scaled_logits, targets_obs, reduction='mean'
            )
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        # Clamp temperature to reasonable range
        with torch.no_grad():
            self.temperature.clamp_(0.1, 10.0)
