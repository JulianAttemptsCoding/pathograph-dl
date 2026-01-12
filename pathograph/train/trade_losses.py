import torch

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes Mean Squared Error only over masked elements.
    
    Args:
        pred: Prediction tensor (any shape)
        target: Target tensor (same shape)
        mask: Mask tensor (same shape or broadcastable), 1=observed, 0=missing
        eps: Epsilon to avoid division by zero
        
    Returns:
        Scalar tensor with the MSE loss.
    """
    # Ensure mask is float matching pred
    mask_f = mask.to(pred.dtype)
    
    sq_err = (pred - target) ** 2
    masked_err = sq_err * mask_f
    
    # We sum over all dimensions
    total_loss = masked_err.sum()
    total_count = mask_f.sum()
    
    return total_loss / (total_count + eps)
