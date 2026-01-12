import torch
import torch.nn as nn
from typing import Dict, Union

class PersistenceBaseline(nn.Module):
    """Simple baseline that predicts y_{t+H} = x_t.
    
    It simply takes the last timestep of the input window.
    If the last timestep x_t is masked (0), then the prediction is 0.
    
    Includes a dummy scalar parameter 'scale' (init=1.0) to ensure
    autograd graph is constructed for integration testing.
    """
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, batch: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Handle dict input (standard batch from DataLoader)
        if isinstance(batch, dict):
            preds = {}
            
            if "base_trade" in batch:
                # Take last time step: index -1 along dim 1
                x_last = batch["base_trade"][:, -1, :, :, :] # (B, N, N, 2)
                preds["y_base_pred"] = x_last * self.scale
                
            if "risk_trade" in batch:
                r_last = batch["risk_trade"][:, -1, :, :, :, :] # (B, N, N, K, 2)
                preds["y_risk_pred"] = r_last * self.scale
                
            return preds
        
        # Handle tensor input (debug scripts may pass raw tensor)
        elif isinstance(batch, torch.Tensor):
            x = batch
            if x.ndim == 5:  # (B, L, N, N, C) - take last timestep
                x_last = x[:, -1]
            elif x.ndim == 4:  # (B, N, N, C) - already single timestep
                x_last = x
            else:
                raise ValueError(f"Unexpected tensor shape: {x.shape}. Expected 4D or 5D tensor.")
            return {"y_base_pred": x_last * self.scale}
        
        else:
            raise TypeError(f"Expected dict or Tensor, got {type(batch)}")

