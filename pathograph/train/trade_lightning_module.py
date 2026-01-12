import torch
from typing import Dict, Any
from pathograph.models.trade_baseline import PersistenceBaseline
from pathograph.train.trade_losses import masked_mse

try:
    import pytorch_lightning as pl
    HAS_PL = True
    BaseClass = pl.LightningModule
except ImportError:
    HAS_PL = False
    BaseClass = torch.nn.Module # Fallback to avoid NameError

class TradeBaselinePL(BaseClass):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        if not HAS_PL:
            print("Warning: pytorch_lightning not found. TradeBaselinePL acts as simple nn.Module.")
        else:
            try:
                self.save_hyperparameters(ignore=["kwargs"])
            except Exception:
                self.save_hyperparameters()
        self.model = PersistenceBaseline()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # 1. Forward pass
        preds = self.model(batch)
        
        # 2. Compute loss
        total_loss = torch.tensor(0.0, device=self.device) # self.device works in PL; in nn.Module it might not exist if not added
        if not HAS_PL and not hasattr(self, "device"):
             # basic fallback
             pass 
             
        valid_losses = 0
        
        # Base loss
        if "y_base" in batch and "y_base_pred" in preds:
            target = batch["y_base"]
            mask = batch.get("y_base_mask", torch.ones_like(target))
            pred = preds["y_base_pred"]
            
            l = masked_mse(pred, target, mask)
            if HAS_PL: self.log("train_loss_base", l, prog_bar=True)
            total_loss += l
            valid_losses += 1
            
        # Risk loss
        if "y_risk" in batch and "y_risk_pred" in preds:
            target = batch["y_risk"]
            mask = batch.get("y_risk_mask", torch.ones_like(target))
            pred = preds["y_risk_pred"]
            
            l = masked_mse(pred, target, mask)
            if HAS_PL: self.log("train_loss_risk", l, prog_bar=True)
            total_loss += l
            valid_losses += 1
            
        if valid_losses == 0:
            # Fallback if no targets found (should not happen in correct config)
            # Need a tensor attached to graph
            return torch.tensor(0.0, requires_grad=True)
            
        if HAS_PL: self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        lr = float(getattr(self.hparams, "lr", self.lr)) if HAS_PL else self.lr
        wd = float(getattr(self.hparams, "weight_decay", self.weight_decay)) if HAS_PL else self.weight_decay
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # 1) Forward pass
        preds = self.model(batch)

        # 2) Compute loss (same structure as training_step but with val_* logs)
        total_loss = torch.tensor(0.0, device=self.device)
        valid_losses = 0

        # Base loss
        if "y_base" in batch and "y_base_pred" in preds:
            target = batch["y_base"]
            mask = batch.get("y_base_mask", torch.ones_like(target))
            pred = preds["y_base_pred"]
            l = masked_mse(pred, target, mask)
            if HAS_PL:
                self.log("val_loss_base", l, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            total_loss = total_loss + l
            valid_losses += 1

        # Risk loss
        if "y_risk" in batch and "y_risk_pred" in preds:
            target = batch["y_risk"]
            mask = batch.get("y_risk_mask", torch.ones_like(target))
            pred = preds["y_risk_pred"]
            l = masked_mse(pred, target, mask)
            if HAS_PL:
                self.log("val_loss_risk", l, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            total_loss = total_loss + l
            valid_losses += 1

        if valid_losses == 0:
            # Avoid checkpoint monitor crash: always emit val_loss
            z = torch.tensor(0.0, device=self.device)
            if HAS_PL:
                self.log("val_loss", z, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return z

        if HAS_PL:
            self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Reuse validation_step logic but log with test_* prefix
        preds = self.model(batch)

        total_loss = torch.tensor(0.0, device=self.device)
        valid_losses = 0

        if "y_base" in batch and "y_base_pred" in preds:
            target = batch["y_base"]
            mask = batch.get("y_base_mask", torch.ones_like(target))
            pred = preds["y_base_pred"]
            l = masked_mse(pred, target, mask)
            if HAS_PL:
                self.log("test_loss_base", l, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            total_loss = total_loss + l
            valid_losses += 1

        if "y_risk" in batch and "y_risk_pred" in preds:
            target = batch["y_risk"]
            mask = batch.get("y_risk_mask", torch.ones_like(target))
            pred = preds["y_risk_pred"]
            l = masked_mse(pred, target, mask)
            if HAS_PL:
                self.log("test_loss_risk", l, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            total_loss = total_loss + l
            valid_losses += 1

        if valid_losses == 0:
            z = torch.tensor(0.0, device=self.device)
            if HAS_PL:
                self.log("test_loss", z, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return z

        if HAS_PL:
            self.log("test_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss
