import os
import sys
import unittest
import torch
import pytest
from pathlib import Path

# Add root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from pathograph.train.trade_lightning_module import TradeBaselinePL

class TestTradeStep6Smoke(unittest.TestCase):
    def test_baseline_optimizer_config(self):
        """Ensure TradeBaselinePL can configure an optimizer."""
        model = TradeBaselinePL(lr=0.01, weight_decay=1e-5)
        opt = model.configure_optimizers()
        
        # Should return an Optimizer instance
        self.assertIsInstance(opt, torch.optim.Optimizer)
        
        # Check params
        self.assertEqual(opt.defaults["lr"], 0.01)
        self.assertEqual(opt.defaults["weight_decay"], 1e-5)

    @pytest.mark.skipif(os.environ.get("RUN_TRADE_STEP6_SLOW") != "1", reason="Slow test")
    def test_entrypoint_fast_dev_run(self):
        """Run the entrypoint in fast_dev_run mode."""
        import subprocess
        
        cmd = [
            sys.executable,
            "-m", "tools.trade_step6_train_entrypoint",
            "--config", "config/trade_step6.yaml",
            "--override", "run.fast_dev_run=true",
            "--override", "logging.name=test_run"
        ]
        
        # run in root
        res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if res.returncode != 0:
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
            
        self.assertEqual(res.returncode, 0, "Entrypoint failed")

if __name__ == "__main__":
    unittest.main()
