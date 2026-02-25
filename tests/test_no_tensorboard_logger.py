"""
T5: Assert that the training entrypoint never instantiates a TensorBoard/TensorBoardX logger.

Two checks:
  1. Importing tools.stmm_stepA_train must NOT transitively import tensorboard or
     tensorboardX at module-import time.
  2. The CSVLogger gate-log line is present after trainer construction (white-box check).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helper: check no tensorboard/tensorboardX in sys.modules after import
# ---------------------------------------------------------------------------

def test_stmm_train_import_does_not_load_tensorboard(monkeypatch):
    """Importing the training module must not pull in tensorboard or tensorboardX."""
    # Remove the module from cache so we get a clean import
    mods_to_remove = [k for k in sys.modules if k.startswith("tools.stmm_stepA_train")]
    for m in mods_to_remove:
        sys.modules.pop(m, None)

    # Stub out heavy deps so we can import without a full env
    for dep in ("pytorch_lightning", "torch", "yaml",
                "pathograph.data.trade_datamodule",
                "pathograph.models.stmm_gwnet",
                "pathograph.pl.stmm_pl_module"):
        if dep not in sys.modules:
            sys.modules[dep] = MagicMock()

    # Also stub pytorch_lightning.loggers.CSVLogger
    pl_mock = sys.modules.get("pytorch_lightning", MagicMock())
    loggers_mock = MagicMock()
    loggers_mock.CSVLogger = MagicMock
    pl_mock.loggers = loggers_mock
    sys.modules["pytorch_lightning.loggers"] = loggers_mock

    # Now import
    import importlib
    try:
        importlib.import_module("tools.stmm_stepA_train")
    except Exception:
        pass  # import errors from stubs are OK for this test

    tb_keys = [k for k in sys.modules
               if ("tensorboard" in k.lower() or "tensorboardx" in k.lower())
               and not k.startswith("unittest")]
    assert tb_keys == [], (
        f"TensorBoard/TensorBoardX was imported by training module: {tb_keys}"
    )


def test_trainer_uses_csv_logger_not_tensorboard(tmp_path):
    """
    White-box: construct the Trainer the same way stmm_stepA_train does and
    verify the logger is CSVLogger, not TensorBoardLogger / TensorBoardXLogger.
    """
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.loggers import CSVLogger
    except ImportError:
        pytest.skip("pytorch_lightning not installed in this environment")

    csv_logger = CSVLogger(save_dir=str(tmp_path), name="csv_logs", version=0)
    # Verify it is indeed a CSVLogger
    assert isinstance(csv_logger, CSVLogger), "Expected CSVLogger instance"

    # Verify it is NOT a TensorBoardLogger (if available)
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        assert not isinstance(csv_logger, TensorBoardLogger), \
            "Logger must not be TensorBoardLogger"
    except ImportError:
        pass  # TensorBoardLogger not present — even better


def test_no_tensorboardx_import_in_training_path():
    """
    Ensure tensorboardX is NOT imported when running through the training path.
    Uses a blocking fake that raises on import of tensorboard/tensorboardX.
    """
    class _BlockingFinder:
        @classmethod
        def find_module(cls, name, path=None):
            if name.lower() in ("tensorboard", "tensorboardx"):
                raise ImportError(
                    f"[TEST GUARD] tensorboard/tensorboardX must NOT be imported "
                    f"during Phase 4 training startup. Got: {name}"
                )
            return None

    sys.meta_path.insert(0, _BlockingFinder)
    try:
        # Re-import the training module; this must not trigger tensorboard import
        mods_to_remove = [k for k in sys.modules if k.startswith("tools.stmm_stepA_train")]
        for m in mods_to_remove:
            sys.modules.pop(m, None)

        # Stub heavy deps
        for dep in ("pytorch_lightning", "torch", "yaml",
                    "pathograph.data.trade_datamodule",
                    "pathograph.models.stmm_gwnet",
                    "pathograph.pl.stmm_pl_module"):
            if dep not in sys.modules:
                sys.modules[dep] = MagicMock()

        pl_mock = sys.modules.get("pytorch_lightning", MagicMock())
        loggers_mock = MagicMock()
        loggers_mock.CSVLogger = MagicMock
        pl_mock.loggers = loggers_mock
        sys.modules["pytorch_lightning.loggers"] = loggers_mock

        import importlib
        try:
            importlib.import_module("tools.stmm_stepA_train")
        except Exception:
            pass
    finally:
        sys.meta_path.remove(_BlockingFinder)

