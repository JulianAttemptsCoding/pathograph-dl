"""
Unit tests for filter_kwargs in tools/stmm_stepA_train.py.

Phase 4 policy:
  - filter_kwargs RAISES RuntimeError if any keys are dropped.
  - STMMGraphWaveNet must accept use_adaptive_adj, adaptive_emb_dim, adaptive_top_k,
    use_film — so no keys should ever be dropped for our three families.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.stmm_stepA_train import filter_kwargs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fn_simple(a, b, c=3):
    pass


def _fn_var_keyword(a, **kwargs):
    pass


class _ClassA:
    def __init__(self, x, y, z=10):
        pass


class _ClassVarKw:
    def __init__(self, x, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Tests — core filter_kwargs behaviour
# ---------------------------------------------------------------------------

class TestFilterKwargsFunction:
    def test_keeps_all_known_keys(self):
        result = filter_kwargs(_fn_simple, {"a": 1, "b": 2, "c": 99})
        assert result == {"a": 1, "b": 2, "c": 99}

    def test_raises_on_unknown_keys(self):
        """Phase 4 policy: unknown keys must raise, not silently drop."""
        with pytest.raises(RuntimeError, match="FATAL"):
            filter_kwargs(_fn_simple, {"a": 1, "b": 2, "c": 3, "d": 4})

    def test_empty_kwargs(self):
        result = filter_kwargs(_fn_simple, {})
        assert result == {}

    def test_all_unknown_raises(self):
        with pytest.raises(RuntimeError, match="FATAL"):
            filter_kwargs(_fn_simple, {"z": 1, "w": 2})

    def test_var_keyword_passes_all_through(self):
        kwargs = {"a": 1, "b": 2, "anything": 99}
        result = filter_kwargs(_fn_var_keyword, kwargs)
        assert result == kwargs

    def test_does_not_mutate_input(self):
        original = {"a": 1, "b": 2}
        filter_kwargs(_fn_simple, original)
        assert original == {"a": 1, "b": 2}


class TestFilterKwargsClass:
    def test_class_init_raises_on_unknown(self):
        with pytest.raises(RuntimeError, match="FATAL"):
            filter_kwargs(_ClassA, {"x": 10, "y": 20, "z": 5, "unknown": 999})

    def test_class_init_self_not_required(self):
        """'self' should never appear as a required kwarg key."""
        # self is special — if it's accidentally included it should be treated as
        # an unknown key and trigger the fatal error
        with pytest.raises(RuntimeError, match="FATAL"):
            filter_kwargs(_ClassA, {"self": "bad", "x": 1, "y": 2})

    def test_class_var_keyword_passes_all(self):
        kwargs = {"x": 1, "random_extra": 42, "another": "hello"}
        result = filter_kwargs(_ClassVarKw, kwargs)
        assert result == kwargs


# ---------------------------------------------------------------------------
# Integration tests against STMMGraphWaveNet
# ---------------------------------------------------------------------------

class TestFilterKwargsSTMMGraphWaveNet:
    """After T3 fix, STMMGraphWaveNet must accept all feature kwargs — zero drops."""

    def _base_kwargs(self):
        return {
            "residual_channels": 16,
            "dilation_channels": 16,
            "skip_channels": 32,
            "end_channels": 64,
            "kernel_size": 2,
            "dilations": [1, 2],
            "diffusion_K": 1,
            "dropout": 0.1,
            "num_pathogens": 8,
            "num_nodes": 194,
        }

    def test_adaptive_kwargs_not_dropped(self):
        """use_adaptive_adj, adaptive_emb_dim, adaptive_top_k must be accepted."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        kwargs = {
            **self._base_kwargs(),
            "use_adaptive_adj": True,
            "adaptive_emb_dim": 8,
            "adaptive_top_k": 10,
        }
        # Must NOT raise
        result = filter_kwargs(STMMGraphWaveNet, kwargs)
        assert result["use_adaptive_adj"] is True
        assert result["adaptive_emb_dim"] == 8
        assert result["adaptive_top_k"] == 10

    def test_film_kwargs_not_dropped(self):
        """use_film must be accepted."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        kwargs = {
            **self._base_kwargs(),
            "use_adaptive_adj": False,
            "use_film": True,
        }
        result = filter_kwargs(STMMGraphWaveNet, kwargs)
        assert result["use_film"] is True
        assert result["use_adaptive_adj"] is False

    def test_blueprint_kwargs_not_dropped(self):
        """Blueprint config: use_adaptive_adj=False, no film — must still pass."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        kwargs = {
            **self._base_kwargs(),
            "use_adaptive_adj": False,
        }
        result = filter_kwargs(STMMGraphWaveNet, kwargs)
        assert result["use_adaptive_adj"] is False

    def test_adaptive_full_trial_kwargs_construct_model(self):
        """Full adaptive trial_00 kwarg set must construct a model (CPU, no GPU)."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        kwargs = {
            "residual_channels": 64,
            "dilation_channels": 32,
            "skip_channels": 64,
            "end_channels": 128,
            "kernel_size": 2,
            "dilations": [1, 2, 4, 8, 16],
            "diffusion_K": 2,
            "dropout": 0.2094,
            "num_pathogens": 8,
            "num_nodes": 194,
            "use_adaptive_adj": True,
            "adaptive_emb_dim": 8,
            "adaptive_top_k": 20,
        }
        filtered = filter_kwargs(STMMGraphWaveNet, kwargs)
        model = STMMGraphWaveNet(**filtered)
        assert model.use_adaptive_adj is True

    def test_film_full_trial_kwargs_construct_model(self):
        """Full film trial kwarg set must construct a model."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        kwargs = {
            "residual_channels": 32,
            "dilation_channels": 32,
            "skip_channels": 64,
            "end_channels": 128,
            "kernel_size": 2,
            "dilations": [1, 2, 4, 8, 16],
            "diffusion_K": 2,
            "dropout": 0.1,
            "num_pathogens": 8,
            "num_nodes": 194,
            "use_adaptive_adj": False,
            "use_film": True,
        }
        filtered = filter_kwargs(STMMGraphWaveNet, kwargs)
        model = STMMGraphWaveNet(**filtered)
        assert model.use_film is True

    def test_blueprint_full_kwargs_construct_model(self):
        """Blueprint config must construct a plain model."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        kwargs = {
            "residual_channels": 32,
            "dilation_channels": 32,
            "skip_channels": 64,
            "end_channels": 128,
            "kernel_size": 2,
            "dilations": [1, 2, 4, 8, 16],
            "diffusion_K": 2,
            "dropout": 0.1,
            "num_pathogens": 8,
            "num_nodes": 194,
            "use_adaptive_adj": False,
        }
        filtered = filter_kwargs(STMMGraphWaveNet, kwargs)
        model = STMMGraphWaveNet(**filtered)
        assert model.use_adaptive_adj is False
        assert model.use_film is False



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fn_simple(a, b, c=3):
    pass


def _fn_varargs(*args, **kwargs):
    pass


def _fn_var_keyword(a, **kwargs):
    pass


class _ClassA:
    def __init__(self, x, y, z=10):
        pass


class _ClassVarKw:
    def __init__(self, x, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFilterKwargsFunction:
    def test_keeps_all_known_keys(self):
        result = filter_kwargs(_fn_simple, {"a": 1, "b": 2, "c": 99})
        assert result == {"a": 1, "b": 2, "c": 99}

    def test_drops_unknown_keys(self):
        result = filter_kwargs(_fn_simple, {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        assert result == {"a": 1, "b": 2, "c": 3}
        assert "d" not in result
        assert "e" not in result

    def test_empty_kwargs(self):
        result = filter_kwargs(_fn_simple, {})
        assert result == {}

    def test_all_unknown_keys_drops_all(self):
        result = filter_kwargs(_fn_simple, {"z": 1, "w": 2})
        assert result == {}

    def test_var_keyword_passes_all_through(self):
        """If callable accepts **kwargs, nothing should be dropped."""
        kwargs = {"a": 1, "b": 2, "anything": 99}
        result = filter_kwargs(_fn_var_keyword, kwargs)
        assert result == kwargs

    def test_does_not_mutate_input(self):
        original = {"a": 1, "b": 2, "unknown_key": 99}
        filter_kwargs(_fn_simple, original)
        assert "unknown_key" in original  # original unchanged


class TestFilterKwargsClass:
    def test_class_init_drops_unknown(self):
        result = filter_kwargs(_ClassA, {"x": 10, "y": 20, "z": 5, "unknown": 999})
        assert result == {"x": 10, "y": 20, "z": 5}
        assert "unknown" not in result

    def test_class_init_self_not_required(self):
        """'self' should never appear in the returned dict as a required kwarg."""
        result = filter_kwargs(_ClassA, {"self": "should_be_dropped", "x": 1, "y": 2})
        assert "self" not in result
        assert result == {"x": 1, "y": 2}

    def test_class_var_keyword_passes_all(self):
        kwargs = {"x": 1, "random_extra": 42, "another": "hello"}
        result = filter_kwargs(_ClassVarKw, kwargs)
        assert result == kwargs


class TestFilterKwargsSTMMGraphWaveNet:
    """Integration test: verify filter against the real STMMGraphWaveNet signature."""

    def test_adaptive_keys_are_dropped(self):
        """adaptive_emb_dim, adaptive_top_k, use_adaptive_adj should all be dropped."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        model_kwargs_from_adaptive_trial = {
            "residual_channels": 64,
            "dilation_channels": 32,
            "skip_channels": 64,
            "end_channels": 128,
            "kernel_size": 2,
            "dilations": [1, 2, 4, 8, 16],
            "diffusion_K": 2,
            "dropout": 0.2094,
            "num_pathogens": 8,
            "num_nodes": 194,
            "use_adaptive_adj": True,        # not in STMMGraphWaveNet
            "adaptive_emb_dim": 8,           # not in STMMGraphWaveNet
            "adaptive_top_k": 20,            # not in STMMGraphWaveNet
        }

        result = filter_kwargs(STMMGraphWaveNet, model_kwargs_from_adaptive_trial)

        # Supported keys must be present
        assert result["residual_channels"] == 64
        assert result["dropout"] == pytest.approx(0.2094)
        assert result["diffusion_K"] == 2

        # Unsupported keys must be absent
        assert "use_adaptive_adj" not in result
        assert "adaptive_emb_dim" not in result
        assert "adaptive_top_k" not in result

    def test_film_keys_are_dropped(self):
        """use_film should be dropped by filter_kwargs."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        model_kwargs_from_film_trial = {
            "residual_channels": 32,
            "dilation_channels": 32,
            "skip_channels": 64,
            "end_channels": 128,
            "kernel_size": 2,
            "dilations": [1, 2, 4, 8, 16],
            "diffusion_K": 2,
            "dropout": 0.1,
            "num_pathogens": 8,
            "num_nodes": 194,
            "use_adaptive_adj": False,   # not in STMMGraphWaveNet
            "use_film": True,            # not in STMMGraphWaveNet
        }

        result = filter_kwargs(STMMGraphWaveNet, model_kwargs_from_film_trial)

        assert "use_adaptive_adj" not in result
        assert "use_film" not in result
        assert result["residual_channels"] == 32

    def test_filtered_kwargs_construct_model(self):
        """Verify the model can actually be instantiated with filtered kwargs (CPU only)."""
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet

        noisy_kwargs = {
            "residual_channels": 16,
            "dilation_channels": 16,
            "skip_channels": 32,
            "end_channels": 64,
            "kernel_size": 2,
            "dilations": [1, 2],
            "diffusion_K": 1,
            "dropout": 0.1,
            "num_pathogens": 8,
            "num_nodes": 194,
            "use_adaptive_adj": True,
            "adaptive_emb_dim": 8,
            "adaptive_top_k": 10,
            "use_film": False,
        }

        filtered = filter_kwargs(STMMGraphWaveNet, noisy_kwargs)
        # Must not raise TypeError
        model = STMMGraphWaveNet(**filtered)
        assert model is not None

