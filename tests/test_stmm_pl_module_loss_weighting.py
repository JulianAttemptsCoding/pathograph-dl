import torch
import torch.nn as nn
import torch.nn.functional as F

from pathograph.pl.stmm_pl_module import STMMPLModule


class _DummyModel(nn.Module):
    """Dummy model only to provide at least one trainable parameter."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        raise RuntimeError("Dummy model forward should not be called in this unit test")


def _expected_equal_per_pathogen_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Blueprint-required loss:
      1) masked BCE per element
      2) per pathogen p: mean over observed (B,N)
      3) average these means equally across pathogens with any observations
    Shapes: (B,N,P)
    """
    per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_elem = per_elem * mask

    loss_sum_p = per_elem.sum(dim=(0, 1))   # (P,)
    mask_sum_p = mask.sum(dim=(0, 1))       # (P,)

    valid = mask_sum_p > 0
    mean_p = torch.zeros_like(loss_sum_p)
    mean_p[valid] = loss_sum_p[valid] / mask_sum_p[valid]

    return mean_p[valid].mean() if valid.any() else logits.sum() * 0.0


def test_masked_bce_is_equal_weighted_across_pathogens__should_fail_until_fixed():
    """
    EXPECTED TO FAIL under current implementation.

    Construction:
      - B=1, N=2, P=2
      - Pathogen 0 has 1 observed element with large loss
      - Pathogen 1 has 2 observed elements with tiny loss

    Current code computes global mean over all observed elements, which downweights p0.
    Blueprint requires equal mean across pathogens.
    """
    module = STMMPLModule(model=_DummyModel())

    logits = torch.tensor(
        [[[-10.0, 10.0],
          [  0.0, 10.0]]],
        dtype=torch.float32
    )
    targets = torch.tensor(
        [[[1.0, 1.0],
          [0.0, 1.0]]],
        dtype=torch.float32
    )
    mask = torch.tensor(
        [[[1.0, 1.0],
          [0.0, 1.0]]],
        dtype=torch.float32
    )

    expected = _expected_equal_per_pathogen_loss(logits, targets, mask)
    actual = module._masked_bce_with_logits(logits, targets, mask)

    global_mean = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * mask).sum() / mask.sum()
    assert not torch.isclose(global_mean, expected, atol=1e-6), "Construction error: global_mean == expected"

    # This is the requirement assertion; should fail until you fix the loss.
    assert torch.isclose(actual, expected, atol=1e-6), (
        f"Loss is not equal-weighted across pathogens.\n"
        f"actual(current module)={actual.item():.6f} vs expected(equal-per-pathogen)={expected.item():.6f}\n"
        f"global_mean(all-elements)={global_mean.item():.6f}"
    )


def test_empty_mask_guard_is_autograd_safe():
    """Test that loss computation with valid (non-empty) mask supports autograd."""
    module = STMMPLModule(model=_DummyModel())

    logits = torch.zeros((1, 2, 2), dtype=torch.float32, requires_grad=True)
    targets = torch.zeros((1, 2, 2), dtype=torch.float32)
    mask = torch.zeros((1, 2, 2), dtype=torch.float32)
    
    # Set at least one element to 1 to avoid the guard
    mask[0, 0, 0] = 1.0

    loss = module._masked_bce_with_logits(logits, targets, mask)
    assert loss.requires_grad is True
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
