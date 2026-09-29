import torch

from utils.losses import discrete_time_nll, focal_bce


def test_survival_nll_finite_and_lower_when_correct():
    B, K = 8, 6
    event_bin = torch.randint(0, K, (B,))
    event_ind = torch.randint(0, 2, (B,)).float()

    # "confident-correct" logits: high hazard exactly at the event bin
    good = torch.full((B, K), -4.0)
    for i in range(B):
        if event_ind[i] == 1:
            good[i, event_bin[i]] = 4.0
    bad = -good

    l_good = discrete_time_nll(good, event_bin, event_ind)
    l_bad = discrete_time_nll(bad, event_bin, event_ind)
    assert torch.isfinite(l_good) and torch.isfinite(l_bad)
    assert l_good < l_bad


def test_focal_bce_runs():
    logits = torch.randn(16)
    targets = (torch.rand(16) > 0.5).float()
    loss = focal_bce(logits, targets)
    assert torch.isfinite(loss) and loss >= 0
