import math

import torch

from vesp.core.diagnostics import source_diagnostics


def _two_shell_setup():
    # 3 sources on a deep shell (r=0.5), 3 on a shallow shell (r=0.9)
    positions = torch.zeros((6, 3), dtype=torch.float64)
    positions[:3, 0] = 0.5
    positions[3:, 0] = 0.9
    weights = torch.ones(6, dtype=torch.float64)
    shell_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    return positions, weights, shell_ids


def test_collapsed_shell_energy_flags_collapse():
    positions, weights, shell_ids = _two_shell_setup()
    sigma = torch.tensor([10.0, 10.0, 10.0, 1.0e-6, 1.0e-6, 1.0e-6], dtype=torch.float64)
    diag = source_diagnostics(
        source_positions=positions, source_weights=weights, shell_ids=shell_ids, sigma=sigma,
        shell_collapse_threshold=0.90,
    )
    assert diag["shell_collapse_flag"] is True
    assert diag["dominant_shell_energy_fraction"] > 0.99
    assert diag["dominant_shell_id"] == 0
    # entropy near 0 for a fully collapsed distribution
    assert diag["shell_energy_entropy"] < 0.05
    assert diag["shell_energy_effective_count"] < 1.1


def test_uniform_shell_energy_no_collapse():
    positions, weights, shell_ids = _two_shell_setup()
    sigma = torch.ones(6, dtype=torch.float64)  # equal energy per shell
    diag = source_diagnostics(
        source_positions=positions, source_weights=weights, shell_ids=shell_ids, sigma=sigma,
        shell_collapse_threshold=0.90,
    )
    assert diag["shell_collapse_flag"] is False
    assert abs(diag["dominant_shell_energy_fraction"] - 0.5) < 1.0e-9
    # entropy near log(2) for a balanced 2-shell distribution
    assert abs(diag["shell_energy_entropy"] - math.log(2.0)) < 1.0e-6
    assert abs(diag["shell_energy_effective_count"] - 2.0) < 1.0e-3


def test_single_shell_never_collapses():
    positions = torch.zeros((4, 3), dtype=torch.float64)
    positions[:, 0] = 0.86
    weights = torch.ones(4, dtype=torch.float64)
    shell_ids = torch.zeros(4, dtype=torch.long)
    sigma = torch.tensor([5.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    diag = source_diagnostics(
        source_positions=positions, source_weights=weights, shell_ids=shell_ids, sigma=sigma,
    )
    # one shell trivially has fraction 1.0 but must NOT be reported as collapsed
    assert diag["dominant_shell_energy_fraction"] == 1.0
    assert diag["shell_collapse_flag"] is False


def test_sigma_norm_warning_threshold():
    positions, weights, shell_ids = _two_shell_setup()
    big = torch.full((6,), 10.0, dtype=torch.float64)
    small = torch.full((6,), 0.01, dtype=torch.float64)
    diag_big = source_diagnostics(
        source_positions=positions, source_weights=weights, shell_ids=shell_ids, sigma=big,
        sigma_l2_warning_threshold=1.0,
    )
    diag_small = source_diagnostics(
        source_positions=positions, source_weights=weights, shell_ids=shell_ids, sigma=small,
        sigma_l2_warning_threshold=1.0,
    )
    assert diag_big["sigma_norm_warning"] is True
    assert diag_small["sigma_norm_warning"] is False
