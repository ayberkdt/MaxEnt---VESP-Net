"""Tests for the experiment-first framework (vesp.experiments).

These are CI-friendly: they exercise the runner / summarizer on the small synthetic
experiment configs only. No real-lunar data is loaded or downloaded.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vesp.common.config import merge_defaults
from vesp.core.models import MultiShellDiscreteVESP
from vesp.experiments.registry import CORE_EXPERIMENTS
from vesp.experiments.runner import (
    Trial,
    expand_trials,
    git_commit_hash,
    load_experiment_config,
    run_experiment,
)
from vesp.experiments.suites import SUITES, resolve_suite
from vesp.experiments.summarize import SUMMARY_COLUMNS, summary_row, write_suite_artifacts
from vesp.training.train_discrete import run

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "configs" / "experiments"
ALL_EXPERIMENT_CONFIGS = sorted(EXP_DIR.glob("*.yaml"))
SYNTHETIC_CONFIGS = [p for p in ALL_EXPERIMENT_CONFIGS if p.name.startswith("synthetic_")]


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize("path", ALL_EXPERIMENT_CONFIGS, ids=lambda p: p.name)
def test_experiment_config_loads_and_expands(path):
    cfg = load_experiment_config(path)
    assert cfg["experiment"]["name"]
    trials = expand_trials(cfg)
    assert len(trials) >= 1
    for trial in trials:
        assert isinstance(trial, Trial)
        assert "model" in trial.config  # merge_defaults applied
        assert "loss" in trial.config


def test_load_experiment_config_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("just_a_string", encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(bad)
    no_base = tmp_path / "nobase.yaml"
    no_base.write_text("experiment:\n  name: x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(no_base)


def test_coupled_grid_is_cartesian_product():
    cfg = load_experiment_config(EXP_DIR / "synthetic_entropy_pareto.yaml")
    trials = expand_trials(cfg)
    # 8 entropy weights x 4 entropy modes
    assert len(trials) == 32
    assert len({t.name for t in trials}) == 32


def test_l2_axis_mirrors_loss_and_solver():
    cfg = load_experiment_config(EXP_DIR / "synthetic_l2_sweep.yaml")
    trials = expand_trials(cfg)
    assert len(trials) == 11
    for trial in trials:
        assert float(trial.config["loss"]["lambda_l2"]) == float(trial.config["solver"]["lambda_l2"])


def test_quick_subsample_reduces_sweep():
    cfg = load_experiment_config(EXP_DIR / "synthetic_l2_sweep.yaml")
    assert len(expand_trials(cfg)) == 11
    quick = expand_trials(cfg, quick=True)
    assert 1 <= len(quick) <= 3


def test_single_experiment_has_one_trial():
    cfg = load_experiment_config(EXP_DIR / "synthetic_exact_recovery.yaml")
    trials = expand_trials(cfg)
    assert len(trials) == 1
    assert trials[0].name == "synthetic_exact_recovery"


# --------------------------------------------------------------------------- registry / suites


def test_registry_covers_core_experiments():
    assert set(CORE_EXPERIMENTS) == {"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"}
    for eid, info in CORE_EXPERIMENTS.items():
        assert info.eid == eid
        assert info.question
        assert (ROOT / info.config).exists()


def test_suite_paths_exist():
    for name in SUITES:
        for rel in resolve_suite(name):
            assert (ROOT / rel).exists(), f"{name} -> {rel} missing"


# --------------------------------------------------------------------------- summary row


def test_summary_row_has_all_columns_for_failed_run():
    cfg = load_experiment_config(EXP_DIR / "synthetic_exact_recovery.yaml")
    trial = expand_trials(cfg)[0]
    row = summary_row(trial.name, trial.config, {}, error="boom")
    assert set(row.keys()) == set(SUMMARY_COLUMNS)
    assert row["acceptability_status"] == "FAILED"
    assert row["error"] == "boom"


# --------------------------------------------------------------------------- runner / smoke


def test_synthetic_exact_recovery_smoke(tmp_path):
    cfg = load_experiment_config(EXP_DIR / "synthetic_exact_recovery.yaml")
    result = run_experiment(cfg, output_root=tmp_path, quick=True, continue_on_error=False)
    assert len(result.metrics) == 1
    metrics = result.metrics[0]
    # identical truth/model geometry -> recovery limited only by conditioning
    assert metrics["relative_acceleration_rmse"] < 1.0e-3
    assert metrics["diagnostics"]["shell_collapse_flag"] is False
    # per-run artifacts written
    run_dir = tmp_path / "synthetic_exact_recovery"
    for fname in ("metrics.json", "diagnostics.json", "summary.txt", "config.yaml"):
        assert (run_dir / fname).exists()


def test_l2_sweep_creates_summary_table(tmp_path):
    cfg = load_experiment_config(EXP_DIR / "synthetic_l2_sweep.yaml")
    result = run_experiment(cfg, output_root=tmp_path / "runs", quick=True)
    assert len(result.rows) >= 2
    artifacts = write_suite_artifacts(
        tmp_path, result.rows, suite_name="t", git_commit=git_commit_hash(), make_plots=False
    )
    csv_text = Path(artifacts["suite_summary_csv"]).read_text(encoding="utf-8")
    data_lines = [line for line in csv_text.splitlines() if line.strip()]
    assert len(data_lines) == 1 + len(result.rows)  # header + rows
    assert "shell_cancellation_ratio" in data_lines[0]
    assert Path(artifacts["pareto_data_csv"]).exists()
    # higher L2 should reduce the source norm sigma_l2
    by_l2 = sorted(
        ((float(r["lambda_l2"]), float(r["sigma_l2"])) for r in result.rows if r["sigma_l2"] != ""),
        key=lambda item: item[0],
    )
    assert by_l2[0][1] > by_l2[-1][1]


def test_run_diagnostics_contains_required_keys(tmp_path):
    cfg = load_experiment_config(EXP_DIR / "synthetic_multishell_truth.yaml")
    result = run_experiment(cfg, output_root=tmp_path, quick=True)
    diagnostics = result.metrics[0]["diagnostics"]
    required = [
        "sigma_l2",
        "sigma_abs_sum",
        "effective_source_count",
        "top_5pct_source_contribution",
        "relative_monopole_leakage",
        "relative_dipole_leakage",
        "shell_energy_distribution",
        "dominant_shell_energy_fraction",
        "shell_energy_entropy",
        "shell_cancellation_ratio",
        "per_shell_field_rms",
    ]
    for key in required:
        assert key in diagnostics, f"missing diagnostic {key}"
    # entropy diagnostics live on the metrics dict
    for key in ("source_entropy_nats", "positive_negative_entropy_nats", "shell_energy_balance_entropy_nats"):
        assert key in result.metrics[0]


def test_constrained_maxent_respects_misfit_and_does_not_lower_entropy():
    """Constrained MaxEnt must keep misfit within tolerance and not reduce source entropy."""

    import torch

    from vesp.core.solvers import RidgeSolveConfig, solve_discrete_ridge
    from vesp.extensions.entropy import effective_source_entropy
    from vesp.training.maxent import MaxEntSolveConfig, solve_discrete_maxent_constrained

    torch.manual_seed(0)
    # underdetermined system -> a null space exists for entropy to exploit
    n_query, n_source = 24, 48
    operator = torch.randn(n_query, n_source, dtype=torch.float64)
    positions = torch.randn(n_source, 3, dtype=torch.float64)
    weights = torch.ones(n_source, dtype=torch.float64)
    shells = torch.zeros(n_source, dtype=torch.long)
    target = operator @ torch.randn(n_source, dtype=torch.float64)

    ridge = solve_discrete_ridge(
        operator=operator,
        target=target,
        source_positions=positions,
        source_weights=weights,
        shell_ids=shells,
        config=RidgeSolveConfig(lambda_l2=1.0e-4, column_normalize=True),
    )
    cfg = MaxEntSolveConfig(
        entropy_mode="positive_negative",
        lambda_l2=1.0e-4,
        mode="constrained",
        misfit_factor=1.3,
        search_iters=14,
        weight_bounds=(1.0e-3, 1.0e2),
        max_iter=200,
    )
    sigma, info = solve_discrete_maxent_constrained(
        operator, target, positions, weights, shells, cfg, warm_start_sigma=ridge
    )

    # the achieved misfit must respect the constraint (best is only kept when feasible)
    assert info["maxent_misfit"] <= info["target_misfit"] * (1.0 + 1.0e-9)
    # and the max-entropy representative must not have LOWER entropy than ridge
    assert float(effective_source_entropy(sigma, weights)) >= float(effective_source_entropy(ridge, weights)) - 1.0e-9
    assert info["chosen_entropy_weight"] >= 0.0


def test_select_lambda_l2_picks_interior_lcurve_corner():
    import torch

    from vesp.core.regularization import select_lambda_l2
    from vesp.core.solvers import RidgeSolveConfig

    torch.manual_seed(0)
    n_query, n_source = 60, 40
    A = torch.randn(n_query, n_source, dtype=torch.float64)
    A[:, :20] *= 1.0e-3  # ill-condition some columns -> a genuine L-curve corner
    positions = torch.randn(n_source, 3, dtype=torch.float64)
    weights = torch.ones(n_source, dtype=torch.float64)
    shells = torch.zeros(n_source, dtype=torch.long)
    b = A @ torch.randn(n_source, dtype=torch.float64) + 1.0e-2 * torch.randn(n_query, dtype=torch.float64)

    lam, curve = select_lambda_l2(
        A,
        b,
        source_positions=positions,
        source_weights=weights,
        shell_ids=shells,
        base_config=RidgeSolveConfig(column_normalize=True),
    )
    grid = [p["lambda_l2"] for p in curve]
    assert lam in grid
    assert lam != grid[0] and lam != grid[-1]  # an interior corner, not an endpoint


def test_auto_lambda_l2_selection_runs_and_records(tmp_path):
    import copy

    cfg = merge_defaults(copy.deepcopy(load_experiment_config(EXP_DIR / "synthetic_l2_sweep.yaml")["base_config"]))
    cfg["solver"]["type"] = "ridge"
    cfg["solver"]["lambda_l2"] = "auto"
    cfg["loss"]["lambda_l2"] = "auto"
    cfg["output"] = {"output_dir": str(tmp_path), "run_name": "autolam"}
    metrics = run(cfg, model_cls=MultiShellDiscreteVESP)
    selected = metrics.get("selected_lambda_l2")
    assert isinstance(selected, float) and selected > 0.0
    assert metrics.get("lambda_l2_selection") == "L-curve"


def test_maxent_entropy_weight_zero_matches_ridge(tmp_path):
    """entropy_weight=0 MaxEnt (warm-started from ridge) should reproduce the ridge data fit."""

    cfg = load_experiment_config(EXP_DIR / "synthetic_entropy_pareto.yaml")
    base = merge_defaults(copy.deepcopy(cfg["base_config"]))

    ridge_cfg = copy.deepcopy(base)
    ridge_cfg["solver"]["type"] = "ridge"
    ridge_cfg["output"] = {"output_dir": str(tmp_path), "run_name": "ridge"}
    ridge_metrics = run(ridge_cfg, model_cls=MultiShellDiscreteVESP)

    maxent_cfg = copy.deepcopy(base)
    maxent_cfg["solver"]["type"] = "maxent"
    maxent_cfg["loss"]["entropy_weight"] = 0.0
    maxent_cfg["output"] = {"output_dir": str(tmp_path), "run_name": "maxent0"}
    maxent_metrics = run(maxent_cfg, model_cls=MultiShellDiscreteVESP)

    ridge_rel = ridge_metrics["relative_acceleration_rmse"]
    maxent_rel = maxent_metrics["relative_acceleration_rmse"]
    assert maxent_rel == pytest.approx(ridge_rel, rel=0.1, abs=1.0e-6)
