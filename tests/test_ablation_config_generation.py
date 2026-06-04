from vesp.training.run_ablation import _trial_configs


def _base():
    return {"model": {"type": "discrete", "shell_alpha": 0.86, "n_source": 64}, "data": {"type": "synthetic"}}


def _config(**extra):
    cfg = {"base_config": _base(), "ablation": {"mode": "quick"}}
    cfg.update(extra)
    return cfg


def test_quick_explicit_list_count():
    cfg = _config(quick=[{"name": "A", "loss.lambda_l2": 1.0e-6}, {"name": "B", "loss.lambda_l2": 1.0e-5}])
    trials = _trial_configs(cfg, mode="quick")
    assert len(trials) == 2
    names = {t["output"]["run_name"] for t in trials}
    assert names == {"A", "B"}
    # override actually applied through the dotted path
    assert trials[0]["loss"]["lambda_l2"] == 1.0e-6


def test_full_grid_cartesian_size():
    cfg = _config(full={"grid": {"loss.lambda_l2": [1e-7, 1e-6, 1e-5], "loss.lambda_moment": [1e-5, 1e-4]}})
    trials = _trial_configs(cfg, mode="full")
    assert len(trials) == 6  # 3 x 2


def test_full_explicit_list_for_shell_sets():
    cfg = _config(
        full=[
            {"name": "S1", "model.shell_alphas": [0.5, 0.8, 0.86], "model.n_sources_per_shell": [8, 8, 8]},
            {"name": "S2", "model.shell_alphas": [0.7, 0.86, 0.95], "model.n_sources_per_shell": [8, 8, 8]},
        ]
    )
    trials = _trial_configs(cfg, mode="full")
    assert len(trials) == 2
    assert trials[1]["model"]["shell_alphas"] == [0.7, 0.86, 0.95]


def test_mode_override_selects_spec():
    cfg = _config(
        quick=[{"name": "q"}],
        full={"grid": {"loss.lambda_l2": [1e-6, 1e-5]}},
    )
    assert len(_trial_configs(cfg, mode="quick")) == 1
    assert len(_trial_configs(cfg, mode="full")) == 2


def test_legacy_schema_still_supported():
    legacy = {
        "data": {"type": "synthetic"},
        "ablation": {"single_shell_alphas": [0.8, 0.86], "n_sources": [64], "multishell_sets": [[0.5, 0.8, 0.95]]},
    }
    trials = _trial_configs(legacy)
    # 2 single-shell + 1 multishell
    assert len(trials) == 3
