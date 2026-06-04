import pytest

from vesp.common.config import merge_defaults, validate_config


def _config() -> dict:
    return merge_defaults(
        {
            "body": {"R_body": 1.0, "normalize_positions": True},
            "model": {"type": "discrete", "shell_alpha": 0.8, "n_source": 8},
            "kernel": {"eps": 0.0},
            "solver": {"type": "ridge", "ridge_method": "augmented_lstsq"},
            "loss": {"use_potential": True, "use_acceleration": True, "normalize_targets": False},
            "split": {"type": "random", "train_fraction": 0.8},
        }
    )


def test_invalid_ridge_method_is_rejected():
    config = _config()
    config["solver"]["ridge_method"] = "svd_magic"

    with pytest.raises(ValueError, match="solver.ridge_method"):
        validate_config(config)


def test_negative_kernel_eps_is_rejected():
    config = _config()
    config["kernel"]["eps"] = -1.0e-6

    with pytest.raises(ValueError, match="kernel.eps"):
        validate_config(config)


def test_invalid_target_scale_is_rejected_when_normalization_enabled():
    config = _config()
    config["loss"]["normalize_targets"] = True
    config["loss"]["potential_scale"] = -1.0

    with pytest.raises(ValueError, match="loss.potential_scale"):
        validate_config(config)


def test_multishell_count_length_mismatch_is_rejected():
    config = _config()
    config["model"] = {"type": "multishell", "shell_alphas": [0.5, 0.8], "n_sources_per_shell": [4]}

    with pytest.raises(ValueError, match="n_sources_per_shell"):
        validate_config(config)


def test_normalized_positions_with_nonunit_body_radius_warns():
    config = _config()
    config["body"]["R_body"] = 1738.0

    with pytest.warns(RuntimeWarning, match="body.R_body=1.0"):
        validate_config(config)
