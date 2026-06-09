import torch

from vesp.core.kernels import build_dense_operator, evaluate_kernel
from vesp.core.models import DiscreteVESP
from vesp.core.sources import make_shell_sources
from vesp.data.dataset import ResidualGravityData
from vesp.training.evaluate import evaluate_model
from vesp.training.train_discrete import solve_ridge


def _query_points(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [1.10, 0.00, 0.00],
            [0.00, 1.20, 0.20],
            [0.25, -0.35, 1.30],
            [-1.15, 0.15, 0.05],
            [0.45, 1.10, -0.10],
            [-0.25, -0.35, 1.45],
        ],
        dtype=dtype,
    )


def test_model_forward_acceleration_sign_flips_acceleration_only():
    dtype = torch.float64
    sources = make_shell_sources([0.58], 6, dtype=dtype)
    model = DiscreteVESP(sources, dtype=dtype)
    model.set_sigma(torch.linspace(-0.2, 0.3, sources.n_sources, dtype=dtype))
    query = _query_points(dtype)

    u_pos, a_pos = model(query, acceleration_sign=1.0)
    u_neg, a_neg = model(query, acceleration_sign=-1.0)

    assert torch.allclose(u_pos, u_neg, atol=1.0e-12)
    assert torch.allclose(a_pos, -a_neg, atol=1.0e-12)


def test_dense_operator_sign_matches_model_forward():
    dtype = torch.float64
    sources = make_shell_sources([0.58], 6, dtype=dtype)
    model = DiscreteVESP(sources, dtype=dtype)
    sigma = torch.linspace(-0.2, 0.3, sources.n_sources, dtype=dtype)
    model.set_sigma(sigma)
    query = _query_points(dtype)

    _, acceleration = model(query, acceleration_sign=-1.0)
    operator = build_dense_operator(
        query,
        model.source_positions,
        model.source_weights,
        include_potential=False,
        include_acceleration=True,
        acceleration_sign=-1.0,
    )
    stacked_acceleration = operator @ sigma

    assert acceleration is not None
    assert torch.allclose(stacked_acceleration[: query.shape[0]], acceleration[:, 0], atol=1.0e-12)
    assert torch.allclose(stacked_acceleration[query.shape[0] : 2 * query.shape[0]], acceleration[:, 1], atol=1.0e-12)
    assert torch.allclose(stacked_acceleration[2 * query.shape[0] :], acceleration[:, 2], atol=1.0e-12)


def test_ridge_train_and_evaluate_use_same_acceleration_sign():
    dtype = torch.float64
    sources = make_shell_sources([0.60], 6, dtype=dtype)
    sigma_truth = torch.linspace(-0.25, 0.35, sources.n_sources, dtype=dtype)
    query = _query_points(dtype)
    out = evaluate_kernel(
        query,
        sources.positions,
        sources.weights * sigma_truth,
        compute_potential=True,
        compute_acceleration=True,
        acceleration_sign=-1.0,
    )
    data = ResidualGravityData(
        positions=query,
        potential=out.potential,
        acceleration=out.acceleration,
        metadata={"position_units": "normalized"},
    )
    model = DiscreteVESP(sources, dtype=dtype)
    config = {
        "kernel": {"eps": 0.0, "acceleration_sign": -1.0},
        "solver": {"type": "ridge", "ridge_method": "augmented_lstsq", "column_normalize": True},
        "loss": {
            "use_potential": True,
            "use_acceleration": True,
            "normalize_targets": False,
            "lambda_potential": 1.0,
            "lambda_acceleration": 1.0,
            "lambda_l2": 0.0,
            "lambda_moment": 0.0,
            "lambda_dipole": 1.0,
        },
    }

    solve_ridge(model, data, config, device=torch.device("cpu"))
    same_sign = evaluate_model(model, data, acceleration_sign=-1.0, device="cpu")
    opposite_sign = evaluate_model(model, data, acceleration_sign=1.0, device="cpu")

    assert same_sign["acceleration_rmse"] < 1.0e-10
    assert opposite_sign["acceleration_rmse"] > 1.0e-3
