import torch

from vesp.core.kernels import build_dense_operator
from vesp.core.models import DiscreteVESP
from vesp.core.sources import make_shell_sources


def test_dense_operator_matches_model_forward_row_order_and_sign():
    dtype = torch.float64
    sources = make_shell_sources([0.62], 8, dtype=dtype)
    model = DiscreteVESP(sources, dtype=dtype)
    sigma = torch.linspace(-0.4, 0.5, sources.n_sources, dtype=dtype)
    model.set_sigma(sigma)

    query = torch.tensor(
        [
            [1.10, 0.00, 0.00],
            [0.00, 1.25, 0.10],
            [-0.40, 0.20, 1.35],
        ],
        dtype=dtype,
    )

    for sign in (1.0, -1.0):
        pred_u, pred_a = model(query, acceleration_sign=sign)
        operator = build_dense_operator(
            query,
            model.source_positions,
            model.source_weights,
            include_potential=True,
            include_acceleration=True,
            acceleration_sign=sign,
        )
        stacked = operator @ sigma
        n = query.shape[0]

        assert pred_u is not None
        assert pred_a is not None
        assert torch.allclose(stacked[:n], pred_u.reshape(-1), atol=1.0e-12)
        assert torch.allclose(stacked[n : 2 * n], pred_a[:, 0], atol=1.0e-12)
        assert torch.allclose(stacked[2 * n : 3 * n], pred_a[:, 1], atol=1.0e-12)
        assert torch.allclose(stacked[3 * n : 4 * n], pred_a[:, 2], atol=1.0e-12)
