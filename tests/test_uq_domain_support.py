"""Tests for the domain-support module and its torch / KDTree backends."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from vesp.core.sources import make_shell_sources
from vesp.uq import VESPUQPlugin
from vesp.uq.domain_support import (
    KDTreeDomainSupportBackend,
    TorchDomainSupportBackend,
    make_domain_backend,
)

_HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def _shell_cloud(n, r_lo, r_hi, seed):
    g = torch.Generator().manual_seed(seed)
    radii = r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=torch.float64)
    dirs = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    return dirs, dirs * radii.unsqueeze(1)


def _fit(domain_backend):
    _, pos = _shell_cloud(500, 1.2, 1.5, seed=0)
    g = torch.Generator().manual_seed(1)
    err = 1.0e-4 * torch.randn(pos.shape[0], 3, generator=g, dtype=torch.float64)
    src = make_shell_sources([0.8], [24], dtype=torch.float64)
    plugin = VESPUQPlugin(src, reg_method="lcurve", seed=0, domain_support=True, domain_backend=domain_backend)
    plugin.fit_error(pos, err)
    return plugin, pos


def test_make_domain_backend_torch_default():
    assert isinstance(make_domain_backend("torch"), TorchDomainSupportBackend)


def test_make_domain_backend_unknown_raises():
    with pytest.raises(ValueError):
        make_domain_backend("not_a_backend")


def test_make_domain_backend_kdtree_falls_back_without_scipy():
    backend = make_domain_backend("kdtree")  # torch when scipy is absent, kdtree when present
    if _HAS_SCIPY:
        assert isinstance(backend, KDTreeDomainSupportBackend)
    else:
        assert isinstance(backend, TorchDomainSupportBackend)


def test_torch_backend_kth_distances_matches_bruteforce():
    backend = TorchDomainSupportBackend()
    g = torch.Generator().manual_seed(3)
    train = torch.randn(40, 3, generator=g, dtype=torch.float64)
    query = torch.randn(7, 3, generator=g, dtype=torch.float64)
    got = backend.kth_distances(query, train, 3)
    expected = torch.topk(torch.cdist(query, train), 3, largest=False).values[:, -1]
    assert torch.allclose(got, expected)


def test_fit_info_records_domain_backend():
    plugin, _ = _fit("torch")
    assert plugin.fit_info["domain_backend"] == "torch"


@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")
def test_kdtree_and_torch_domain_scores_agree():
    plugin_t, pos = _fit("torch")
    plugin_k, _ = _fit("kdtree")
    query = pos[:48]
    st = plugin_t.domain_support_score(query)
    sk = plugin_k.domain_support_score(query)
    # both backends compute the exact Euclidean k-th nearest distance -> scores agree closely
    assert torch.allclose(st, sk, atol=1e-9, rtol=1e-6)
