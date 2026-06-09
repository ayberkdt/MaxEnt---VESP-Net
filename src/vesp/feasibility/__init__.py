"""Feasibility / MaxEnt pillar (Stage 1-3A).

The original deterministic equivalent-source framework -- single/multi-shell ridge/Tikhonov fitting
with entropy regularization kept as an ablation -- now namespaced under ``vesp.feasibility`` so its
training / experiment / analysis / app modules do not collide with the current VESP-UQ layer
(:mod:`vesp.uq`) as new systems are added. Both pillars share the equivalent-source core
(:mod:`vesp.core`) and the :mod:`vesp.common` / :mod:`vesp.data` utilities.
"""
