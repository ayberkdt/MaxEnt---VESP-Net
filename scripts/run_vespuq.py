"""Thin CLI shim for the VESP-UQ experiments. Prefer:

    python -m vesp.uq.run --config configs/vespuq/vespuq_real_lunar.yaml

This script forwards to the same entry point so it works from a bare checkout.
"""

from vesp.uq.run import main

if __name__ == "__main__":
    main()
