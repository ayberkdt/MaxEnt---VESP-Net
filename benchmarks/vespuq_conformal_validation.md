# VESP-UQ Operational Conformal Validation

This report reruns the real-lunar L60 and L90 residual-band configs with `uq.conformal.apply: true` and per-band conformal scaling enabled. It validates the operational prediction path, not the older audit-only conformal threshold path.

Acceptance target: z_std in [0.70, 1.30], PICP90 in [0.85, 0.95] for low/mid/high bands.

| case | band | baseline z_std | conformal z_std | baseline PICP90 | conformal PICP90 | band scale | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| L60 | low | 1.09 | 0.648 | 0.87 | 0.978 | 1.69 | no |
| L60 | mid | 0.64 | 0.716 | 0.97 | 0.962 | 0.901 | no |
| L60 | high | 0.27 | 0.667 | 1 | 0.986 | 0.41 | no |
| L90 | low | 0.17 | 0.621 | 1 | 0.975 | 0.271 | no |
| L90 | mid | 0.07 | 0.833 | 1 | 0.937 | 0.0842 | yes |
| L90 | high | 0.02 | 0.563 | 1 | 0.98 | 0.035 | no |

## Run Directories

- L60: `conformal_validation_runs\vespuq_real_lunar_conformal`
- L90: `conformal_validation_runs\vespuq_real_lunar_L90_conformal`
