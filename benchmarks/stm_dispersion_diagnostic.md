# VESP-UQ STM Dispersion Diagnostic

Exploratory diagnostic derived from the force-error posterior. This tests whether
weighting the force-error posterior by trajectory dynamics (using `linear_propagation`)
better correlates with long-horizon ST-LRPS position drift.

**WARNING**: This is a diagnostic evaluation, NOT a validated position-error prediction
claim. VESP-UQ remains a force-risk calibration layer.

- **Scoring mode:** `stm_dispersion` (`max(sqrt(trace(P_rr)))` along integrated trajectory)
- **Trajectories:** 512 Keplerian orbits
- **Rerun fraction:** 10%

### Results

- **Spearman correlation:** -0.0477
- **Capture Rate:** 13.5%
- **Precision:** 13.5%
- **Lift over random:** 1.33x
- **Mean true error flagged:** 0.036 km vs accepted: 0.035 km (ratio 1.01x)
- **Random baseline capture:** 10.4% +/- 4.3%
