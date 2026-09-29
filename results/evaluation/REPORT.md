# Evaluation report

Runs: 2 datasets x 6 incident sizes x 10 seeds. Evaluation excludes the first 14 warm-up days.


## Detector selection (F1 averaged over all incident sizes)

The DSS uses the detector with the best F1 across both datasets.

| detector | startup | kaggle | both |
|---|---|---|---|
| consensus | 0.80 | 0.52 | 0.66 |
| median_mad | 0.84 | 0.37 | 0.60 |
| moving_avg | 0.54 | 0.56 | 0.55 |
| isolation_forest | 0.63 | 0.43 | 0.53 |
| three_sigma | 0.59 | 0.42 | 0.51 |
| fixed_budget | 0.26 | 0.26 | 0.26 |

## Detection - startup (incident size 30% of a normal day)

| detector | precision | recall_days | f1 | event_recall | false_alerts_per_30d | mean_delay_days | cost_exposed_pct |
|---|---|---|---|---|---|---|---|
| consensus | 1.00 | 0.96 | 0.98 | 98% | 0.0 | 0.00 | 2% |
| fixed_budget | 0.80 | 0.08 | 0.14 | 9% | 0.0 | 0.06 | 87% |
| isolation_forest | 0.74 | 0.70 | 0.72 | 75% | 0.8 | 0.07 | 23% |
| median_mad | 0.98 | 0.96 | 0.97 | 97% | 0.1 | 0.00 | 4% |
| moving_avg | 1.00 | 0.88 | 0.93 | 93% | 0.0 | 0.00 | 6% |
| three_sigma | 0.97 | 0.51 | 0.66 | 63% | 0.1 | 0.03 | 40% |

### Event recall by incident size - startup

| detector | 5% | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| consensus | 24% | 57% | 86% | 98% | 100% | 100% |
| fixed_budget | 0% | 0% | 0% | 9% | 29% | 100% |
| isolation_forest | 31% | 51% | 67% | 75% | 84% | 88% |
| median_mad | 27% | 71% | 97% | 97% | 98% | 100% |
| moving_avg | 0% | 0% | 31% | 93% | 100% | 100% |
| three_sigma | 23% | 45% | 61% | 63% | 68% | 73% |

## Attribution - startup (all sizes pooled)

| method | top1 | top3 | mrr | n |
|---|---|---|---|---|
| largest_cost | 33% | 85% | 0.62 | 720 |
| cost_delta | 99% | 100% | 1.00 | 720 |
| robust_z | 92% | 100% | 0.96 | 720 |
| shap_if | 81% | 98% | 0.90 | 720 |

### Top-1 accuracy by incident size - startup

| method | 5% | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| largest_cost | 17% | 17% | 17% | 17% | 34% | 100% |
| cost_delta | 96% | 99% | 100% | 100% | 100% | 100% |
| robust_z | 89% | 91% | 92% | 92% | 92% | 93% |
| shap_if | 72% | 81% | 82% | 83% | 83% | 83% |

### Multi-service incidents: both drivers in top-3

| method | both_in_top3 |
|---|---|
| largest_cost | 33% |
| cost_delta | 98% |
| robust_z | 99% |
| shap_if | 94% |

## Detection - kaggle (incident size 30% of a normal day)

The real background already contains unlabelled real spikes (see the real-cases section), so alerts on them count as false here: precision is a lower bound.

| detector | precision | recall_days | f1 | event_recall | false_alerts_per_30d | mean_delay_days | cost_exposed_pct |
|---|---|---|---|---|---|---|---|
| consensus | 0.52 | 0.61 | 0.56 | 68% | 2.1 | 0.02 | 33% |
| fixed_budget | 0.15 | 1.00 | 0.27 | 100% | 20.8 | 0.00 | 0% |
| isolation_forest | 0.50 | 0.43 | 0.46 | 52% | 1.6 | 0.10 | 46% |
| median_mad | 0.30 | 0.43 | 0.35 | 48% | 3.8 | 0.00 | 54% |
| moving_avg | 0.62 | 0.76 | 0.68 | 80% | 1.7 | 0.00 | 23% |
| three_sigma | 0.57 | 0.41 | 0.47 | 55% | 1.2 | 0.03 | 51% |

### Event recall by incident size - kaggle

| detector | 5% | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| consensus | 32% | 40% | 60% | 68% | 88% | 100% |
| fixed_budget | 90% | 93% | 100% | 100% | 100% | 100% |
| isolation_forest | 18% | 20% | 40% | 52% | 73% | 87% |
| median_mad | 30% | 35% | 47% | 48% | 73% | 93% |
| moving_avg | 22% | 35% | 63% | 80% | 97% | 100% |
| three_sigma | 22% | 33% | 52% | 55% | 62% | 75% |

## Attribution - kaggle (all sizes pooled)

| method | top1 | top3 | mrr | n |
|---|---|---|---|---|
| largest_cost | 67% | 81% | 0.76 | 360 |
| cost_delta | 84% | 98% | 0.91 | 360 |
| robust_z | 65% | 94% | 0.80 | 360 |
| shap_if | 57% | 80% | 0.71 | 360 |

### Top-1 accuracy by incident size - kaggle

| method | 5% | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| largest_cost | 27% | 33% | 52% | 90% | 100% | 100% |
| cost_delta | 50% | 72% | 92% | 93% | 100% | 100% |
| robust_z | 38% | 53% | 70% | 73% | 77% | 80% |
| shap_if | 28% | 42% | 53% | 63% | 73% | 82% |

### Multi-service incidents: both drivers in top-3

| method | both_in_top3 |
|---|---|
| largest_cost | 23% |
| cost_delta | 78% |
| robust_z | 67% |
| shap_if | 42% |

# Real cost spikes explained (no injected data)

Driver ranking = cost_delta (increase vs each service's trailing 7-day median).

## Public Azure subscription (Kaggle) - five largest day-over-normal increases

| Date | Cost | Normal | Increase | Flagged by | Top drivers (increase) |
|---|---|---|---|---|---|
| 2023-02-01 | 164.49 | 133.68 | +30.81 (23%) | consensus, isolation_forest, three_sigma, moving_avg, fixed_budget | Azure Firewall +12.52; Virtual Machines +7.86; Azure Database for MariaDB +3.73 |
| 2023-02-03 | 169.61 | 136.01 | +33.60 (25%) | consensus, isolation_forest, moving_avg, fixed_budget | Azure Synapse Analytics +54.95; Azure Database for MariaDB +2.67; Log Analytics +2.62 |
| 2023-03-15 | 130.16 | 96.17 | +34.00 (35%) | consensus, isolation_forest, median_mad, three_sigma, moving_avg, fixed_budget | Virtual Machines +15.22; SQL Database +15.17; Storage +4.16 |
| 2023-03-17 | 138.57 | 97.78 | +40.78 (42%) | consensus, median_mad, moving_avg, fixed_budget | Virtual Machines +28.25; Log Analytics +6.32; Storage +3.60 |
| 2023-03-18 | 147.01 | 111.39 | +35.62 (32%) | consensus, isolation_forest, median_mad, moving_avg, fixed_budget | Virtual Machines +36.92; Azure Defender +4.83; Log Analytics +3.38 |

## Case organisation (Sri Lankan startup) - 28 June 2026 vs 1 June 2026

Total USD 3.03 vs 1.77 (+71%).

| Service | 1 Jun | 28 Jun | Change |
|---|---|---|---|
| MS Bing Services | 0.000 | 1.302 | +1.302 |
| Foundry Models | 0.032 | 0.542 | +0.510 |
| Azure App Service | 0.000 | 0.150 | +0.150 |
| Traffic Manager | 0.036 | 0.022 | -0.014 |
| Storage | 0.154 | 0.098 | -0.057 |
| Container Registry | 0.167 | 0.097 | -0.069 |
| Virtual Network | 0.240 | 0.150 | -0.090 |
| Virtual Machines | 1.133 | 0.661 | -0.472 |

Note: fixed-price services are ~58% of their normal day on 28 June, so that export likely covers part of the day; the AI-service increase is real regardless.