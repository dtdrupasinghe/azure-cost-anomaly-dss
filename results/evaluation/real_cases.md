# Real cost spikes explained (no injected data)

Driver ranking = cost_delta (increase vs each service's trailing 7-day median).

## Public Azure subscription (Kaggle) - five largest day-over-normal increases

| Date | Cost | Normal | Increase | Flagged by | Top drivers (increase) |
|---|---|---|---|---|---|
| 2023-02-01 | 164.49 | 133.68 | +30.81 (23%) | isolation_forest, three_sigma, moving_avg, fixed_budget | Azure Firewall +12.52; Virtual Machines +7.86; Azure Database for MariaDB +3.73 |
| 2023-02-03 | 169.61 | 136.01 | +33.60 (25%) | isolation_forest, moving_avg, fixed_budget | Azure Synapse Analytics +54.95; Azure Database for MariaDB +2.67; Log Analytics +2.62 |
| 2023-03-15 | 130.16 | 96.17 | +34.00 (35%) | isolation_forest, median_mad, three_sigma, moving_avg, fixed_budget | Virtual Machines +15.22; SQL Database +15.17; Storage +4.16 |
| 2023-03-17 | 138.57 | 97.78 | +40.78 (42%) | median_mad, moving_avg, fixed_budget | Virtual Machines +28.25; Log Analytics +6.32; Storage +3.60 |
| 2023-03-18 | 147.01 | 111.39 | +35.62 (32%) | isolation_forest, median_mad, moving_avg, fixed_budget | Virtual Machines +36.92; Azure Defender +4.83; Log Analytics +3.38 |

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