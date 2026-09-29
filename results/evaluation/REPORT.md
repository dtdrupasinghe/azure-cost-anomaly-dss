# Evaluation report

Runs: 2 datasets x 6 incident sizes x 10 seeds = 120 runs per detector. Evaluation excludes the first 14 warm-up days. Labels are used only for scoring, never for fitting; model settings were fixed in advance, not tuned on labels.


## Models compared

| Detector | Type | Rule / model |
|---|---|---|
| consensus | Ensemble (final DSS) | anomalous when >= 2 of isolation_forest, median_mad, moving_avg, three_sigma agree |
| isolation_forest | ML | 200 trees, contamination 0.09, 4 features |
| one_class_svm | ML | RBF kernel, nu 0.09, standardised features |
| lof | ML | Local Outlier Factor, 10 neighbours, contamination 0.09 |
| kmeans | ML | 3 clusters; distance to nearest centre, top 9% flagged |
| median_mad | Statistical | > trailing-14-day median + 3.5 robust SD |
| moving_avg | Statistical | > 1.2 x trailing 7-day mean |
| three_sigma | Statistical | > trailing-14-day mean + 3 SD |
| ewma | Statistical | > EWMA(span 7) mean + 3 EWMA SD |
| fixed_budget | Statistical | > 1.3 x mean of the first 14 days |

All ML models use the same 4 features: daily total, deviation from the 7-day median, day of week, weekend flag.


## Detector selection (F1 averaged over all incident sizes)

The DSS uses the detector with the best F1 across both datasets. The consensus rule was fixed before the extra ML models were added, so it was not tuned on these results.

| detector | type | startup | kaggle | both |
|---|---|---|---|---|
| consensus | Ensemble | 0.80 | 0.52 | 0.66 |
| median_mad | Statistical | 0.84 | 0.37 | 0.60 |
| moving_avg | Statistical | 0.54 | 0.56 | 0.55 |
| isolation_forest | ML | 0.63 | 0.43 | 0.53 |
| ewma | Statistical | 0.64 | 0.38 | 0.51 |
| three_sigma | Statistical | 0.59 | 0.42 | 0.51 |
| kmeans | ML | 0.49 | 0.31 | 0.40 |
| one_class_svm | ML | 0.42 | 0.29 | 0.36 |
| lof | ML | 0.23 | 0.38 | 0.30 |
| fixed_budget | Statistical | 0.26 | 0.26 | 0.26 |

## ML model evaluation

### 1. Threshold-free ranking quality (ROC-AUC and PR-AUC, all sizes, mean ± SD over runs)

ROC-AUC = chance that a random anomalous day scores higher than a random normal day (0.5 = random). PR-AUC focuses on the rare anomalous days.


**startup**

| detector | type | ROC-AUC | PR-AUC |
|---|---|---|---|
| consensus | Ensemble | 0.983 ± 0.032 | 0.920 ± 0.148 |
| moving_avg | Statistical | 0.979 ± 0.038 | 0.930 ± 0.131 |
| three_sigma | Statistical | 0.973 ± 0.039 | 0.894 ± 0.111 |
| median_mad | Statistical | 0.973 ± 0.044 | 0.919 ± 0.126 |
| ewma | Statistical | 0.972 ± 0.039 | 0.894 ± 0.117 |
| fixed_budget | Statistical | 0.929 ± 0.090 | 0.736 ± 0.289 |
| isolation_forest | ML | 0.927 ± 0.090 | 0.697 ± 0.226 |
| kmeans | ML | 0.819 ± 0.129 | 0.524 ± 0.188 |
| one_class_svm | ML | 0.750 ± 0.088 | 0.425 ± 0.119 |
| lof | ML | 0.629 ± 0.143 | 0.216 ± 0.087 |

**kaggle**

| detector | type | ROC-AUC | PR-AUC |
|---|---|---|---|
| moving_avg | Statistical | 0.852 ± 0.152 | 0.657 ± 0.303 |
| ewma | Statistical | 0.831 ± 0.141 | 0.578 ± 0.240 |
| consensus | Ensemble | 0.827 ± 0.164 | 0.617 ± 0.295 |
| fixed_budget | Statistical | 0.799 ± 0.153 | 0.556 ± 0.281 |
| isolation_forest | ML | 0.792 ± 0.173 | 0.534 ± 0.281 |
| three_sigma | Statistical | 0.782 ± 0.146 | 0.499 ± 0.227 |
| lof | ML | 0.776 ± 0.175 | 0.479 ± 0.280 |
| one_class_svm | ML | 0.751 ± 0.132 | 0.381 ± 0.178 |
| median_mad | Statistical | 0.742 ± 0.139 | 0.335 ± 0.140 |
| kmeans | ML | 0.700 ± 0.129 | 0.395 ± 0.164 |

### 2. ML models at incident size 30% (mean ± SD over 10 seeds)


**startup**

| detector | F1 | incidents caught | false alerts / 30d | ROC-AUC |
|---|---|---|---|---|
| consensus | 0.98 ± 0.03 | 98% | 0.0 | 1.000 |
| isolation_forest | 0.72 ± 0.09 | 75% | 0.8 | 0.973 |
| kmeans | 0.47 ± 0.12 | 44% | 1.1 | 0.787 |
| one_class_svm | 0.46 ± 0.06 | 51% | 1.4 | 0.808 |
| lof | 0.23 ± 0.11 | 26% | 2.3 | 0.601 |

**kaggle**

| detector | F1 | incidents caught | false alerts / 30d | ROC-AUC |
|---|---|---|---|---|
| consensus | 0.56 ± 0.07 | 68% | 2.1 | 0.912 |
| isolation_forest | 0.46 ± 0.05 | 52% | 1.6 | 0.874 |
| lof | 0.33 ± 0.12 | 42% | 2.0 | 0.837 |
| kmeans | 0.31 ± 0.07 | 40% | 2.1 | 0.766 |
| one_class_svm | 0.31 ± 0.14 | 40% | 1.6 | 0.829 |

### 3. Walk-forward (deployment-style) vs whole-period fitting, incident size 30%

Walk-forward retrains each ML model every 7 days on past days only and scores the next 7 days, as a live deployment would. Whole-period fitting sees all days at once.

| dataset | detector | f1_whole | roc_auc_whole | f1_walk | roc_auc_walk |
|---|---|---|---|---|---|
| kaggle | isolation_forest | 0.46 | 0.87 | 0.35 | 0.76 |
| kaggle | kmeans | 0.31 | 0.77 | 0.28 | 0.69 |
| kaggle | lof | 0.33 | 0.84 | 0.30 | 0.70 |
| kaggle | one_class_svm | 0.31 | 0.83 | 0.29 | 0.69 |
| startup | isolation_forest | 0.72 | 0.97 | 0.66 | 0.97 |
| startup | kmeans | 0.47 | 0.79 | 0.51 | 0.81 |
| startup | lof | 0.23 | 0.60 | 0.67 | 0.88 |
| startup | one_class_svm | 0.46 | 0.81 | 0.35 | 0.83 |

### 4. Sensitivity to the contamination setting (incident size 30%, F1)

Contamination = expected share of anomalous days. 0.09 is the fixed prior used everywhere else.

| dataset | detector | c=0.03 | c=0.05 | c=0.09 | c=0.15 |
|---|---|---|---|---|---|
| kaggle | isolation_forest | 0.39 | 0.42 | 0.43 | 0.51 |
| kaggle | kmeans | 0.28 | 0.29 | 0.31 | 0.41 |
| kaggle | lof | 0.31 | 0.34 | 0.33 | 0.39 |
| kaggle | one_class_svm | 0.19 | 0.17 | 0.31 | 0.46 |
| startup | isolation_forest | 0.45 | 0.56 | 0.73 | 0.75 |
| startup | kmeans | 0.42 | 0.46 | 0.47 | 0.46 |
| startup | lof | 0.10 | 0.18 | 0.23 | 0.22 |
| startup | one_class_svm | 0.18 | 0.35 | 0.46 | 0.54 |

### 5. Statistical significance (Wilcoxon signed-rank test on paired F1, 60 runs per dataset)

p < 0.05 means the difference is unlikely to be chance. `ref_better_runs` = runs where the first detector had the higher F1.

| dataset | comparison | mean_f1_diff | ref_better_runs | p_value |
|---|---|---|---|---|
| startup | consensus vs isolation_forest | +0.168 | 56/60 | 0.0000 |
| startup | consensus vs one_class_svm | +0.377 | 55/60 | 0.0000 |
| startup | consensus vs lof | +0.574 | 56/60 | 0.0000 |
| startup | consensus vs kmeans | +0.306 | 54/60 | 0.0000 |
| startup | consensus vs median_mad | -0.037 | 14/60 | 0.0078 |
| startup | consensus vs moving_avg | +0.256 | 40/60 | 0.0000 |
| startup | consensus vs three_sigma | +0.212 | 53/60 | 0.0000 |
| startup | consensus vs ewma | +0.163 | 50/60 | 0.0000 |
| startup | consensus vs fixed_budget | +0.535 | 50/60 | 0.0000 |
| kaggle | consensus vs isolation_forest | +0.087 | 45/60 | 0.0000 |
| kaggle | consensus vs one_class_svm | +0.232 | 57/60 | 0.0000 |
| kaggle | consensus vs lof | +0.141 | 47/60 | 0.0000 |
| kaggle | consensus vs kmeans | +0.210 | 51/60 | 0.0000 |
| kaggle | consensus vs median_mad | +0.150 | 55/60 | 0.0000 |
| kaggle | consensus vs moving_avg | -0.042 | 13/60 | 0.0006 |
| kaggle | consensus vs three_sigma | +0.097 | 45/60 | 0.0000 |
| kaggle | consensus vs ewma | +0.137 | 46/60 | 0.0000 |
| kaggle | consensus vs fixed_budget | +0.260 | 49/60 | 0.0000 |
| startup | isolation_forest vs one_class_svm | +0.209 | 51/60 | 0.0000 |
| startup | isolation_forest vs lof | +0.405 | 51/60 | 0.0000 |
| startup | isolation_forest vs kmeans | +0.138 | 41/60 | 0.0000 |
| kaggle | isolation_forest vs one_class_svm | +0.145 | 41/60 | 0.0000 |
| kaggle | isolation_forest vs lof | +0.054 | 25/60 | 0.0038 |
| kaggle | isolation_forest vs kmeans | +0.123 | 37/60 | 0.0000 |

## Detection - startup (incident size 30% of a normal day)

| detector | precision | recall_days | f1 | event_recall | false_alerts_per_30d | mean_delay_days | cost_exposed_pct | roc_auc |
|---|---|---|---|---|---|---|---|---|
| consensus | 1.00 | 0.96 | 0.98 | 98% | 0.0 | 0.00 | 2% | 1.00 |
| ewma | 0.97 | 0.61 | 0.74 | 88% | 0.1 | 0.00 | 11% | 0.99 |
| fixed_budget | 0.80 | 0.08 | 0.14 | 9% | 0.0 | 0.06 | 87% | 1.00 |
| isolation_forest | 0.74 | 0.70 | 0.72 | 75% | 0.8 | 0.07 | 23% | 0.97 |
| kmeans | 0.54 | 0.42 | 0.47 | 44% | 1.1 | 0.14 | 52% | 0.79 |
| lof | 0.24 | 0.22 | 0.23 | 26% | 2.3 | 0.13 | 74% | 0.60 |
| median_mad | 0.98 | 0.96 | 0.97 | 97% | 0.1 | 0.00 | 4% | 1.00 |
| moving_avg | 1.00 | 0.88 | 0.93 | 93% | 0.0 | 0.00 | 6% | 1.00 |
| one_class_svm | 0.49 | 0.43 | 0.46 | 51% | 1.4 | 0.14 | 46% | 0.81 |
| three_sigma | 0.97 | 0.51 | 0.66 | 63% | 0.1 | 0.03 | 40% | 0.99 |

### Event recall by incident size - startup

| detector | 5% | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| consensus | 24% | 57% | 86% | 98% | 100% | 100% |
| ewma | 26% | 60% | 83% | 88% | 90% | 91% |
| fixed_budget | 0% | 0% | 0% | 9% | 29% | 100% |
| isolation_forest | 31% | 51% | 67% | 75% | 84% | 88% |
| kmeans | 37% | 62% | 38% | 44% | 52% | 59% |
| lof | 36% | 37% | 30% | 26% | 15% | 8% |
| median_mad | 27% | 71% | 97% | 97% | 98% | 100% |
| moving_avg | 0% | 0% | 31% | 93% | 100% | 100% |
| one_class_svm | 35% | 46% | 44% | 51% | 52% | 52% |
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

| detector | precision | recall_days | f1 | event_recall | false_alerts_per_30d | mean_delay_days | cost_exposed_pct | roc_auc |
|---|---|---|---|---|---|---|---|---|
| consensus | 0.52 | 0.61 | 0.56 | 68% | 2.1 | 0.02 | 33% | 0.91 |
| ewma | 0.64 | 0.38 | 0.47 | 58% | 0.8 | 0.00 | 43% | 0.90 |
| fixed_budget | 0.15 | 1.00 | 0.27 | 100% | 20.8 | 0.00 | 0% | 0.84 |
| isolation_forest | 0.50 | 0.43 | 0.46 | 52% | 1.6 | 0.10 | 46% | 0.87 |
| kmeans | 0.34 | 0.30 | 0.31 | 40% | 2.1 | 0.25 | 62% | 0.77 |
| lof | 0.36 | 0.31 | 0.33 | 42% | 2.0 | 0.19 | 57% | 0.84 |
| median_mad | 0.30 | 0.43 | 0.35 | 48% | 3.8 | 0.00 | 54% | 0.79 |
| moving_avg | 0.62 | 0.76 | 0.68 | 80% | 1.7 | 0.00 | 23% | 0.93 |
| one_class_svm | 0.38 | 0.26 | 0.31 | 40% | 1.6 | 0.13 | 60% | 0.83 |
| three_sigma | 0.57 | 0.41 | 0.47 | 55% | 1.2 | 0.03 | 51% | 0.83 |

### Event recall by incident size - kaggle

| detector | 5% | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| consensus | 32% | 40% | 60% | 68% | 88% | 100% |
| ewma | 13% | 18% | 40% | 58% | 70% | 85% |
| fixed_budget | 90% | 93% | 100% | 100% | 100% | 100% |
| isolation_forest | 18% | 20% | 40% | 52% | 73% | 87% |
| kmeans | 17% | 17% | 30% | 40% | 53% | 60% |
| lof | 17% | 25% | 30% | 42% | 67% | 83% |
| median_mad | 30% | 35% | 47% | 48% | 73% | 93% |
| moving_avg | 22% | 35% | 63% | 80% | 97% | 100% |
| one_class_svm | 15% | 18% | 40% | 40% | 47% | 50% |
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
| 2023-02-01 | 164.49 | 133.68 | +30.81 (23%) | consensus, isolation_forest, lof, moving_avg, three_sigma, ewma, fixed_budget | Azure Firewall +12.52; Virtual Machines +7.86; Azure Database for MariaDB +3.73 |
| 2023-02-03 | 169.61 | 136.01 | +33.60 (25%) | consensus, isolation_forest, lof, kmeans, moving_avg, fixed_budget | Azure Synapse Analytics +54.95; Azure Database for MariaDB +2.67; Log Analytics +2.62 |
| 2023-03-15 | 130.16 | 96.17 | +34.00 (35%) | consensus, isolation_forest, one_class_svm, median_mad, moving_avg, three_sigma, ewma, fixed_budget | Virtual Machines +15.22; SQL Database +15.17; Storage +4.16 |
| 2023-03-17 | 138.57 | 97.78 | +40.78 (42%) | consensus, median_mad, moving_avg, fixed_budget | Virtual Machines +28.25; Log Analytics +6.32; Storage +3.60 |
| 2023-03-18 | 147.01 | 111.39 | +35.62 (32%) | consensus, isolation_forest, one_class_svm, lof, kmeans, median_mad, moving_avg, fixed_budget | Virtual Machines +36.92; Azure Defender +4.83; Log Analytics +3.38 |

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