# Explainable Azure Cost Anomaly Decision Support System

Prototype for the BSc (MIS) final-year research *"An Explainable Intelligent Decision Support
System for Azure Cloud Cost Anomaly Management in a Sri Lankan Startup"* (NSBM Green University).

The dashboard identifies unusual daily Azure spend, explains which services drove the increase,
quantifies the financial impact, suggests checks, and records the user's decision. It recommends
actions only; it never changes Azure resources.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

| File | Source |
|---|---|
| `Dataset/clean/augmented_daily.csv` | 181-day daily series anchored on the case organisation's real Azure bills (Jan, Apr, May 2026), with 12 injected incidents of known cause. Resource names anonymised. |
| `Dataset/clean/kaggle_daily_by_service.csv` | Daily per-service totals derived from c.carrucciu, *Azure Subscription Costs*, Kaggle (2023), https://www.kaggle.com/datasets/carrucciu/azure-costs. Raw file not redistributed. |

Raw billing exports are not included in this repository.

## Pipeline

```bash
python scripts/build_real_monthly.py      # consolidate the real exports (raw files required)
python scripts/build_kaggle_daily.py      # per-service daily series from the Kaggle export
python scripts/augment_dataset.py         # anchored 6-month dataset + ground truth
python scripts/evaluate.py                # detectors vs baselines, attribution accuracy
```

Results: `results/evaluation/REPORT.md`.
