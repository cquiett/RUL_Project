# Aircraft Engine RUL Prediction
### Predictive Maintenance Using NASA CMAPSS FD001

---

## Overview

This project demonstrates end-to-end data science applied to aviation predictive maintenance — a domain directly relevant to C-130 and DoD supply chain operations. Using NASA's CMAPSS turbofan engine dataset, I build and evaluate both **regression** and **classification** models to predict when aircraft engines will fail.

This type of modeling directly supports supply chain optimization: knowing when a component will fail allows maintenance crews and logistics teams to pre-position replacement parts, reduce aircraft downtime, and avoid costly unscheduled maintenance events.

---

## Problem Statement

**Regression:** Given sensor readings from an engine at a point in time, predict how many operational cycles remain before failure (Remaining Useful Life).

**Classification:** Given the same sensor readings, predict whether the engine will fail within the next 30 cycles (binary: Yes / No).

---

## Dataset

**Source:** NASA Prognostics Center of Excellence — CMAPSS (Commercial Modular Aero-Propulsion System Simulation)

**Subset used:** FD001 (single operating condition, single fault mode)

## Data
Download the CMAPSS FD001 dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place the files in a `CMAPSSdata/` folder before running the script.

| | Count |
|---|---|
| Training engines | 100 |
| Training rows | 20,631 |
| Test engines | 100 |
| Sensors | 21 (14 used after variance filtering) |

Each row represents one engine at one time cycle. Engines run until failure, so the last cycle observed = failure point. RUL is calculated as:

```
RUL = max_cycle_for_engine - current_cycle
```

RUL is capped at 125 cycles — a common practice that removes noise from very early, healthy-engine readings and focuses the model on the degradation window.

---

## Methodology

### 1. Data Loading & Provenance
Raw data has no headers. Column names assigned manually per NASA documentation:
- Columns 1–2: `unit_id`, `cycle`
- Columns 3–5: `op_setting_1/2/3`
- Columns 6–26: `sensor_1` through `sensor_21`

### 2. Feature Engineering
- RUL calculated from training data (run-to-failure structure)
- RUL for test set reconstructed using provided ground-truth file (`RUL_FD001.txt`)
- Binary label `failure_soon` created: `1` if RUL ≤ 30, else `0`

### 3. Data Quality Assessment
Seven sensors dropped due to near-zero variance across all engines:
`sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19`

These sensors carry no discriminative signal in FD001's single operating condition.

### 4. Scaling
All features normalized to [0, 1] using MinMaxScaler fit on training data only (no data leakage).

### 5. Models
| Task | Models Compared |
|---|---|
| Regression | Linear Regression, Random Forest Regressor |
| Classification | Logistic Regression, Random Forest Classifier |

---

## Results

### Regression — Predict Exact RUL

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 20.60 | 16.37 | 0.736 |
| **Random Forest** | **17.08** | **11.99** | **0.818** |

The Random Forest model explains **81.8%** of the variance in RUL with a mean absolute error of ~12 cycles — meaning on average, predictions are within 12 engine cycles of the true value.

### Classification — Predict Failure Within 30 Cycles

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 92% | 0.967 |
| **Random Forest** | **91%** | **0.981** |

The Random Forest classifier achieves an **AUC of 0.981** — near-perfect separation between engines approaching failure and healthy engines.

---

## Visualizations

| Plot | Description |
|---|---|
| `01_eda.png` | RUL distribution, engine lifetimes, class balance |
| `02_sensor_degradation.png` | Top sensor readings over time for sample engines |
| `03_regression_results.png` | Predicted vs. actual RUL scatter plots |
| `04_confusion_matrices.png` | Classification confusion matrices |
| `05_roc_curves.png` | ROC curves with AUC scores |
| `06_feature_importance.png` | Top 10 features by Random Forest importance |

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run full analysis
python rul_analysis.py
```

## Run `python rul_analysis.py` to regenerate all plots in the `plots/` directory.
---

## Key Takeaways & Supply Chain Relevance

1. **Sensor selection matters.** 7 of 21 sensors were removed as uninformative — in real systems this reduces telemetry costs and model complexity.

2. **Classification is more actionable than regression** for logistics decisions. A binary "will this fail in the next 30 cycles?" is easier to act on than an exact number.

3. **Random Forest outperforms linear models** on both tasks, suggesting non-linear degradation patterns — consistent with real engine wear behavior.

4. **This framework generalizes** to any run-to-failure asset: C-130 hydraulic systems, landing gear actuators, auxiliary power units, or any component with sensor telemetry and maintenance records.

---

## File Structure

```
rul_project/
├── rul_analysis.py          # Full analysis script
├── README.md                # This file
└── plots/
    ├── 01_eda.png
    ├── 02_sensor_degradation.png
    ├── 03_regression_results.png
    ├── 04_confusion_matrices.png
    ├── 05_roc_curves.png
    └── 06_feature_importance.png
```

---

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. *Proceedings of the 1st International Conference on Prognostics and Health Management.*
- NASA CMAPSS Dataset: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
