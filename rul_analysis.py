# =============================================================================
# Aircraft Engine RUL Prediction — CMAPSS FD001
# Portfolio Project
# =============================================================================
# This script covers:
#   1. Data loading & feature engineering
#   2. Exploratory Data Analysis (EDA)
#   3. Regression Model  — predict exact RUL
#   4. Classification Model — predict failure within 30 cycles
#   5. Model evaluation & visualizations
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, roc_auc_score,
                             roc_curve)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4d',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#a0a0b0',
    'ytick.color':      '#a0a0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
ACCENT   = '#00c8ff'
ACCENT2  = '#ff6b6b'
ACCENT3  = '#7bed9f'
ACCENT4  = '#ffa502'

# =============================================================================
# 1. LOAD DATA
# =============================================================================
COLS = (['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
        [f'sensor_{i}' for i in range(1, 22)])

train = pd.read_csv('CMAPSSdata/train_FD001.txt',
                    sep=r'\s+', header=None, names=COLS)
test  = pd.read_csv('test_FD001.txt',
                    sep=r'\s+', header=None, names=COLS)
rul_test = pd.read_csv('CMAPSSdata/RUL_FD001.txt',
                       header=None, names=['RUL_true'])

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")
print(f"Engines (train): {train['unit_id'].nunique()}")

# =============================================================================
# 2. FEATURE ENGINEERING — Calculate RUL
# =============================================================================
# Training set: RUL = max_cycle - current_cycle  (engine runs to failure)
max_cycles        = train.groupby('unit_id')['cycle'].max().reset_index()
max_cycles.columns = ['unit_id', 'max_cycle']
train = train.merge(max_cycles, on='unit_id')
train['RUL'] = train['max_cycle'] - train['cycle']
train.drop(columns='max_cycle', inplace=True)

# Cap RUL at 125 — engines degrade meaningfully only in the last ~125 cycles
# (common practice; reduces noise from very early, healthy cycles)
RUL_CAP = 125
train['RUL'] = train['RUL'].clip(upper=RUL_CAP)

# Test set: RUL at last observed cycle comes from RUL_FD001.txt
last_test = test.groupby('unit_id')['cycle'].max().reset_index()
last_test.columns = ['unit_id', 'last_cycle']
rul_test['unit_id'] = range(1, len(rul_test) + 1)
test = test.merge(last_test, on='unit_id')
test = test.merge(rul_test, on='unit_id')
# Reconstruct full RUL for each row in test
test['RUL'] = test['RUL_true'] + (test['last_cycle'] - test['cycle'])
test['RUL'] = test['RUL'].clip(upper=RUL_CAP)
test.drop(columns=['last_cycle', 'RUL_true'], inplace=True)

# Binary classification label: failure within 30 cycles?
FAILURE_WINDOW = 30
train['failure_soon'] = (train['RUL'] <= FAILURE_WINDOW).astype(int)
test['failure_soon']  = (test['RUL']  <= FAILURE_WINDOW).astype(int)

print(f"\nClass balance (train): {train['failure_soon'].value_counts().to_dict()}")

# =============================================================================
# 3. DROP LOW-VARIANCE SENSORS
# =============================================================================
# Several sensors are constant in FD001 — they add noise, not signal
low_var = [col for col in [f'sensor_{i}' for i in range(1, 22)]
           if train[col].std() < 0.01]
print(f"\nDropping low-variance sensors: {low_var}")
train.drop(columns=low_var, inplace=True)
test.drop(columns=low_var,  inplace=True)

sensor_cols  = [c for c in train.columns if c.startswith('sensor_')]
op_cols      = ['op_setting_1', 'op_setting_2', 'op_setting_3']
feature_cols = op_cols + sensor_cols

# =============================================================================
# 4. SCALE FEATURES
# =============================================================================
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols]  = scaler.transform(test[feature_cols])

# =============================================================================
# 5. TRAIN / TEST SPLITS
# =============================================================================
X_train = train[feature_cols]
y_reg_train  = train['RUL']
y_cls_train  = train['failure_soon']

# For test set use only last observed cycle per engine (standard eval protocol)
test_last = test.groupby('unit_id').last().reset_index()
X_test       = test_last[feature_cols]
y_reg_test   = test_last['RUL']
y_cls_test   = test_last['failure_soon']

# =============================================================================
# 6. REGRESSION MODELS
# =============================================================================
print("\n── Regression ──────────────────────────────────")

lr = LinearRegression()
lr.fit(X_train, y_reg_train)
lr_preds = lr.predict(X_test).clip(0, RUL_CAP)

rf_reg = RandomForestRegressor(n_estimators=150, max_depth=12,
                                random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_reg_train)
rf_preds = rf_reg.predict(X_test).clip(0, RUL_CAP)

def reg_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:30s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return rmse, mae, r2

lr_rmse,  lr_mae,  lr_r2  = reg_metrics("Linear Regression",      y_reg_test, lr_preds)
rf_rmse,  rf_mae,  rf_r2  = reg_metrics("Random Forest Regressor", y_reg_test, rf_preds)

# =============================================================================
# 7. CLASSIFICATION MODELS
# =============================================================================
print("\n── Classification ──────────────────────────────")

log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X_train, y_cls_train)
log_preds     = log_reg.predict(X_test)
log_proba     = log_reg.predict_proba(X_test)[:, 1]

rf_cls = RandomForestClassifier(n_estimators=150, max_depth=12,
                                 random_state=42, n_jobs=-1)
rf_cls.fit(X_train, y_cls_train)
rf_cls_preds  = rf_cls.predict(X_test)
rf_cls_proba  = rf_cls.predict_proba(X_test)[:, 1]

print("\nLogistic Regression:")
print(classification_report(y_cls_test, log_preds,
                             target_names=['Safe', 'Failure Soon']))
print("Random Forest Classifier:")
print(classification_report(y_cls_test, rf_cls_preds,
                             target_names=['Safe', 'Failure Soon']))

# =============================================================================
# 8. FEATURE IMPORTANCE
# =============================================================================
importance_df = pd.DataFrame({
    'feature':    feature_cols,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False).head(10)

# =============================================================================
# 9. VISUALISATIONS
# =============================================================================
os.makedirs('/Users/carramahquiett/datasci-portfolio/rul_project/rul_analysis.py', exist_ok=True) if False else None
import os
os.makedirs('/Users/carramahquiett/datasci-portfolio/rul_project/plots', exist_ok=True)

# ── Figure 1: EDA ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Exploratory Data Analysis — CMAPSS FD001', fontsize=14,
             color='white', fontweight='bold', y=1.02)

# 1a: RUL distribution
axes[0].hist(train['RUL'], bins=30, color=ACCENT, edgecolor='black', alpha=0.85)
axes[0].set_title('RUL Distribution (Train)', color='white')
axes[0].set_xlabel('RUL (cycles)')
axes[0].set_ylabel('Count')

# 1b: Engine cycle lengths
cycle_lengths = train.groupby('unit_id')['cycle'].max()
axes[1].hist(cycle_lengths, bins=20, color=ACCENT3, edgecolor='black', alpha=0.85)
axes[1].set_title('Engine Lifetime (cycles)', color='white')
axes[1].set_xlabel('Total Cycles')
axes[1].set_ylabel('Count')

# 1c: Class balance
counts = train['failure_soon'].value_counts()
axes[2].bar(['Safe (>30 cycles)', 'Failure Soon (≤30)'],
            [counts[0], counts[1]],
            color=[ACCENT3, ACCENT2], edgecolor='black', alpha=0.9)
axes[2].set_title('Classification Label Balance', color='white')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('/Users/carramahquiett/datasci-portfolio/rul_project/plots/01_eda.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()

# ── Figure 2: Sensor degradation over time ───────────────────────────────────
top_sensors = importance_df['feature'].head(4).tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Top Sensor Readings vs. Cycle (Sample Engines)',
             fontsize=13, color='white', fontweight='bold')
axes = axes.flatten()
sample_engines = [1, 5, 10, 15, 20]
colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4, '#b388ff']

for idx, sensor in enumerate(top_sensors[:4]):
    for eidx, eng in enumerate(sample_engines):
        eng_data = train[train['unit_id'] == eng].sort_values('cycle')
        axes[idx].plot(eng_data['cycle'], eng_data[sensor],
                       color=colors[eidx], alpha=0.7, linewidth=1.2,
                       label=f'Engine {eng}')
    axes[idx].set_title(sensor, color='white')
    axes[idx].set_xlabel('Cycle')
    axes[idx].set_ylabel('Scaled Value')
    if idx == 0:
        axes[idx].legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig('/Users/carramahquiett/datasci-portfolio/rul_project/plots/02_sensor_degradation.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()

# ── Figure 3: Regression results ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Regression Model Results — Predicted vs. Actual RUL',
             fontsize=13, color='white', fontweight='bold')

for ax, preds, name, color, rmse, r2 in [
    (axes[0], lr_preds,  'Linear Regression',       ACCENT,  lr_rmse,  lr_r2),
    (axes[1], rf_preds,  'Random Forest Regressor',  ACCENT3, rf_rmse,  rf_r2),
]:
    ax.scatter(y_reg_test, preds, alpha=0.5, s=18, color=color)
    lims = [0, RUL_CAP]
    ax.plot(lims, lims, '--', color='white', linewidth=1, alpha=0.6, label='Perfect')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'{name}\nRMSE={rmse:.1f}  R²={r2:.3f}', color='white')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/Users/carramahquiett/datasci-portfolio/rul_project/plots/03_regression_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()

# ── Figure 4: Confusion matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Classification Model — Confusion Matrices',
             fontsize=13, color='white', fontweight='bold')

for ax, preds, name in [
    (axes[0], log_preds,    'Logistic Regression'),
    (axes[1], rf_cls_preds, 'Random Forest Classifier'),
]:
    cm = confusion_matrix(y_cls_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Blues',
                xticklabels=['Safe', 'Failure Soon'],
                yticklabels=['Safe', 'Failure Soon'])
    ax.set_title(name, color='white')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('/Users/carramahquiett/datasci-portfolio/rul_project/plots/04_confusion_matrices.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()

# ── Figure 5: ROC curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('ROC Curves — Classification Models',
             fontsize=13, color='white', fontweight='bold')

for proba, name, color in [
    (log_proba,    'Logistic Regression',       ACCENT),
    (rf_cls_proba, 'Random Forest Classifier',  ACCENT3),
]:
    fpr, tpr, _ = roc_curve(y_cls_test, proba)
    auc = roc_auc_score(y_cls_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

ax.plot([0,1],[0,1],'--', color='white', alpha=0.4, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.set_title('', color='white')

plt.tight_layout()
plt.savefig('/Users/carramahquiett/datasci-portfolio/rul_project/plots/05_roc_curves.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()

# ── Figure 6: Feature importance ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(importance_df['feature'][::-1],
               importance_df['importance'][::-1],
               color=ACCENT, edgecolor='black', alpha=0.85)
ax.set_title('Top 10 Feature Importances (Random Forest Regressor)',
             color='white', fontsize=12)
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('/Users/carramahquiett/datasci-portfolio/rul_project/plots/06_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()


print("\n✅ All plots saved to /Users/carramahquiett/datasci-portfolio/rul_project/plots/")
print(f"\nSummary:")
print(f"  Best Regression  : Random Forest  (RMSE={rf_rmse:.1f}, R²={rf_r2:.3f})")
rf_auc = roc_auc_score(y_cls_test, rf_cls_proba)
print(f"  Best Classifier  : Random Forest  (AUC={rf_auc:.3f})")
