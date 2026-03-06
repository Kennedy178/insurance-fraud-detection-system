"""
Model Evaluation Module — Day 13
=================================
Final evaluation on the TEST SET using the production XGBoost model.
Covers all classification metrics, visual analysis, SHAP interpretation,
and business impact — everything needed for the evaluation notebook.

File location: ml_pipeline/model_evaluation.py

Run with:
    cd ml_pipeline
    python model_evaluation.py

Prerequisites (run these first if not done):
    python model_training.py
    python model_training.py baseline
    python model_training.py advanced
    pip install shap
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# ── Paths — all relative to ml_pipeline/ just like model_training.py ──────
MODEL_PATH      = '../models/xgboost_v1.pkl'
THRESHOLD_PATH  = '../models/xgboost_threshold.json'
TEST_DATA_PATH  = '../data/processed/test.csv'
OUTPUT_DIR      = '../visualizations/evaluation'
REPORT_DIR      = '../data/processed'


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_column_names(df):
    """Match the same cleaning done in model_training.py for XGBoost/LightGBM"""
    df = df.copy()
    df.columns = (df.columns
                  .str.replace(':', '_')
                  .str.replace('-', '_')
                  .str.replace(' ', '_'))
    return df


def load_test_data():
    """Load the held-out test set — never seen during training or tuning"""
    print("Loading test set...")
    df = pd.read_csv(TEST_DATA_PATH)
    X_test = df.drop('FraudFound_P', axis=1)
    y_test = df['FraudFound_P']
    X_test = clean_column_names(X_test)
    print(f"✓ Test set loaded: {X_test.shape}")
    print(f"  Fraud cases: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    return X_test, y_test


def load_model_and_threshold():
    """Load production model and its saved deployed threshold"""
    print("\nLoading production model and threshold...")
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, 'r') as f:
        threshold_data = json.load(f)

    deployed_threshold = threshold_data['deployed_threshold']
    f1_threshold       = threshold_data['f1_threshold']

    print(f"✓ Model loaded: {MODEL_PATH}")
    print(f"  Deployed threshold (business): {deployed_threshold:.4f}")
    print(f"  F1 threshold (reported):       {f1_threshold:.4f}")
    return model, threshold_data, deployed_threshold, f1_threshold


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classification_metrics(model, X_test, y_test,
                                    deployed_threshold, f1_threshold):
    """
    Compute all classification metrics on the test set.
    Reports both at the F1 threshold (for fair comparison)
    and at the deployed threshold (for business reporting).
    """
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS — TEST SET")
    print("="*60)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Predictions at both thresholds
    y_pred_f1       = (y_proba >= f1_threshold).astype(int)
    y_pred_deployed = (y_proba >= deployed_threshold).astype(int)

    # ── F1-threshold metrics (for fair model comparison) ──────
    acc_f1  = accuracy_score(y_test, y_pred_f1)
    prec_f1 = precision_score(y_test, y_pred_f1)
    rec_f1  = recall_score(y_test, y_pred_f1)
    f1_f1   = f1_score(y_test, y_pred_f1)

    # ── Deployed-threshold metrics (for business reporting) ───
    acc_dep  = accuracy_score(y_test, y_pred_deployed)
    prec_dep = precision_score(y_test, y_pred_deployed)
    rec_dep  = recall_score(y_test, y_pred_deployed)
    f1_dep   = f1_score(y_test, y_pred_deployed)

    # ── Threshold-independent ─────────────────────────────────
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)   # Precision-Recall AUC
    brier   = brier_score_loss(y_test, y_proba)          # Probability calibration quality

    print(f"\n{'Metric':<30} {'F1-Threshold':>14} {'Deployed':>14}")
    print(f"{'─'*60}")
    print(f"{'Threshold':<30} {f1_threshold:>14.4f} {deployed_threshold:>14.4f}")
    print(f"{'Accuracy':<30} {acc_f1:>14.4f} {acc_dep:>14.4f}")
    print(f"{'Precision':<30} {prec_f1:>14.4f} {prec_dep:>14.4f}")
    print(f"{'Recall':<30} {rec_f1:>14.4f} {rec_dep:>14.4f}")
    print(f"{'F1-Score':<30} {f1_f1:>14.4f} {f1_dep:>14.4f}")
    print(f"\n{'ROC-AUC (threshold-independent)':<30} {roc_auc:>14.4f}")
    print(f"{'PR-AUC (threshold-independent)':<30} {pr_auc:>14.4f}")
    print(f"{'Brier Score (lower=better)':<30} {brier:>14.4f}")

    print(f"\nClassification Report (F1 threshold = {f1_threshold:.4f}):")
    print(classification_report(y_test, y_pred_f1,
                                target_names=['Legitimate', 'Fraud']))

    print(f"\nClassification Report (Deployed threshold = {deployed_threshold:.4f}):")
    print(classification_report(y_test, y_pred_deployed,
                                target_names=['Legitimate', 'Fraud']))

    metrics = {
        # F1-threshold
        'accuracy_f1'  : acc_f1,
        'precision_f1' : prec_f1,
        'recall_f1'    : rec_f1,
        'f1_score_f1'  : f1_f1,
        # Deployed-threshold
        'accuracy_dep' : acc_dep,
        'precision_dep': prec_dep,
        'recall_dep'   : rec_dep,
        'f1_score_dep' : f1_dep,
        # Threshold-independent
        'roc_auc'      : roc_auc,
        'pr_auc'       : pr_auc,
        'brier_score'  : brier,
    }

    return metrics, y_proba, y_pred_f1, y_pred_deployed


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUSINESS METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_business_metrics(y_test, y_pred_deployed, y_pred_f1,
                               deployed_threshold, f1_threshold, y_proba):
    """
    Calculate real business cost on the test set.
    Also calculates the break-even threshold.
    """
    print("\n" + "="*60)
    print("BUSINESS METRICS — TEST SET")
    print("="*60)

    COST_MISSED_FRAUD  = 10_000   # Cost of missing 1 fraud ($)
    COST_FALSE_ALARM   = 200      # Cost of 1 false investigation ($)

    def calc_cost(y_true, y_pred, label):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        missed  = fn
        falarms = fp
        caught  = tp
        total   = tp + fn
        cost    = missed * COST_MISSED_FRAUD + falarms * COST_FALSE_ALARM
        print(f"\n  [{label}]")
        print(f"    Frauds caught:    {caught}/{total} ({caught/total*100:.1f}%)")
        print(f"    Missed frauds:    {missed}  → cost ${missed * COST_MISSED_FRAUD:,}")
        print(f"    False alarms:     {falarms} → cost ${falarms * COST_FALSE_ALARM:,}")
        print(f"    Total cost:       ${cost:,}")
        return cost, missed, falarms, caught, total

    # Default 0.50
    y_pred_default = (y_proba >= 0.50).astype(int)
    cost_default, *_ = calc_cost(y_test, y_pred_default, "Default 0.50")

    # F1-optimised
    cost_f1, *_ = calc_cost(y_test, y_pred_f1,
                             f"F1-Optimised ({f1_threshold:.4f})")

    # Deployed (business-optimal)
    cost_dep, missed_dep, falarms_dep, caught_dep, total_dep = calc_cost(
        y_test, y_pred_deployed, f"Deployed ({deployed_threshold:.4f})")

    print(f"\n  💰 Deployed vs Default saving: ${cost_default - cost_dep:,}")
    print(f"  💰 Deployed vs F1-opt saving:  ${cost_f1 - cost_dep:,}")

    # ── Break-even threshold ───────────────────────────────────
    # At what threshold does switching from default to lower become
    # financially neutral? Cost(missed fraud) / (Cost(missed) + Cost(alarm))
    breakeven = COST_MISSED_FRAUD / (COST_MISSED_FRAUD + COST_FALSE_ALARM)
    print(f"\n  Break-even recall threshold: {breakeven:.4f}")
    print(f"  Interpretation: Any threshold that improves recall above")
    print(f"  {breakeven*100:.1f}% is financially justified given these costs.")

    business_metrics = {
        'cost_default'        : cost_default,
        'cost_f1_threshold'   : cost_f1,
        'cost_deployed'       : cost_dep,
        'saving_vs_default'   : cost_default - cost_dep,
        'frauds_caught'       : caught_dep,
        'total_frauds'        : total_dep,
        'missed_frauds'       : missed_dep,
        'false_alarms'        : falarms_dep,
        'breakeven_threshold' : breakeven,
    }

    return business_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(y_test, y_pred_f1, y_pred_deployed,
                             f1_threshold, deployed_threshold):
    """Side-by-side confusion matrices — F1 threshold and deployed threshold"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrices — Test Set', fontsize=15, fontweight='bold')

    for ax, y_pred, thresh, title in [
        (axes[0], y_pred_f1,       f1_threshold,       f'F1-Optimised\n(threshold={f1_threshold:.4f})'),
        (axes[1], y_pred_deployed, deployed_threshold, f'Deployed (Business)\n(threshold={deployed_threshold:.4f})')
    ]:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

        # Annotate with fraud catch rate
        tn, fp, fn, tp = cm.ravel()
        ax.text(0.5, -0.15, f'Frauds caught: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)',
                ha='center', transform=ax.transAxes, fontsize=10, color='darkblue')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/confusion_matrices_test.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrices saved → {path}")


def plot_roc_curve(y_test, y_proba, roc_auc):
    """ROC curve with AUC annotation"""
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', linewidth=2.5,
             label=f'XGBoost (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve — Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/roc_curve_test.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved → {path}")


def plot_precision_recall_curve(y_test, y_proba, pr_auc,
                                 deployed_threshold, f1_threshold):
    """Precision-Recall curve with both threshold points marked"""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(9, 6))
    plt.plot(recalls, precisions, color='darkorange', linewidth=2.5,
             label=f'XGBoost (PR-AUC = {pr_auc:.4f})')
    plt.fill_between(recalls, precisions, alpha=0.1, color='darkorange')

    # Baseline (random classifier)
    baseline = y_test.mean()
    plt.axhline(y=baseline, color='gray', linestyle='--', linewidth=1,
                label=f'Random Classifier (precision={baseline:.3f})')

    # Mark deployed threshold point
    dep_idx = np.argmin(np.abs(thresholds - deployed_threshold))
    plt.scatter(recalls[dep_idx], precisions[dep_idx],
                color='red', s=120, zorder=5,
                label=f'Deployed threshold ({deployed_threshold:.4f})')

    # Mark F1 threshold point
    f1_idx = np.argmin(np.abs(thresholds - f1_threshold))
    plt.scatter(recalls[f1_idx], precisions[f1_idx],
                color='green', s=120, zorder=5,
                label=f'F1-optimal threshold ({f1_threshold:.4f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve — Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/precision_recall_curve_test.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curve saved → {path}")


def plot_feature_importance(model, feature_names, top_n=20):
    """Feature importance bar chart — top N features"""
    importances = pd.DataFrame({
        'feature'   : feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    colors = ['#c0392b' if i < 3 else '#2980b9'
              for i in range(len(importances))]
    bars = plt.barh(importances['feature'][::-1],
                    importances['importance'][::-1],
                    color=colors[::-1], edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, importances['importance'][::-1]):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9)

    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances — XGBoost', fontsize=14,
              fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/feature_importance_top{top_n}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance plot saved → {path}")


def plot_calibration_curve(y_test, y_proba):
    """
    Calibration curve — checks if predicted probabilities match actual rates.
    A perfectly calibrated model's curve sits on the diagonal.
    For fraud detection, slight over-confidence (curve below diagonal)
    is acceptable and common with scale_pos_weight.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10, strategy='uniform'
    )

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
    plt.plot(mean_predicted_value, fraction_of_positives,
             color='steelblue', marker='o', linewidth=2.5, markersize=6,
             label='XGBoost')
    plt.fill_between(mean_predicted_value, fraction_of_positives,
                     mean_predicted_value, alpha=0.1, color='steelblue')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives (Actual Fraud Rate)', fontsize=12)
    plt.title('Calibration Curve — Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/calibration_curve_test.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Calibration curve saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(model, X_test, y_test, y_proba, deployed_threshold):
    """
    Full SHAP analysis:
      - Summary plot (global feature importance across all test samples)
      - Bar plot (mean absolute SHAP — cleaner version of importance)
      - Waterfall plot (single prediction explained — highest-risk claim)
      - Saves top fraud indicators to CSV for the notebook
    """
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)

    print("\nCalculating SHAP values (this may take 1-2 minutes)...")

    # TreeExplainer is the correct explainer for XGBoost — fast and exact
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("✓ SHAP values calculated")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 4a. Summary Plot (beeswarm) ────────────────────────────
    # Each dot = one test sample
    # Colour = feature value (red=high, blue=low)
    # X-axis = how much that feature pushed the prediction toward fraud
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test,
                      max_display=20,
                      show=False,
                      plot_type='dot')
    plt.title('SHAP Summary Plot — Test Set\n(Impact of Each Feature on Fraud Prediction)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/shap_summary_plot.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP summary plot saved → {path}")

    # ── 4b. Bar Plot (mean |SHAP|) ─────────────────────────────
    # Cleaner than beeswarm — shows average magnitude of impact
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test,
                      max_display=20,
                      show=False,
                      plot_type='bar')
    plt.title('SHAP Feature Importance (Mean |SHAP Value|)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/shap_bar_plot.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP bar plot saved → {path}")

    # ── 4c. Waterfall Plot — highest-risk claim ────────────────
    # Pick the claim the model is MOST confident is fraud
    # This is the single most compelling visualisation for stakeholders
    highest_risk_idx = int(np.argmax(y_proba))
    actual_label     = int(y_test.iloc[highest_risk_idx])
    predicted_prob   = float(y_proba[highest_risk_idx])
    predicted_label  = int(predicted_prob >= deployed_threshold)

    print(f"\nWaterfall plot — highest-risk claim (index {highest_risk_idx}):")
    print(f"  Predicted fraud probability: {predicted_prob:.4f}")
    print(f"  Predicted label (deployed):  {'FRAUD' if predicted_label else 'LEGITIMATE'}")
    print(f"  Actual label:                {'FRAUD' if actual_label else 'LEGITIMATE'}")

    # Build Explanation object for waterfall
    explanation = shap.Explanation(
        values    = shap_values[highest_risk_idx],
        base_values = explainer.expected_value,
        data      = X_test.iloc[highest_risk_idx].values,
        feature_names = X_test.columns.tolist()
    )

    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(
        f'SHAP Waterfall — Highest-Risk Claim\n'
        f'Predicted: {predicted_prob:.4f} | '
        f'Actual: {"FRAUD ✓" if actual_label else "LEGITIMATE ✗"}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/shap_waterfall_highest_risk.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP waterfall plot saved → {path}")

    # ── 4d. Document top fraud indicators ─────────────────────
    mean_shap = pd.DataFrame({
        'feature'         : X_test.columns,
        'mean_abs_shap'   : np.abs(shap_values).mean(axis=0),
        'mean_shap'       : shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    print(f"\n=== TOP 10 FRAUD INDICATORS (SHAP) ===")
    print(mean_shap.head(10).to_string(index=False))

    shap_path = f'{REPORT_DIR}/shap_feature_importance.csv'
    mean_shap.to_csv(shap_path, index=False)
    print(f"\n✓ SHAP feature importance saved → {shap_path}")

    return shap_values, explainer, mean_shap


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_evaluation_report(metrics, business_metrics, threshold_data):
    """Save a complete JSON evaluation report for the notebook to reference"""
    report = {
        'dataset'              : 'test set — never seen during training or tuning',
        'model'                : 'xgboost_v1.pkl',

        # Thresholds
        'f1_threshold'         : threshold_data['f1_threshold'],
        'deployed_threshold'   : threshold_data['deployed_threshold'],

        # Classification metrics at F1 threshold
        'accuracy_f1'          : metrics['accuracy_f1'],
        'precision_f1'         : metrics['precision_f1'],
        'recall_f1'            : metrics['recall_f1'],
        'f1_score_f1'          : metrics['f1_score_f1'],

        # Classification metrics at deployed threshold
        'accuracy_deployed'    : metrics['accuracy_dep'],
        'precision_deployed'   : metrics['precision_dep'],
        'recall_deployed'      : metrics['recall_dep'],
        'f1_score_deployed'    : metrics['f1_score_dep'],

        # Threshold-independent
        'roc_auc'              : metrics['roc_auc'],
        'pr_auc'               : metrics['pr_auc'],
        'brier_score'          : metrics['brier_score'],

        # Business
        'frauds_caught'        : business_metrics['frauds_caught'],
        'total_frauds'         : business_metrics['total_frauds'],
        'fraud_detection_rate' : business_metrics['frauds_caught'] / business_metrics['total_frauds'],
        'missed_frauds'        : business_metrics['missed_frauds'],
        'false_alarms'         : business_metrics['false_alarms'],
        'business_cost'        : business_metrics['cost_deployed'],
        'cost_saving_vs_default': business_metrics['saving_vs_default'],
        'breakeven_threshold'  : business_metrics['breakeven_threshold'],
    }

    # Convert numpy types to native Python for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    report_serialisable = {k: convert(v) for k, v in report.items()}

    path = f'{REPORT_DIR}/test_evaluation_report.json'
    with open(path, 'w') as f:
        json.dump(report_serialisable, f, indent=2)
    print(f"\n✓ Evaluation report saved → {path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────────

def print_model_limitations():
    """
    Document known limitations — important for any honest portfolio project
    and required for a proper model card (Day 14).
    """
    print("\n" + "="*60)
    print("MODEL LIMITATIONS & CONSIDERATIONS")
    print("="*60)
    print("""
1. DATASET SIZE
   Only 15,420 claims total — a real insurer processes millions.
   Performance on a larger, more diverse dataset may differ.

2. CLASS IMBALANCE CEILING
   At 5.97% fraud rate, F1 ~0.26 is near the theoretical ceiling
   for this feature set and dataset size. More data or richer
   features (e.g. network graphs of claimants) would be needed
   to push meaningfully higher.

3. TEMPORAL DRIFT
   The model was trained on historical claims with no date-based
   split. In production, fraud patterns evolve — the model should
   be retrained on a rolling window (e.g. every 6 months).

4. COST ASSUMPTIONS
   Business costs ($10,000 missed fraud / $200 false alarm) are
   assumed. Real deployment requires actuary-validated figures —
   the threshold should be recalibrated when actual costs are known.

5. FEATURE AVAILABILITY
   Some engineered features (e.g. external_agent_holder_fault)
   may not be immediately available at claim submission time.
   A real-time scoring pipeline would need to handle missing
   features gracefully.

6. EXPLAINABILITY
   SHAP values explain individual predictions but require
   data scientist interpretation. For regulatory use (e.g. GDPR,
   insurance regulation), formal reason codes and audit trails
   would be required.
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation():
    """
    Complete Day 13 evaluation pipeline.
    Run from ml_pipeline/ directory: python model_evaluation.py
    """
    print("=" * 60)
    print("DAY 13: MODEL EVALUATION — TEST SET")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────
    X_test, y_test = load_test_data()
    model, threshold_data, deployed_threshold, f1_threshold = load_model_and_threshold()

    # ── 1. Classification metrics ─────────────────────────────
    metrics, y_proba, y_pred_f1, y_pred_deployed = evaluate_classification_metrics(
        model, X_test, y_test, deployed_threshold, f1_threshold
    )

    # ── 2. Business metrics ───────────────────────────────────
    business_metrics = evaluate_business_metrics(
        y_test, y_pred_deployed, y_pred_f1,
        deployed_threshold, f1_threshold, y_proba
    )

    # ── 3. Visualisations ─────────────────────────────────────
    print("\n" + "="*60)
    print("GENERATING VISUALISATIONS")
    print("="*60)

    plot_confusion_matrices(y_test, y_pred_f1, y_pred_deployed,
                             f1_threshold, deployed_threshold)
    plot_roc_curve(y_test, y_proba, metrics['roc_auc'])
    plot_precision_recall_curve(y_test, y_proba, metrics['pr_auc'],
                                 deployed_threshold, f1_threshold)
    plot_feature_importance(model, X_test.columns, top_n=20)
    plot_calibration_curve(y_test, y_proba)

    # ── 4. SHAP ───────────────────────────────────────────────
    shap_values, explainer, shap_importance = run_shap_analysis(
        model, X_test, y_test, y_proba, deployed_threshold
    )

    # ── 5. Save report ────────────────────────────────────────
    report = save_evaluation_report(metrics, business_metrics, threshold_data)

    # ── 6. Limitations ────────────────────────────────────────
    print_model_limitations()

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("DAY 13 COMPLETE")
    print("="*60)
    print(f"\n TEST SET FINAL RESULTS:")
    print(f"  ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:               {metrics['pr_auc']:.4f}")
    print(f"  F1-Score (reported):  {metrics['f1_score_f1']:.4f}  (at threshold {f1_threshold:.4f})")
    print(f"  Frauds caught:        {business_metrics['frauds_caught']}/{business_metrics['total_frauds']} "
          f"({business_metrics['frauds_caught']/business_metrics['total_frauds']*100:.1f}%)")
    print(f"  Business saving:      ${business_metrics['saving_vs_default']:,} vs default 0.50")
    print(f"\n Outputs saved to:")
    print(f"  Visualisations: {OUTPUT_DIR}/")
    print(f"  Report JSON:    {REPORT_DIR}/test_evaluation_report.json")
    print(f"  SHAP CSV:       {REPORT_DIR}/shap_feature_importance.csv")
    print(f"\n Next: Build notebooks/03_model_evaluation.ipynb")

    return {
        'metrics'         : metrics,
        'business_metrics': business_metrics,
        'shap_values'     : shap_values,
        'shap_importance' : shap_importance,
        'report'          : report
    }


if __name__ == "__main__":
    run_full_evaluation()