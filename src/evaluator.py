"""
Evaluation Module
Compares LLM-extracted severity against ground truth labels.
Produces classification metrics and a confusion matrix.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, cohen_kappa_score
)
from typing import Optional
import json


SEVERITY_ORDER = ["Low", "Medium", "High"]


def compute_metrics(df: pd.DataFrame,
                    true_col: str = "true_severity",
                    pred_col: str = "llm_severity") -> dict:
    """Compute all evaluation metrics."""
    y_true = df[true_col].values
    y_pred = df[pred_col].fillna("Low").values

    # Ordinal accuracy (within-1)
    def within_one(yt, yp):
        ti = SEVERITY_ORDER.index(yt) if yt in SEVERITY_ORDER else 1
        pi = SEVERITY_ORDER.index(yp) if yp in SEVERITY_ORDER else 1
        return abs(ti - pi) <= 1

    exact_acc = accuracy_score(y_true, y_pred)
    relaxed_acc = np.mean([within_one(t, p) for t, p in zip(y_true, y_pred)])
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic",
                              labels=SEVERITY_ORDER)

    report = classification_report(y_true, y_pred, labels=SEVERITY_ORDER,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=SEVERITY_ORDER).tolist()

    # Confidence calibration
    if "llm_confidence" in df.columns:
        mean_conf = df["llm_confidence"].mean()
        high_conf_mask = df["llm_confidence"] >= 0.8
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(
                y_true[high_conf_mask], y_pred[high_conf_mask]
            )
        else:
            high_conf_acc = None
    else:
        mean_conf = None
        high_conf_acc = None

    return {
        "exact_accuracy": round(exact_acc, 4),
        "relaxed_accuracy": round(relaxed_acc, 4),
        "quadratic_kappa": round(kappa, 4),
        "per_class": report,
        "confusion_matrix": cm,
        "labels": SEVERITY_ORDER,
        "mean_confidence": round(mean_conf, 4) if mean_conf else None,
        "high_confidence_accuracy": round(high_conf_acc, 4) if high_conf_acc else None,
        "n_samples": len(df),
        "n_high_conf": int(high_conf_mask.sum()) if "llm_confidence" in df.columns else None,
    }


def fault_type_distribution(df: pd.DataFrame) -> dict:
    """Distribution of extracted fault types."""
    if "llm_fault_type" not in df.columns:
        return {}
    counts = df["llm_fault_type"].value_counts().to_dict()
    return counts


def action_required_rate(df: pd.DataFrame) -> float:
    """% of records flagged action_required."""
    if "llm_action_required" not in df.columns:
        return 0.0
    return round(df["llm_action_required"].mean() * 100, 1)


def urgency_stats(df: pd.DataFrame) -> dict:
    """Descriptive stats on urgency_days."""
    if "llm_urgency_days" not in df.columns:
        return {}
    s = df["llm_urgency_days"].dropna()
    if s.empty:
        return {}
    return {
        "mean": round(s.mean(), 1),
        "median": round(s.median(), 1),
        "min": int(s.min()),
        "max": int(s.max()),
        "n_with_urgency": len(s)
    }


def run_evaluation(csv_path: str,
                   true_col: str = "true_severity",
                   pred_col: str = "llm_severity") -> None:
    df = pd.read_csv(csv_path)
    print(f"\n=== Evaluation on {len(df)} records ===\n")

    metrics = compute_metrics(df, true_col, pred_col)
    print(f"Exact Accuracy      : {metrics['exact_accuracy']:.2%}")
    print(f"Relaxed Accuracy    : {metrics['relaxed_accuracy']:.2%}")
    print(f"Quadratic κ (Kappa) : {metrics['quadratic_kappa']:.4f}")
    if metrics["mean_confidence"]:
        print(f"Mean LLM Confidence : {metrics['mean_confidence']:.2%}")
    if metrics["high_confidence_accuracy"]:
        print(f"High-Conf Accuracy  : {metrics['high_confidence_accuracy']:.2%}  (n={metrics['n_high_conf']})")

    print("\n--- Per-class Report ---")
    for cls in SEVERITY_ORDER:
        r = metrics["per_class"].get(cls, {})
        print(f"  {cls:<8} P={r.get('precision', 0):.2f} R={r.get('recall', 0):.2f} F1={r.get('f1-score', 0):.2f}")

    print("\n--- Confusion Matrix (rows=true, cols=pred) ---")
    print(f"        {'  '.join(SEVERITY_ORDER)}")
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {SEVERITY_ORDER[i]:<8} {row}")

    print(f"\n--- Fault Type Distribution ---")
    for ft, cnt in sorted(fault_type_distribution(df).items(), key=lambda x: -x[1]):
        print(f"  {ft:<12} {cnt}")

    print(f"\nAction Required Rate : {action_required_rate(df)}%")

    ustat = urgency_stats(df)
    if ustat:
        print(f"Urgency Days (mean/median) : {ustat['mean']} / {ustat['median']}")

    return metrics


if __name__ == "__main__":
    run_evaluation("data/enriched_sample.csv")
