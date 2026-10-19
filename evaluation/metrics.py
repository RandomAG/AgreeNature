"""
Evaluation utilities for detection and operational threat assessment.

Includes standard detection metrics (Precision, Recall, F1) alongside the
system-level operational metrics:
    - Actionable Alerts (%)               
    - Alert Prioritization Accuracy (%)   
    - Critical Threat Identification Rate 
"""

import numpy as np


def precision(true_positive, false_positive):
    denominator = true_positive + false_positive
    if denominator == 0:
        return 0.0
    return true_positive / denominator


def recall(true_positive, false_negative):
    denominator = true_positive + false_negative
    if denominator == 0:
        return 0.0
    return true_positive / denominator


def f1_score(precision_value, recall_value):
    denominator = precision_value + recall_value
    if denominator == 0:
        return 0.0
    return (2 * precision_value * recall_value) / denominator


def summarize_detection_metrics(true_positive, false_positive, false_negative):
    """Computing basic detection metrics."""
    p = precision(true_positive, false_positive)
    r = recall(true_positive, false_negative)
    f1 = f1_score(p, r)

    return {
        "precision": p,
        "recall": r,
        "f1_score": f1
    }


# System-Level Operational Metrics

def actionable_alerts_rate(n_valid_true_threats: int, n_total_alerts: int) -> float:
    """
    Compute Actionable Alerts (%)
    
    Measures the ratio of security alerts corresponding to confirmed, persistent
    threat incidents rather than transient false alarms.
    """
    if n_total_alerts == 0:
        return 0.0
    return (n_valid_true_threats / n_total_alerts) * 100.0


def alert_prioritization_accuracy(predicted_levels: list, ground_truth_levels: list) -> float:
    """
    Compute Alert Prioritization Accuracy (%) 
    
    Evaluates the exact match rate between the model's assigned threat level
    and the expert ground-truth risk level.
    """
    if not ground_truth_levels:
        return 0.0
    matches = sum(
        1 for p, gt in zip(predicted_levels, ground_truth_levels)
        if p.strip().lower() == gt.strip().lower()
    )
    return (matches / len(ground_truth_levels)) * 100.0


def critical_threat_identification_rate(n_detected_critical: int, n_total_ground_truth_critical: int) -> float:
    """
    Compute Critical Threat Identification Rate (%)
    
    Quantifies the recall rate specifically for high-severity lethal threats (Rifles).
    """
    if n_total_ground_truth_critical == 0:
        return 0.0
    return (n_detected_critical / n_total_ground_truth_critical) * 100.0