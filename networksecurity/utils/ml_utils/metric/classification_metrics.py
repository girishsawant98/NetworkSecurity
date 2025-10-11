from networksecurity.logging.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from networksecurity.exception.exceptions import NetworkSecurityException
import sys  
import os
import numpy as np
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics: accuracy, precision, recall, and F1-score.

    Args:
        y_true : True labels.
        y_pred : Predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    try:
        model_precision_score = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        model_recall_score = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        model_f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        classification_metric = ClassificationMetricArtifact(
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            f1_score=model_f1_score
        )

        return classification_metric
    except Exception as e:
        raise NetworkSecurityException(e, sys)