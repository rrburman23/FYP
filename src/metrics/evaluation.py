# metrics/evaluation.py

from typing import List, Dict
import numpy as np

# Helper function to compute Precision, Recall, F1-Score
def compute_detection_metrics(true_positive: int, false_positive: int, false_negative: int) -> Dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# Compute Object Tracking Metrics: MOTA, IDE, FP, FN
def compute_tracking_metrics(true_tracks: List[int], predicted_tracks: List[int]) -> Dict[str, float]:
    # Assuming both lists are of the same length and contain IDs for each tracked object
    tp = np.sum(np.array(true_tracks) == np.array(predicted_tracks))  # True Positives
    fp = np.sum(np.array(true_tracks) != np.array(predicted_tracks))  # False Positives
    fn = len(true_tracks) - tp  # False Negatives

    # MOTA: Multiple Object Tracking Accuracy
    mota = 1 - (fp + fn) / len(true_tracks) if len(true_tracks) > 0 else 0.0

    # IDE: Identity Switch Error (assuming each change in identity is counted as an error)
    ide = np.sum(np.array(true_tracks) != np.array(predicted_tracks))

    return {
        "mota": mota,
        "ide": ide,
        "fp": fp,
        "fn": fn
    }

# Detection accuracy: True positives / total number of objects in ground truth
def compute_detection_accuracy(true_objects: List[int], predicted_objects: List[int]) -> float:
    true_positives = sum([1 for obj in predicted_objects if obj in true_objects])
    accuracy = true_positives / len(true_objects) if len(true_objects) > 0 else 0.0
    return accuracy
