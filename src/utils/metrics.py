from typing import Sequence, Dict, Any
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, Any]:
    """
    Compute accuracy and macro F1-score for classification.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with 'accuracy' and 'macro_f1'.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}
