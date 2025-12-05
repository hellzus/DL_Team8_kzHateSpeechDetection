from src.utils.metrics import compute_classification_metrics


def test_compute_classification_metrics_perfect():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
