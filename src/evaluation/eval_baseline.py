from pathlib import Path

import pandas as pd
import torch

from src.data.datamodule_tfidf import DataConfig, create_dataloaders
from src.models.baseline_traditional import MLPClassifier
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed


def eval_main(
    checkpoint_path: str = "results/checkpoints/best_baseline_mlp.pt",
) -> None:
    data_cfg = DataConfig(
        dataset_path="src/data/hate_dataset.csv",
        batch_size=64,
        max_features=30000,
        ngram_min=1,
        ngram_max=2,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    set_seed(42)

    _, _, test_loader, vectorizer, label2id = create_dataloaders(data_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = vectorizer.max_features or len(vectorizer.vocabulary_)
    num_classes = len(label2id)

    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_preds)
    print("Test metrics:", metrics)

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "split": "test",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        ]
    )
    df.to_csv(metrics_dir / "baseline_mlp_metrics_eval.csv", index=False)


if __name__ == "__main__":
    eval_main()
