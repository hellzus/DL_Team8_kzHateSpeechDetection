from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datamodule_tfidf import DataConfig, create_dataloaders
from src.models.baseline_traditional import MLPClassifier
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(loader.dataset)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    return compute_classification_metrics(all_labels, all_preds)


def train_main() -> None:
    # Можно позже вынести в configs/baseline.yaml, пока захардкожено.
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

    num_epochs = 50
    learning_rate = 1e-3
    seed = 42

    set_seed(seed)

    (
        train_loader,
        val_loader,
        test_loader,
        vectorizer,
        label2id,
    ) = create_dataloaders(data_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = vectorizer.max_features or len(vectorizer.vocabulary_)
    num_classes = len(label2id)

    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = 0.0
    results_dir = Path("results")
    ckpt_dir = results_dir / "checkpoints"
    logs_dir = results_dir / "logs"
    metrics_dir = results_dir / "metrics"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, device)
        msg = (
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss={train_loss:.4f} "
            f"- val_acc={val_metrics['accuracy']:.4f} "
            f"- val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        print(msg)
        log_lines.append(msg)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                model.state_dict(),
                ckpt_dir / "best_baseline_mlp.pt",
            )

    (logs_dir / "train_baseline.log").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )

    # Финальная оценка на тесте
    model.load_state_dict(torch.load(ckpt_dir / "best_baseline_mlp.pt", map_location=device))
    test_metrics = evaluate_epoch(model, test_loader, device)
    print("Test metrics:", test_metrics)

    df_metrics = pd.DataFrame(
        [
            {
                "split": "test",
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
            }
        ]
    )
    df_metrics.to_csv(metrics_dir / "baseline_mlp_metrics.csv", index=False)


if __name__ == "__main__":
    train_main()
