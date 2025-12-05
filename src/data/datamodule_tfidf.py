from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    dataset_path: str = "src/data/hate_dataset.csv"  # <-- тут твой текущий путь
    text_column: str = "text"
    label_column: str = "label"
    val_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42
    max_features: int = 30000
    ngram_min: int = 1
    ngram_max: int = 2
    batch_size: int = 64


class TfidfDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _encode_labels(labels: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Map string or numeric labels to consecutive integers starting from 0.
    """
    if labels.dtype == object:
        unique = sorted(labels.unique())
        label2id = {lab: i for i, lab in enumerate(unique)}
        encoded = labels.map(label2id).to_numpy()
    else:
        unique = sorted(labels.unique())
        label2id = {str(lab): int(lab) for lab in unique}
        encoded = labels.to_numpy()

    return encoded, label2id


def create_dataloaders(
    cfg: DataConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, TfidfVectorizer, Dict[str, int]]:
    """
    Create train/val/test dataloaders and TF-IDF vectorizer.

    Returns:
        train_loader, val_loader, test_loader, vectorizer, label2id
    """
    path = Path(cfg.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    df = df.dropna(subset=[cfg.text_column, cfg.label_column])

    labels_encoded, label2id = _encode_labels(df[cfg.label_column])

    test_size = cfg.test_size
    val_size = cfg.val_size / (1.0 - test_size)

    df_train, df_temp, y_train, y_temp = train_test_split(
        df[cfg.text_column],
        labels_encoded,
        test_size=test_size,
        random_state=cfg.random_state,
        stratify=labels_encoded,
    )

    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        test_size=val_size,
        random_state=cfg.random_state,
        stratify=y_temp,
    )

    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(df_train).toarray()
    X_val = vectorizer.transform(df_val).toarray()
    X_test = vectorizer.transform(df_test).toarray()

    train_ds = TfidfDataset(X_train, y_train)
    val_ds = TfidfDataset(X_val, y_val)
    test_ds = TfidfDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(train_ds.__class__(X_val, y_val), batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    return train_loader, val_loader, test_loader, vectorizer, label2id
