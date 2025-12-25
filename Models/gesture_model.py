import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class Hand_Gesture_Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.f1 = nn.Linear(21 * 2, 22)
        self.f2 = nn.Linear(22, 11)
        self.f3 = nn.Linear(11, num_classes)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.f2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.f3(x)
        return x


def load_labels(label_path: Path) -> list[str]:
    label_path = Path(label_path)
    # Use utf-8-sig to automatically strip BOM if present
    labels = [line.strip() for line in label_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {label_path}")
    return labels


def train_evalute(
    data_path: str | Path = "../keypoint.csv",
    label_path: str | Path = "../keypoint_classifier_label.csv",
    epochs: int = 200,
    patience: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
):
    data_path = Path(data_path)
    label_path = Path(label_path)

    labels = load_labels(label_path)
    num_classes = len(labels)

    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hand_Gesture_Model(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, train_loss = 0, 0, 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_total, val_correct, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:02d}: Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "labels": labels,
            }
            torch.save(checkpoint, "best_model.pth")
            no_improve = 0
            print("Model improved and saved!")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Evaluate best model
    checkpoint = torch.load("best_model.pth", map_location=device)
    best_model = Hand_Gesture_Model(checkpoint["num_classes"]).to(device)
    best_model.load_state_dict(checkpoint["state_dict"])
    best_model.eval()

    X_test_t = X_test_t.to(device)
    y_test_t = y_test_t.to(device)
    with torch.no_grad():
        outputs = best_model(X_test_t)
        preds = torch.argmax(outputs, dim=1)

    y_true = y_test_t.cpu().numpy()
    y_pred = preds.cpu().numpy()
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    cmx_data = confusion_matrix(y_true, y_pred)
    print(cmx_data)


if __name__ == "__main__":
    train_evalute()
