import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix

# Global variables for training
epochs = 200
patience = 5
best_val_acc = 0.0
no_improve = 0

class Hand_Gesture_Model(nn.Module):
    def __init__(self):
        super(Hand_Gesture_Model, self).__init__()
        self.f1=nn.Linear(21*2,22)
        self.f2=nn.Linear(22,11)
        self.f3=nn.Linear(11,4)
    def forward(self,x):
        x=F.relu(self.f1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x=F.relu(self.f2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.f3(x)
        return x

def train_evalute():
    global best_val_acc, no_improve
    # Load data
    df = pd.read_csv("../keypoint.csv", header=None)
    df = pd.read_csv("../keypoint.csv",header=None)
    df.head()
    df_label = df_label = pd.read_csv("../keypoint_classifier_label.csv", header=None)
    df_label

    X=df.iloc[:, 1:]
    y=df.iloc[:, 0]
    y.value_counts()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)
    y_train.value_counts()
    y_test.value_counts()

    y_train=torch.LongTensor(y_train.to_numpy())
    y_test=torch.LongTensor(y_test.to_numpy())
    X_val = torch.FloatTensor(X_val.to_numpy())
    y_val = torch.LongTensor(y_val.to_numpy())
    X_train=torch.FloatTensor(X_train.to_numpy())
    X_test=torch.FloatTensor(X_test.to_numpy())
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)
    val_dataset=TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=128)
    val_loader=DataLoader(val_dataset, batch_size=128)
    
    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hand_Gesture_Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
        total, correct, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)  # Soft max probs are computed and loss is computed
                val_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= total
        val_acc = correct / total

        print(f"Epoch {epoch:02d}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.3f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model , "best_model.pth")
            no_improve = 0
            print("Model improved and saved!")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(" Early stopping triggered.")
                break
    model = torch.load("best_model.pth", map_location=device, weights_only=False)
    model.eval()
    X_test_t = torch.tensor(X_test ,dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(X_test_t)
        preds = torch.argmax(outputs, dim=1)
    y_true = y_test_t.cpu().numpy()
    y_pred = preds.cpu().numpy()
    print(classification_report(y_true, y_pred))

    print("Accuracy:", accuracy_score(y_true, y_pred))

    cmx_data = confusion_matrix(y_true, y_pred)

    print(cmx_data)
if __name__ == "__main__":
    train_evalute()
