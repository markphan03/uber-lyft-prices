import os
import torch
import torch.nn as nn
import numpy as np
from nn_model import CabPriceModel


# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# TENSOR PREPARATION (TRAIN / TEST SPLIT)
# =========================================================
def to_tensor(X, y, device):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    return X_t, y_t


def load_data(path="train_test_split.npz"):
    data = np.load(path, allow_pickle=True)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()


X_train_t, y_train_t = to_tensor(X_train, y_train, device)
X_test_t, y_test_t = to_tensor(X_test, y_test, device)


# =========================================================
# MODEL + OPTIMIZER
# =========================================================
model = CabPriceModel(X_train.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================================================
# CHECKPOINT
# =========================================================
checkpoint_path = "nn_checkpoint.pth"

start_epoch = 0
best_loss = float("inf")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("loss", float("inf"))

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


# =========================================================
# TRAINING LOOP (WITH TEST EVAL)
# =========================================================
def evaluate(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        loss = criterion(preds, y_test)
    return loss.item()


num_epochs = 800

for epoch in range(start_epoch, num_epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)

    loss.backward()
    optimizer.step()

    # -------------------------
    # evaluation on test set
    # -------------------------
    test_loss = evaluate(model, X_test_t, y_test_t, criterion)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Test Loss: {test_loss:.4f}"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item()
        }, checkpoint_path)


# =========================================================
# FINAL SAVE
# =========================================================
torch.save({
    "epoch": num_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item()
}, checkpoint_path)

print("Training complete and checkpoint saved.")