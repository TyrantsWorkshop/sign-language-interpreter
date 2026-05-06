"""
Training script for Sign Language Model
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging

from models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, num_epochs=50,
                learning_rate=1e-3, device='cuda'):
    """
    Train sign language model

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += y_batch.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item()
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += y_batch.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_sign_language_model.pt')
            logger.info(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")

    return model


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model on the test set."""
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            test_loss += loss.item()
            test_correct += (logits.argmax(1) == y_batch).sum().item()
            test_total += y_batch.size(0)

    test_acc = test_correct / test_total
    logger.info(f"Test Loss: {test_loss/len(test_loader):.4f}, Acc: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sign language model")
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--model-type', type=str, default='vit-conv',
                       choices=['vit', 'vit-conv', 'multi-scale'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default='best_sign_language_model.pt')
    parser.add_argument('--eval-only', action='store_true', help="Skip training and only evaluate")

    args = parser.parse_args()

    # Load data
    X = np.load(f"{args.dataset}/X_keypoints.npy")
    y = np.load(f"{args.dataset}/y_labels.npy")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    model = create_model(
        model_type=args.model_type,
        num_classes=len(np.unique(y))
    )

    if args.eval_only:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        # Train
        train_model(model, train_loader, val_loader,
                   num_epochs=args.epochs,
                   learning_rate=args.learning_rate,
                   device=args.device)

    # Evaluate
    evaluate_model(model, test_loader, device=args.device)
