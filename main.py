import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_dataloaders
from models.resnet_transfer import get_model
from utils.plots import plot_training

def train(version, num_epochs, dropout_p, learning_rate, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders()

    model = get_model(version=version, dropout_p=dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 1
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at '{checkpoint_path}', resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"  → Resumed from epoch {checkpoint['epoch']} (val_loss = {best_val_loss:.4f})")
    else:
        print("No checkpoint found. Starting from epoch 1.")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch [{epoch}/{start_epoch + num_epochs - 1}]"
              f" Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}"
              f" Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"  → New best model at epoch {epoch}, checkpoint saved.")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    torch.save(model.state_dict(), 'model_final.pth')
    print("Final model saved to 'model_final.pth'.")
    plot_training(history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pneumonia detection model (5 strategies)')
    parser.add_argument('--version', type=int, required=True, choices=range(1, 6),
                        help='Model version: 1=Pretrained ResNet18, 2=Untrained ResNet18, 3=ResNet18+Dropout, 4=ResNet18+2xDropout, 5=SimpleCNN')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default='best_checkpoint.pth', help='Path for checkpoint file')
    args = parser.parse_args()

    train(
        version=args.version,
        num_epochs=args.epochs,
        dropout_p=args.dropout,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint
    )
