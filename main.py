import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from data.dataloader import get_dataloaders
from models.resnet_pretrained import get_model
from utils.plots import plot_training

def train(resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    num_epochs = 10
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 0

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        best_epoch = checkpoint.get('epoch', 0)
        start_epoch = best_epoch
        print(f"Wczytano checkpoint z '{resume_path}' (epoka {best_epoch}, val_loss = {best_val_loss:.4f}). Kontynuuję od epoki {start_epoch+1}.")

    for epoch in range(start_epoch, num_epochs):
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

        current_epoch = epoch + 1
        print(f"Epoch [{current_epoch}/{num_epochs}]"
              f" Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}"
              f" Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = current_epoch
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_checkpoint.pth')
            print(f"  → New best model (epoka {best_epoch}), checkpoint saved.")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    checkpoint = torch.load('best_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded best model from epoki {checkpoint['epoch']} "
          f"(val_loss = {checkpoint['val_loss']:.4f})")

    torch.save(model.state_dict(), 'model.pth')

    plot_training(history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trening modelu z możliwością wznowienia z checkpointu")
    parser.add_argument('--resume', type=str, default=None,
                        help="Ścieżka do pliku checkpointu, aby wznowić trening (opcjonalne)")
    args = parser.parse_args()
    train(resume_path=args.resume)
