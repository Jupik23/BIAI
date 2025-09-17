import torch
import os
from utils.plots import plot_training
from tqdm import tqdm

class Train:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, device, num_epochs, model_name):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.model_name = model_name
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        os.makedirs('weights', exist_ok=True)
        self.checkpoint_path = f'weights/{self.model_name}_best_checkpoint.pth'

    def _train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0 , 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch{self.start_epoch+1}")
        for images, labels in progress_bar: 
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=running_loss/total, acc=correct/total)

        return running_loss / total, correct / total

    def _validate_epoch(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.start_epoch + 1}/{self.num_epochs} [Validation]")
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix(loss=running_loss/total, acc=correct/total)
                
        return running_loss / total, correct / total

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  -> New best model found! Saving checkpoint to {self.checkpoint_path}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': self.best_val_loss,
                }, self.checkpoint_path)

        print("\nTraining finished!")
        plot_training(self.history)