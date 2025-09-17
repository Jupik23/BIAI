import matplotlib.pyplot as plt

def plot_training(history, save_path=None):
    epochs = range(len(history['train_loss']))
    
    plt.figure(figsize=(10,5))
    
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.title('Training History')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
