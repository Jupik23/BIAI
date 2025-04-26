from torchvision import transforms
from PIL import Image
import torch

def predict_image(model, image_path, device, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    return class_names[preds.item()]
