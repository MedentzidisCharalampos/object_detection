import torch
import torchvision.transforms as transforms
from PIL import Image
import io

class ImageClassifier:
    def __init__(self, model_path):
        """
        Load a PyTorch image classification model.
        """
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.labels = [str(i) for i in range(1000)]  # Replace with actual label list if available

    def classify(self, img_bytes):
        """
        Run classification on the given image bytes.

        Args:
            img_bytes (bytes): Input image file content

        Returns:
            str: Predicted label
        """
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = outputs.max(1)
        return self.labels[predicted.item()]
