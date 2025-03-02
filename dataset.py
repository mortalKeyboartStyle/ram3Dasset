import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),  # Dla uproszczenia – można dostosować dla fragmentów 4K
            transforms.ToTensor()
        ])
        self.image = Image.open(self.image_path).convert("RGB")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.transform(self.image)
