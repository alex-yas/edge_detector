import os

import cv2
from torch.utils.data import Dataset
from torchvision import transforms



class EdgeDetectionDataset(Dataset):
    """Custom dataset class for loading images. Directory must contain source
        and target subdirectories
    """
    def __init__(self, path):
        self.source_dir = os.path.join(path, "source")
        self.target_dir = os.path.join(path, "target")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.image_files = os.listdir(self.source_dir)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.source_dir, self.image_files[idx])
        target_path = os.path.join(self.target_dir, self.image_files[idx])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        edge = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image = self.transform(image)
            edge = self.transform(edge)
        
        return image, edge
    