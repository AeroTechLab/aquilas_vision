import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class EfficientDummyDataset(Dataset):
    """Dataset dummy eficiente para testes rápidos"""
    
    def __init__(self, num_samples=100, img_size=224, num_classes=1000, channels=3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.channels = channels
        
        # Pré-computa algumas imagens para ser mais rápido
        self.precomputed_images = []
        self.precomputed_targets = []
        
        # Gera dados uma vez e reutiliza
        for i in range(min(100, num_samples)):  # Máximo 100 imagens pré-computadas
            img_array = np.random.randint(0, 255, (img_size, img_size, channels), dtype=np.uint8)
            self.precomputed_images.append(img_array)
            self.precomputed_targets.append(np.random.randint(0, num_classes))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Reusa imagens pré-computadas ou gera novas se necessário
        if idx < len(self.precomputed_images):
            image_array = self.precomputed_images[idx]
            target = self.precomputed_targets[idx]
        else:
            image_array = np.random.randint(0, 255, (self.img_size, self.img_size, self.channels), dtype=np.uint8)
            target = np.random.randint(0, self.num_classes)
        
        image = Image.fromarray(image_array)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

class DummyCocoDataLoader:
    """DataLoader dummy rápido para COCO"""
    
    def __init__(self, config):
        self.config = config
    
    def get_calibration_loader(self, split='val', num_samples=None):
        if num_samples is None:
            num_samples = self.config.BATCH_SIZE * getattr(self.config, 'NUM_CALIBRATION_BATCHES', 2)
        
        dataset = EfficientDummyDataset(
            num_samples=num_samples,
            img_size=self.config.IMG_SIZE,
            num_classes=getattr(self.config, 'NUM_CLASSES', 1000)
        )
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0  # 0 workers para ser mais rápido
        )
    
    def get_train_loader(self, batch_size=32):
        dataset = EfficientDummyDataset(
            num_samples=100,  # Poucas amostras para teste rápido
            img_size=self.config.IMG_SIZE,
            num_classes=getattr(self.config, 'NUM_CLASSES', 1000)
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    
    def get_val_loader(self, batch_size=32):
        dataset = EfficientDummyDataset(
            num_samples=50,  # Poucas amostras para teste rápido
            img_size=self.config.IMG_SIZE,
            num_classes=getattr(self.config, 'NUM_CLASSES', 1000)
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )