import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from PIL import Image
import numpy as np
import os
from .coco_downloader import coco_manager

class DummyCocoDataset(Dataset):
    """Dataset dummy quando COCO não está disponível"""
    def __init__(self, num_samples=1000, img_size=224, num_classes=80):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Gera imagem aleatória
        image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        
        # Gera target dummy (para classificação)
        target = np.random.randint(0, self.num_classes)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

class RobustCocoDataLoader:
    def __init__(self, config):
        self.config = config
        self.coco_paths = coco_manager.get_paths()
        self.use_dummy = False
        
        # Verifica se COCO está disponível
        if not coco_manager.is_downloaded():
            print("⚠️  COCO não disponível, usando dataset dummy")
            self.use_dummy = True
    
    def get_calibration_loader(self, split='val', num_samples=None):
        """Retorna DataLoader para calibração"""
        if num_samples is None:
            num_samples = self.config.BATCH_SIZE * self.config.NUM_CALIBRATION_BATCHES
        
        if self.use_dummy:
            dataset = DummyCocoDataset(
                num_samples=num_samples,
                img_size=self.config.IMG_SIZE,
                num_classes=self.config.NUM_CLASSES
            )
        else:
            transform = self._get_transform(augmentation=False)
            try:
                dataset = CocoDetection(
                    root=self.coco_paths[f'{split}_images'],
                    annFile=self.coco_paths[f'{split}_annotations'],
                    transform=transform
                )
                # Limita o número de amostras
                from torch.utils.data import Subset
                indices = torch.randperm(len(dataset))[:num_samples]
                dataset = Subset(dataset, indices)
            except Exception as e:
                print(f"❌ Erro ao carregar COCO: {e}. Usando dataset dummy.")
                dataset = DummyCocoDataset(
                    num_samples=num_samples,
                    img_size=self.config.IMG_SIZE,
                    num_classes=self.config.NUM_CLASSES
                )
        
        return DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=min(2, os.cpu_count() // 2)  # Número seguro de workers
        )
    
    def _get_transform(self, augmentation=False):
        """Retorna transformações apropriadas"""
        if augmentation:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])