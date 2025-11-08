# configs/mobilenetv2_config.py
from .base_config import BaseConfig

class MobileNetV2Config(BaseConfig):
    MODEL_TYPE = 'torchvision'
    IMG_SIZE = 224
    NUM_CLASSES = 1000  # ImageNet
    
    # Configurações de otimização
    OUTPUT_DIR = "./models/mobilenetv2"
    
    # Para fine-tuning no COCO
    COCO_NUM_CLASSES = 80
    FINE_TUNE_EPOCHS = 10
    FINE_TUNE_LR = 0.001
    
    def __init__(self, optimization_type='ptq'):
        super().__init__()
        self.OPTIMIZATION_TYPE = optimization_type
        self.OUTPUT_DIR = f"{self.OUTPUT_DIR}/{optimization_type}"