# configs/yolo_config.py
from .base_config import BaseConfig

class YOLOConfig(BaseConfig):
    MODEL_TYPE = 'ultralytics'
    IMG_SIZE = 640
    BATCH_SIZE = 8
    
    # Configurações YOLO específicas
    YOLO_VERSION = 'yolov8n'  # ou yolov9c, etc
    NUM_CLASSES = 80  # COCO
    
    # Configurações de otimização
    OUTPUT_DIR = "./models/yolo"
    
    def __init__(self, optimization_type='ptq'):
        super().__init__()
        self.OPTIMIZATION_TYPE = optimization_type
        self.OUTPUT_DIR = f"{self.OUTPUT_DIR}/{optimization_type}"