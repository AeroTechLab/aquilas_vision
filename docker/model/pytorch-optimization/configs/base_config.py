# configs/base_config.py
import torch

class BaseConfig:
    # Configurações W&B
    WANDB_PROJECT = "Model_Optimization_Framework"
    WANDB_ENTITY = None  # Sua equipe no W&B (opcional)
    
    # Paths do dataset
    COCO_BASE_PATH = "./datasets/coco"
    
    # Configurações de dispositivo
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configurações de treino/validação
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Configurações de quantização
    NUM_CALIBRATION_BATCHES = 50
    QUANTIZATION_BITS = 8
    
    # Configurações de logging
    LOG_INTERVAL = 10
    LOG_IMAGES = True
    LOG_HISTOGRAMS = True
    
    def __init__(self):
        # Garante que o dataset está disponível
        from utils.coco_downloader import coco_manager
        if not coco_manager.is_downloaded():
            print("📥 Dataset COCO não encontrado. Iniciando download...")
            coco_manager.download_all()