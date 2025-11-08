import torch

class QuickTestConfig:
    """Configurações otimizadas para testes rápidos"""
    
    # Configurações gerais
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configurações de dataset
    BATCH_SIZE = 8
    IMG_SIZE = 224  # Para MobileNetV2
    
    # Configurações de calibração (reduzidas)
    NUM_CALIBRATION_BATCHES = 5
    CALIBRATION_DATA = "coco"
    
    # Configurações de treino (reduzidas)
    NUM_EPOCHS = 2
    FINE_TUNE_EPOCHS = 1
    TRAIN_BATCHES = 10
    FINE_TUNE_BATCHES = 5
    
    # Configurações de quantização
    QUANTIZATION_BITS = 8
    PER_CHANNEL_QUANTIZATION = True
    
    # Configurações de pruning
    PRUNING_AMOUNT = 0.2  # 20% para teste rápido
    
    # Configurações de saída
    OUTPUT_DIR = "quick_test_results"
    
    # W&B (opcional para testes rápidos)
    USE_WANDB = False
    
    def apply_to_config(self, config):
        """Aplica configurações rápidas a qualquer config"""
        for key, value in self.__dict__.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config