# Configuracoes organizadas

# Configuracoes de treinamento
class TrainingConfig:
    # Hiperparametros
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 4
    IMG_SIZE = 640
    NUM_EPOCHS = 100
    NUM_CLASSES = 80
    
    # Configuracoes float8
    FLOAT8_ENABLED = True
    COMPILE_MODEL = True
    
    # Paths
    MODEL_PATH = 'yolov9c.pt'
    OUTPUT_DIR = 'checkpoints'
    
    # Dataset
    DATA_YAML = 'data/coco.yaml'  # Para COCO
    # DATA_YAML = 'data/meu_dataset.yaml'  # Para dataset customizado

# Configuracoes de modelo
class ModelConfig:
    ARCHITECTURE = 'yolov9'
    PRETRAINED = True
    FLOAT8_FILTER_EXCLUDE = ['detect', 'cv2', 'cv3', 'dfl']

