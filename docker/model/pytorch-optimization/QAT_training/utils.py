# Funcoes auxiliares

import torch
import numpy as np

"""Configura o ambiente de treinamento"""
def setup_training_environment():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

"""Verifica quais camadas foram convertidas para float8"""
def verify_float8_conversion(model):
    float8_layers = []
    normal_layers = []
    
    for name, module in model.named_modules():
        # Verificar se é uma camada float8 (depende da implementação)
        if hasattr(module, 'weight_scale') or 'float8' in str(type(module)).lower():
            float8_layers.append(name)
        elif isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            normal_layers.append(name)
    
    print("Verificação de conversão Float8:")
    print(f"Camadas Float8: {len(float8_layers)}")
    print(f"Camadas normais: {len(normal_layers)}")
    
    if float8_layers:
        print("\nCamadas convertidas para Float8:")
        for layer in float8_layers[:5]:  # Mostrar apenas as primeiras 5
            print(f"  - {layer}")
        if len(float8_layers) > 5:
            print(f"  ... e mais {len(float8_layers) - 5} camadas")
    
    return float8_layers, normal_layers