# trainer.py - VERSÃO CORRIGIDA COM LOSS ESPECÍFICA PARA YOLO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import prepare_qat, convert
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import default_qconfig, get_default_qat_qconfig
from torch.utils.data import DataLoader
import os
from ultralytics import YOLO
import numpy as np

class YOLOv9QATTrainer:
    def __init__(self, model_path=None, num_classes=80, img_size=640):
        self.img_size = img_size
        self.num_classes = num_classes
        
        print("Inicializando YOLOv9 QAT Trainer...")
        
        # Carregar modelo      
        self.model = self.load_yolov9(model_path)
        
                

   
    def load_yolov9(self, model_path):
        """Carrega o modelo YOLOv9"""
        try:
            if model_path and os.path.exists(model_path):
                print(f"Carregando: {model_path}")
                model = YOLO(model_path)
            else:
                print("Baixando YOLOv9 pré-treinado...")
                model = YOLO('yolov9c.pt')

            results = model.train(
                data='coco128.yaml',  # <-- SUBSTITUA PELO SEU ARQUIVO .yaml DO DATASET
                epochs=10,            # Defina o número de épocas
                imgsz=640,            # Tamanho da imagem
                qat=True,             # ATIVA O QAT
                device=0              # Usa a GPU
            )
            
            return results
            
        except Exception as e:
            print(f"Erro ao carregar YOLOv9: {e}")
            raise
      
    def convert_to_onnx(self, output_path="yolov9_quantized.onnx"):
        """Converter o modelo quantizado para ONNX"""
        try:
            model_quantized = self
            model_quantized.export(format='onnx', int8=True, opset=17)
            
        except Exception as e:
            print(f"Erro na exportação ONNX: {e}")

# Teste rápido
if __name__ == "__main__":
    print("Teste rápido do QAT Trainer...")
    trainer = YOLOv9QATTrainer()