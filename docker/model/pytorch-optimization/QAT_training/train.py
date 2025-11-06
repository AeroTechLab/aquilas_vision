# train.py - VERSÃO ATUALIZADA

import torch
from trainer import YOLOv9QATTrainer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Treinar YOLOv9 com QAT')
    
    parser.add_argument('--model-path', type=str, default='yolov9c.pt', 
                        help='Caminho do modelo .pt base')
    parser.add_argument('--data-yaml', type=str, default='coco128.yaml', 
                        help='Caminho para o arquivo data.yaml')
    parser.add_argument('--epochs', type=int, default=3,  # Reduzido para teste
                        help='Número de épocas para QAT')
    parser.add_argument('--img-size', type=int, default=320,  # Reduzido para teste
                        help='Tamanho da imagem de entrada')
    parser.add_argument('--wandb-project', type=str, default=None, 
                        help='Nome do projeto no Weights & Biases')
    
    args = parser.parse_args()
    
    print("🚀 Inicializando YOLO Quantizer...")
    quantizer = YOLOv9QATTrainer(model_path=args.model_path)
    
    print("🔧 Iniciando processo de treinamento e exportação...")
    model_path = quantizer.train_qat(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.img_size,
        wandb_project=args.wandb_project
    )
    
    if model_path:
        print("📤 Exportando para ONNX...")
        onnx_path = quantizer.export_to_onnx(imgsz=args.img_size)
        
        if onnx_path:
            print(f"\n🎉 Processo concluído com sucesso!")
            print(f"📁 Modelo treinado: {model_path}")
            print(f"📁 ONNX exportado: {onnx_path}")
        else:
            print(f"\n⚠️  Processo parcialmente concluído")
            print(f"📁 Modelo treinado: {model_path}")
            print("❌ ONNX não foi exportado")
    else:
        print("\n❌ Processo falhou completamente")

if __name__ == "__main__":
    main()