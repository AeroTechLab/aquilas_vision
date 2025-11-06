# trainer.py - VERSÃO SIMPLIFICADA E FUNCIONAL

import torch
from ultralytics import YOLO
import os

class YOLOv9QATTrainer:
    def __init__(self, model_path='yolov9c.pt'):        
        self.base_model_path = model_path
        self.model = YOLO(self.base_model_path)
        self.qat_model_path = None
        print("✅ Modelo base carregado com sucesso")

    def train_qat(self, data_yaml, epochs=10, imgsz=640, wandb_project=None):
        """
        Treina o modelo e prepara para quantização
        """
        print(f"🚀 Iniciando treinamento QAT para {self.base_model_path}...")
        
        try:
            # 1. Treinamento padrão do YOLO
            print("📚 Fase 1: Treinamento do modelo...")
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                project=wandb_project.replace('/', '_') if wandb_project else 'qat_training',
                name=f'train_epochs_{epochs}',
                save=True,
                exist_ok=True
            )
            
            # 2. Encontrar o melhor modelo treinado
            print("🔍 Buscando melhor modelo treinado...")
            if hasattr(results, 'save_dir'):
                weights_dir = results.save_dir
            else:
                # Padrão do Ultralytics para diretório de treinamento
                weights_dir = 'runs/detect/train_epochs_{epochs}' if epochs > 1 else 'runs/detect/train'
            
            best_model_path = os.path.join(weights_dir, 'weights/best.pt')
            
            if os.path.exists(best_model_path):
                print(f"✅ Melhor modelo encontrado: {best_model_path}")
                trained_model = YOLO(best_model_path)
            else:
                print("⚠️  Melhor modelo não encontrado, usando modelo base")
                trained_model = self.model
            
            # 3. Exportação para formato compatível com quantização
            print("📤 Fase 2: Exportação do modelo...")
            
            # Primeiro exportar para TorchScript (mais compatível)
            try:
                print("🔄 Exportando para TorchScript...")
                exported_path = trained_model.export(
                    format='torchscript',
                    imgsz=imgsz,
                    optimize=False  # Mais estável
                )
                self.qat_model_path = exported_path
                print(f"✅ Modelo exportado com sucesso: {exported_path}")
                
            except Exception as export_error:
                print(f"❌ Erro na exportação TorchScript: {export_error}")
                # Fallback: exportar para ONNX sem quantização
                print("🔄 Tentando exportação ONNX (sem quantização)...")
                exported_path = trained_model.export(
                    format='onnx',
                    imgsz=imgsz,
                    dynamic=False,
                    simplify=True
                )
                self.qat_model_path = exported_path
            
            return self.qat_model_path
            
        except Exception as e:
            print(f"❌ Erro durante o processo: {e}")
            print("🔄 Tentando abordagem alternativa...")
            return self._fallback_export(imgsz)

    def _fallback_export(self, imgsz=640):
        """Fallback se o treinamento falhar"""
        try:
            print("🔄 Exportação direta do modelo base...")
            exported_path = self.model.export(
                format='torchscript',
                imgsz=imgsz
            )
            self.qat_model_path = exported_path
            print(f"✅ Modelo base exportado: {exported_path}")
            return exported_path
        except Exception as e:
            print(f"❌ Falha total na exportação: {e}")
            return None

    def export_to_onnx(self, imgsz=640):
        """
        Exporta para ONNX (sem quantização INT8)
        """
        try:
            print("📤 Exportando para ONNX...")
            
            # Carregar o melhor modelo disponível
            if self.qat_model_path and os.path.exists(self.qat_model_path):
                model_to_export = YOLO(self.qat_model_path)
            else:
                model_to_export = self.model
            
            exported_path = model_to_export.export(
                format='onnx',
                imgsz=imgsz,
                dynamic=False,
                simplify=True,
                opset=17
            )
            
            print(f"✅ ONNX exportado: {exported_path}")
            return exported_path
            
        except Exception as e:
            print(f"❌ Erro na exportação ONNX: {e}")
            return None