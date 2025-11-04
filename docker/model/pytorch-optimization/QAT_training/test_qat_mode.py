# test_qat_mode.py
from trainer import YOLOv9QATTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import prepare_qat, convert
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import default_qconfig, get_default_qat_qconfig
from torch.utils.data import DataLoader

def test_model_mode():
    """Testa se o modelo está no modo correto para QAT"""
    try:
        print("🧪 Testando modo do modelo...")
        trainer = YOLOv9QATTrainer()
        
        print(f"📊 Modo final do modelo: {'treinamento' if trainer.model.training else 'avaliação'}")
        print(f"📊 Tem QAT config: {hasattr(trainer.model, 'qconfig') and trainer.model.qconfig is not None}")
        
        # Testar forward pass
        dummy_input = torch.randn(1, 3, 640, 640).cuda()
        with torch.no_grad():
            output = trainer.model(dummy_input)
            print(f"✅ Forward pass funciona! Output: {type(output)}")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_mode()