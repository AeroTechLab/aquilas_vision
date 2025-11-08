import argparse
import sys
import os
import time

# Adicionar o diretório raiz ao path do Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.coco_downloader import coco_manager
from configs.mobilenetv2_config import MobileNetV2Config
from configs.yolo_config import YOLOConfig

'''def setup_environment(force_download=False):
    """Configura o ambiente uma única vez"""
    print("🚀 Configurando ambiente de otimização...")
    
    # Garante que o COCO está disponível
    if force_download or not coco_manager.is_downloaded():
        print("📥 Baixando dataset COCO...")
        coco_manager.download_all()
    else:
        print("✅ Dataset COCO já disponível")
    
    print("✅ Ambiente configurado!")'''

# ===============================
# MobileNetV2 Methods
# ===============================

def run_mobilenetv2_ptq():
    """Executa PTQ no MobileNetV2"""
    try:
        from models.mobilenetv2.ptq_pos_training.trainer import MobileNetV2PTQTrainer
        
        print("\n🎯 Executando PTQ no MobileNetV2...")
        config = MobileNetV2Config(optimization_type='ptq')
        trainer = MobileNetV2PTQTrainer(config)
        return trainer.run_ptq_pipeline()
    except Exception as e:
        print(f"❌ MobileNetV2 PTQ falhou: {e}")
        return False

def run_mobilenetv2_qat():
    """Executa QAT no MobileNetV2"""
    try:
        from models.mobilenetv2.qat_training.trainer import MobileNetV2QATTrainer
        
        print("\n🎯 Executando QAT no MobileNetV2...")
        config = MobileNetV2Config(optimization_type='qat')
        trainer = MobileNetV2QATTrainer(config)
        return trainer.run_qat_pipeline()
    except Exception as e:
        print(f"❌ MobileNetV2 QAT falhou: {e}")
        return False

def run_mobilenetv2_pruning():
    """Executa Pruning no MobileNetV2"""
    try:
        from models.mobilenetv2.pruning.trainer import MobileNetV2PruningTrainer
        
        print("\n🎯 Executando Pruning no MobileNetV2...")
        config = MobileNetV2Config(optimization_type='pruning')
        trainer = MobileNetV2PruningTrainer(config)
        return trainer.run_pruning_pipeline()
    except Exception as e:
        print(f"❌ MobileNetV2 Pruning falhou: {e}")
        return False

def run_mobilenetv2_mp():
    """Executa Mixed Precision no MobileNetV2"""
    try:
        from models.mobilenetv2.mp_mixed_precision.trainer import MobileNetV2MPTrainer
        
        print("\n🎯 Executando Mixed Precision no MobileNetV2...")
        config = MobileNetV2Config(optimization_type='mp')
        trainer = MobileNetV2MPTrainer(config)
        return trainer.run_mp_pipeline()
    except Exception as e:
        print(f"❌ MobileNetV2 Mixed Precision falhou: {e}")
        return False

def run_mobilenetv2_int8():
    """Executa PTQ INT8 no MobileNetV2"""
    try:
        from models.mobilenetv2.ptq_int8.trainer import MobileNetV2PTQINT8Trainer
        
        print("\n🎯 Executando PTQ INT8 no MobileNetV2...")
        config = MobileNetV2Config(optimization_type='ptq_int8')
        trainer = MobileNetV2PTQINT8Trainer(config)
        return trainer.run_int8_pipeline()
    except Exception as e:
        print(f"❌ MobileNetV2 PTQ INT8 falhou: {e}")
        return False

# ===============================
# YOLO Methods - TODOS CORRIGIDOS
# ===============================

def run_yolo_ptq():
    """Executa PTQ no YOLO"""
    try:
        from models.yolov9.ptq_pos_training.trainer import YOLOPTQTrainer
        
        print("\n🎯 Executando PTQ no YOLO...")
        config = YOLOConfig(optimization_type='ptq')
        trainer = YOLOPTQTrainer(config)
        return trainer.run_ptq_pipeline()
    except Exception as e:
        print(f"❌ YOLO PTQ falhou: {e}")
        return False

def run_yolo_qat():
    """Executa QAT no YOLO"""
    try:
        from models.yolov9.qat_training.trainer import YOLOQATTrainer
        
        print("\n🎯 Executando QAT no YOLO...")
        config = YOLOConfig(optimization_type='qat')
        trainer = YOLOQATTrainer(config)
        return trainer.run_qat_pipeline()
    except Exception as e:
        print(f"❌ YOLO QAT falhou: {e}")
        return False

def run_yolo_pruning():
    """Executa Pruning no YOLO"""
    try:
        from models.yolov9.pruning.trainer import YOLOPruningTrainer
        
        print("\n🎯 Executando Pruning no YOLO...")
        config = YOLOConfig(optimization_type='pruning')
        trainer = YOLOPruningTrainer(config)
        return trainer.run_pruning_pipeline()
    except Exception as e:
        print(f"❌ YOLO Pruning falhou: {e}")
        return False

def run_yolo_mp():
    """Executa Mixed Precision no YOLO"""
    try:
        from models.yolov9.mp_mixed_precision.trainer import YOLOMPTrainer
        
        print("\n🎯 Executando Mixed Precision no YOLO...")
        config = YOLOConfig(optimization_type='mp')
        trainer = YOLOMPTrainer(config)
        return trainer.run_mp_pipeline()
    except Exception as e:
        print(f"❌ YOLO Mixed Precision falhou: {e}")
        return False

def run_yolo_int8():
    """Executa PTQ INT8 no YOLO"""
    try:
        from models.yolov9.ptq_int8.trainer import YOLOPTQINT8Trainer
        
        print("\n🎯 Executando PTQ INT8 no YOLO...")
        config = YOLOConfig(optimization_type='ptq_int8')
        trainer = YOLOPTQINT8Trainer(config)
        return trainer.run_int8_pipeline()
    except Exception as e:
        print(f"❌ YOLO PTQ INT8 falhou: {e}")
        return False

def run_yolo_qlora():
    """Executa QLoRA no YOLO"""
    try:
        from models.yolov9.QLoRA.trainer import YOLOQLoRATrainer
        
        print("\n🎯 Executando QLoRA no YOLO...")
        config = YOLOConfig(optimization_type='qlora')
        trainer = YOLOQLoRATrainer(config)
        return trainer.run_qlora_pipeline()
    except Exception as e:
        print(f"❌ YOLO QLoRA falhou: {e}")
        return False

# ===============================
# Quick Test System - CORRIGIDO
# ===============================

def run_quick_test():
    """Executa teste rápido de todas as otimizações"""
    print("⚡ EXECUTANDO TESTE RÁPIDO DE TODAS AS OTIMIZAÇÕES")
    print("=" * 60)
    
    # Configurações de teste rápido
    quick_config = {
        'BATCH_SIZE': 4,  # Reduzido para teste rápido
        'NUM_CALIBRATION_BATCHES': 2,  # Muito reduzido
        'NUM_EPOCHS': 1,
        'FINE_TUNE_EPOCHS': 1,
        'TRAIN_BATCHES': 2,
        'FINE_TUNE_BATCHES': 2,
    }
    
    results = {}
    
    # MobileNetV2 Quick Tests
    print("\n🧠 MOBILENETV2 (Teste Rápido)")
    print("-" * 40)
    
    # PTQ
    try:
        from models.mobilenetv2.ptq_pos_training.trainer import MobileNetV2PTQTrainer
        from models.mobilenetv2.ptq_pos_training.config import MobileNetV2PTQConfig
        
        config = MobileNetV2PTQConfig()
        for key, value in quick_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        trainer = MobileNetV2PTQTrainer(config)
        results['mobilenetv2_ptq'] = trainer.run_ptq_pipeline()
    except Exception as e:
        print(f"❌ MobileNetV2 PTQ falhou: {e}")
        results['mobilenetv2_ptq'] = False
    
    # Pruning
    try:
        from models.mobilenetv2.pruning.trainer import MobileNetV2PruningTrainer
        from models.mobilenetv2.pruning.config import MobileNetV2PruningConfig
        
        config = MobileNetV2PruningConfig()
        for key, value in quick_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        trainer = MobileNetV2PruningTrainer(config)
        results['mobilenetv2_pruning'] = trainer.run_pruning_pipeline()
    except Exception as e:
        print(f"❌ MobileNetV2 Pruning falhou: {e}")
        results['mobilenetv2_pruning'] = False
    
    # YOLO Quick Tests
    print("\n🎯 YOLO (Teste Rápido)")
    print("-" * 40)
    
    # PTQ
    try:
        from models.yolov9.ptq_pos_training.trainer import YOLOPTQTrainer
        from models.yolov9.ptq_pos_training.config import YOLOPTQConfig
        
        config = YOLOPTQConfig()
        for key, value in quick_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        trainer = YOLOPTQTrainer(config)
        results['yolo_ptq'] = trainer.run_ptq_pipeline()
    except Exception as e:
        print(f"❌ YOLO PTQ falhou: {e}")
        results['yolo_ptq'] = False
    
    # Pruning
    try:
        from models.yolov9.pruning.trainer import YOLOPruningTrainer
        from models.yolov9.pruning.config import YOLOPruningConfig
        
        config = YOLOPruningConfig()
        for key, value in quick_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        trainer = YOLOPruningTrainer(config)
        results['yolo_pruning'] = trainer.run_pruning_pipeline()
    except Exception as e:
        print(f"❌ YOLO Pruning falhou: {e}")
        results['yolo_pruning'] = False
    
    # Relatório final
    print("\n📊 RELATÓRIO FINAL DO TESTE RÁPIDO")
    print("=" * 50)
    for optimization, success in results.items():
        status = "✅ SUCESSO" if success else "❌ FALHA"
        print(f"{optimization}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\n📈 TAXA DE SUCESSO: {success_rate:.1f}%")
    
    return success_count > 0

# ===============================
# Funções Auxiliares
# ===============================

def list_available_optimizations():
    """Lista todas as otimizações disponíveis"""
    print("\n📋 OTIMIZAÇÕES DISPONÍVEIS:")
    print("\nMobileNetV2:")
    print("  • ptq       - Post-Training Quantization")
    print("  • qat       - Quantization-Aware Training") 
    print("  • pruning   - Pruning")
    print("  • mp        - Mixed Precision")
    print("  • int8      - PTQ INT8")
    
    print("\nYOLO:")
    print("  • ptq       - Post-Training Quantization")
    print("  • qat       - Quantization-Aware Training")
    print("  • pruning   - Pruning")
    print("  • mp        - Mixed Precision")
    print("  • int8      - PTQ INT8")
    print("  • qlora     - QLoRA Fine-tuning")

def test_individual_optimization(model, optimization):
    """Testa uma otimização específica"""
    print(f"\n🎯 TESTANDO {model.upper()} - {optimization.upper()}")
    print("=" * 40)
    
    if model == 'mobilenetv2':
        if optimization == 'ptq':
            return run_mobilenetv2_ptq()
        elif optimization == 'qat':
            return run_mobilenetv2_qat()
        elif optimization == 'pruning':
            return run_mobilenetv2_pruning()
        elif optimization == 'mp':
            return run_mobilenetv2_mp()
        elif optimization == 'int8':
            return run_mobilenetv2_int8()
    
    elif model == 'yolo':
        if optimization == 'ptq':
            return run_yolo_ptq()
        elif optimization == 'qat':
            return run_yolo_qat()
        elif optimization == 'pruning':
            return run_yolo_pruning()
        elif optimization == 'mp':
            return run_yolo_mp()
        elif optimization == 'int8':
            return run_yolo_int8()
        elif optimization == 'qlora':
            return run_yolo_qlora()
    
    print(f"❌ Otimização não encontrada: {optimization}")
    return False

def setup_environment(force_download=False, skip_coco=False):
    """Configura o ambiente uma única vez"""
    print("🚀 Configurando ambiente de otimização...")
    
    if skip_coco:
        print("⏭️  Pulando download do COCO - usando dataset dummy")
        return
    
    # Garante que o COCO está disponível
    if force_download or not coco_manager.is_downloaded():
        print("📥 Baixando dataset COCO...")
        success = coco_manager.download_all()
        if not success:
            print("⚠️  Download do COCO falhou. Continuando com dataset dummy...")
    else:
        print("✅ Dataset COCO já disponível")
    
    print("✅ Ambiente configurado!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Framework de Otimização')
    parser.add_argument('--model', type=str, choices=['mobilenetv2', 'yolo', 'all'], default='all')
    parser.add_argument('--optimization', type=str, 
                       choices=['ptq', 'qat', 'pruning', 'mp', 'int8', 'qlora', 'all'], 
                       default='all')
    parser.add_argument('--setup-only', action='store_true', help='Apenas configura o ambiente')
    parser.add_argument('--force-download', action='store_true', help='Força novo download do COCO')
    parser.add_argument('--list', action='store_true', help='Lista otimizações disponíveis')
    parser.add_argument('--quick-test', action='store_true', help='Executa teste rápido')
    parser.add_argument('--test', type=str, help='Testa otimização específica (ex: mobilenetv2:ptq)')
    #parser = argparse.ArgumentParser(description='Framework de Otimização')
    parser.add_argument('--skip-coco', action='store_true', help='Pular download do COCO e usar dataset dummy')
    args = parser.parse_args()
    
    # Listar otimizações disponíveis
    if args.list:
        list_available_optimizations()
        exit(0)
    
    # Setup do ambiente
    setup_environment(force_download=args.force_download)
    
    if args.setup_only:
        print("✅ Setup concluído. Use outros comandos para executar otimizações.")
        exit(0)
    
    # Teste específico
    if args.test:
        if ':' in args.test:
            model, optimization = args.test.split(':')
            success = test_individual_optimization(model, optimization)
            exit(0 if success else 1)
        else:
            print("❌ Formato inválido. Use: --test modelo:otimizacao")
            exit(1)
    
    # Teste rápido
    if args.quick_test:
        success = run_quick_test()
        exit(0 if success else 1)
    
    # Executa otimizações baseadas nos argumentos
    if args.model == 'all' and args.optimization == 'all':
        # Executa apenas os testes básicos para não demorar muito
        print("🔬 EXECUTANDO OTIMIZAÇÕES BÁSICAS")
        results = {}
        results['mobilenetv2_ptq'] = run_mobilenetv2_ptq()
        results['yolo_ptq'] = run_yolo_ptq()
        results['mobilenetv2_pruning'] = run_mobilenetv2_pruning()
        results['yolo_pruning'] = run_yolo_pruning()
        
        success = any(results.values())  # Considera sucesso se pelo menos um funcionar
        exit(0 if success else 1)
    
    elif args.model == 'mobilenetv2':
        if args.optimization in ['ptq', 'all']:
            run_mobilenetv2_ptq()
        if args.optimization in ['qat', 'all']:
            run_mobilenetv2_qat()
        if args.optimization in ['pruning', 'all']:
            run_mobilenetv2_pruning()
        if args.optimization in ['mp', 'all']:
            run_mobilenetv2_mp()
        if args.optimization in ['int8', 'all']:
            run_mobilenetv2_int8()
    
    elif args.model == 'yolo':
        if args.optimization in ['ptq', 'all']:
            run_yolo_ptq()
        if args.optimization in ['qat', 'all']:
            run_yolo_qat()
        if args.optimization in ['pruning', 'all']:
            run_yolo_pruning()
        if args.optimization in ['mp', 'all']:
            run_yolo_mp()
        if args.optimization in ['int8', 'all']:
            run_yolo_int8()
        if args.optimization in ['qlora', 'all']:
            run_yolo_qlora()

    setup_environment(force_download=args.force_download, skip_coco=args.skip_coco)