# main_wandb.py
import argparse
import wandb
from utils.coco_downloader import coco_manager
from utils.wandb_utils import setup_wandb
from configs.mobilenetv2_config import MobileNetV2Config
from configs.yolo_config import YOLOConfig

def setup_environment():
    """Configura ambiente com W&B"""
    print("Configurando ambiente com Weights & Biases...")
    
    # Garante que o COCO está disponível
    if coco_manager.is_downloaded():
        print("Dataset COCO está disponível")
    else:
        print("Dataset COCO não está disponível")
    
    # Verifica login W&B
    try:
        wandb.login()
        print("W&B configurado")
    except Exception as e:
        print(f"Erro no W&B: {e}")
        print("Continuando sem W&B...")

def run_mobilenetv2_ptq_with_wandb():
    """Executa PTQ no MobileNetV2 com W&B"""
    from models.mobilenetv2.ptq_pos_training import MobileNetV2PTQTrainer
    
    config = MobileNetV2Config(optimization_type='ptq')
    config.WANDB_PROJECT = "MobileNetV2_Optimization"
    
    trainer = MobileNetV2PTQTrainer(config)
    return trainer.run_ptq_pipeline()

def run_comparative_analysis():
    """Executa análise comparativa completa com W&B"""
    print("EXECUTANDO ANÁLISE COMPARATIVA COM W&B")
    print("=" * 50)
    
    # Dashboard principal no W&B
    setup_wandb(
        project_name="Model_Optimization_Comparison",
        experiment_name="Comparative_Analysis",
        config={}
    )
    
    results = {}
    
    try:
        # MobileNetV2
        print("\nOtimizando MobileNetV2...")
        results['mobilenetv2_ptq'] = run_mobilenetv2_ptq_with_wandb()
        
        # YOLO (adaptar similarmente)
        # results['yolo_ptq'] = run_yolo_ptq_with_wandb()
        
        # Logar resultados finais
        from utils.wandb_utils import get_wandb_logger
        logger = get_wandb_logger()
        if logger:
            logger.log_metrics({
                "final/mobilenetv2_ptq_success": results['mobilenetv2_ptq'],
                "final/all_success": all(results.values())
            })
        
        return all(results.values())
        
    finally:
        from utils.wandb_utils import get_wandb_logger
        logger = get_wandb_logger()
        if logger:
            logger.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Framework de Otimização com W&B')
    parser.add_argument('--wandb-project', type=str, default='Model_Optimization_Framework')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--compare', action='store_true', help='Executar análise comparativa')
    
    args = parser.parse_args()
    
    # Setup do ambiente
    setup_environment()
    
    if args.compare:
        run_comparative_analysis()
    else:
        # Executar otimização específica
        run_mobilenetv2_ptq_with_wandb()