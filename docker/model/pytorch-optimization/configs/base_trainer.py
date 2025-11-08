# utils/base_trainer.py
import torch
import time
from pathlib import Path
from utils.wandb_utils import get_wandb_logger

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.best_metric = 0.0
        
        # Setup W&B
        self.wandb_logger = get_wandb_logger()
        
    def setup_training(self):
        """Configura ambiente de treino"""
        raise NotImplementedError
        
    def train_epoch(self, epoch):
        """Executa uma época de treino"""
        raise NotImplementedError
        
    def validate(self, epoch):
        """Executa validação"""
        raise NotImplementedError
        
    def log_training_metrics(self, metrics, epoch):
        """Loga métricas de treino no W&B"""
        if self.wandb_logger:
            self.wandb_logger.log_metrics({
                f"train/{k}": v for k, v in metrics.items()
            }, step=epoch)
    
    def log_validation_metrics(self, metrics, epoch):
        """Loga métricas de validação no W&B"""
        if self.wandb_logger:
            self.wandb_logger.log_metrics({
                f"val/{k}": v for k, v in metrics.items()
            }, step=epoch)
            
            # Log do melhor métrica
            if 'accuracy' in metrics and metrics['accuracy'] > self.best_metric:
                self.best_metric = metrics['accuracy']
                self.wandb_logger.log_metrics({
                    "best_accuracy": self.best_metric
                }, step=epoch)
    
    def log_model_summary(self, model, input_size=(1, 3, 224, 224)):
        """Loga resumo do modelo no W&B"""
        if self.wandb_logger and self.wandb_logger.run:
            # Log da arquitetura do modelo
            self.wandb_logger.run.log_code("./models")
            
            # Log do número de parâmetros
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.wandb_logger.log_metrics({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params
            })
    
    def save_checkpoint(self, epoch, is_best=False):
        """Salva checkpoint e loga no W&B"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Salvar localmente
        checkpoint_path = f"./checkpoints/epoch_{epoch}.pt"
        Path("./checkpoints").mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Logar como artefato no W&B
        if self.wandb_logger:
            artifact = wandb.Artifact(
                name=f"checkpoint-epoch-{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}"
            )
            artifact.add_file(checkpoint_path)
            self.wandb_logger.run.log_artifact(artifact)