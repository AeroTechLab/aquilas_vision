# utils/wandb_utils.py
import wandb
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as transforms

class WandBLogger:
    def __init__(self, project_name, experiment_name, config):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.run = None
        
    def init_run(self):
        """Inicializa uma run no W&B"""
        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=self.config,
            dir="./wandb_logs"
        )
        
        # Criar diretório para logs
        Path("./wandb_logs").mkdir(exist_ok=True)
        return self.run
    
    def log_metrics(self, metrics, step=None):
        """Loga métricas no W&B"""
        if self.run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_model(self, model, model_name, metadata=None):
        """Loga modelo como artefato"""
        if self.run:
            # Salvar modelo temporariamente
            model_path = f"./wandb_logs/{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Logar como artefato
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                description=f"Optimized model: {model_name}",
                metadata=metadata or {}
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
    
    def log_images_with_predictions(self, images, predictions, targets=None, class_names=None, max_images=16):
        """Loga imagens com predições para visualização"""
        if not self.run:
            return
            
        # Limitar número de imagens
        images = images[:max_images]
        predictions = predictions[:max_images]
        
        wandb_images = []
        for i, (image, pred) in enumerate(zip(images, predictions)):
            # Converter tensor para PIL se necessário
            if torch.is_tensor(image):
                image = self._tensor_to_pil(image)
            
            # Criar figura
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image)
            ax.set_title(f"Pred: {pred}", fontsize=12)
            ax.axis('off')
            
            # Converter figura para imagem wandb
            wandb_images.append(wandb.Image(fig, caption=f"Sample {i}"))
            plt.close(fig)
        
        self.run.log({"predictions": wandb_images})
    
    def log_histograms(self, model, step=None):
        """Loga histogramas de pesos e gradientes"""
        if not self.run:
            return
            
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.run.log({
                    f"weights/{name}": wandb.Histogram(param.data.cpu()),
                    f"gradients/{name}": wandb.Histogram(param.grad.cpu())
                }, step=step)
    
    def log_optimization_comparison(self, original_metrics, optimized_metrics):
        """Loga comparação entre modelo original e otimizado"""
        comparison_data = []
        
        for metric_name in original_metrics.keys():
            comparison_data.append({
                "Metric": metric_name,
                "Original": original_metrics[metric_name],
                "Optimized": optimized_metrics.get(metric_name, 0),
                "Improvement": optimized_metrics.get(metric_name, 0) - original_metrics[metric_name]
            })
        
        # Tabela de comparação
        table = wandb.Table(data=comparison_data, columns=["Metric", "Original", "Optimized", "Improvement"])
        self.run.log({"optimization_comparison": table})
        
        # Gráfico de barras
        self.run.log({
            "metrics_comparison": wandb.plot.bar(
                table, "Metric", "Original", 
                title="Original vs Optimized Metrics"
            )
        })
    
    def _tensor_to_pil(self, tensor):
        """Converte tensor para PIL Image"""
        # Desnormalizar se necessário
        if tensor.min() < 0:
            transform = transforms.Compose([
                transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                ),
                transforms.ToPILImage()
            ])
        else:
            transform = transforms.ToPILImage()
        
        return transform(tensor.cpu())
    
    def finish(self):
        """Finaliza a run do W&B"""
        if self.run:
            self.run.finish()

# Singleton para logger global
wandb_logger = None

def setup_wandb(project_name, experiment_name, config):
    """Configura logger global do W&B"""
    global wandb_logger
    wandb_logger = WandBLogger(project_name, experiment_name, config)
    return wandb_logger.init_run()

def get_wandb_logger():
    """Retorna o logger global do W&B"""
    return wandb_logger