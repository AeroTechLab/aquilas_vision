# train.py 

import torch
from trainer import YOLOv9QATTrainer
from torch.utils.data import DataLoader
import argparse
import os

def create_dummy_dataloader(batch_size=4, img_size=640, num_classes=80):
    """
    Dataset fictício 
    """
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Criar targets no formato YOLO: [class_id, x_center, y_center, width, height]
    targets = []
    for i in range(batch_size):
        # Criar algumas bboxes fictícias por imagem
        num_boxes = torch.randint(1, 3, (1,)).item()
        boxes = []
        for j in range(num_boxes):
            class_id = torch.randint(0, num_classes, (1,)).item()
            x_center, y_center = torch.rand(2).tolist()
            width, height = torch.rand(2).mul(0.3).add(0.1).tolist()  # 0.1-0.4 size
            boxes.append([class_id, x_center, y_center, width, height])
        
        targets.append(torch.tensor(boxes))
    
    return images, targets

def main():
    parser = argparse.ArgumentParser(description='Treinar YOLOv9 com QAT')
    parser.add_argument('--model-path', type=str, default='yolov9c.pt', help='Caminho do modelo')
    parser.add_argument('--epochs', type=int, default=3, help='Número de épocas para QAT')  # Reduzido
    parser.add_argument('--batch-size', type=int, default=2, help='Tamanho do batch')
    parser.add_argument('--img-size', type=int, default=640, help='Tamanho da imagem')
    parser.add_argument('--num-classes', type=int, default=80, help='Número de classes')
    
    args = parser.parse_args()
    
    # Inicializar trainer QAT
    print("Inicializando YOLOv9 QAT Trainer...")
    trainer = YOLOv9QATTrainer(
        model_path=args.model_path,
        num_classes=args.num_classes,
        img_size=args.img_size
    )
    
    # Loop de treinamento QAT
    print("Iniciando treinamento QAT...")
    for epoch in range(args.epochs):
        print(f"\nÉpoca {epoch+1}/{args.epochs}")
        
        # Treinar com alguns batches
        epoch_loss = 0
        num_batches = 5  # Número reduzido de batches para teste
        
        for batch_idx in range(num_batches):
            # Criar batch dinâmico
            images, targets = create_dummy_dataloader(
                batch_size=args.batch_size, 
                img_size=args.img_size,
                num_classes=args.num_classes
            )
            
            # Treinar um batch
            trainer.model.train()
            trainer.optimizer.zero_grad()
            
            # Mover para GPU
            images = images.cuda()
            
            # Forward
            predictions = trainer.model(images)
            loss = trainer.compute_loss(predictions, targets)
            
            # Backward
            loss.backward()
            trainer.optimizer.step()
            
            epoch_loss += loss.item()
            
            print(f'  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / num_batches
        print(f'Época {epoch+1} - Loss médio: {avg_loss:.4f}')
        
        # Salvar checkpoint
        checkpoint_path = f'checkpoints/yolov9_qat_epoch_{epoch+1}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        trainer.save_checkpoint(checkpoint_path, epoch)
    
    # Converter para modelo quantizado final
    print("Convertendo para modelo quantizado final...")
    quantized_model_path = 'yolov9_quantized.pth'
    trainer.save_quantized_model(quantized_model_path)
    
    print("QAT concluído! Modelo quantizado pronto para inferência.")

if __name__ == "__main__":
    main()