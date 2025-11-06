# Funcoes auxiliares
#NÃO ESTÁ SENDO USADA

import torch
import numpy as np
import os
import sys
import time
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
#from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms
# Imports Específicos do PT2E QAT e Backend (XNNPACK)
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
import torchao.quantization.pt2e
from ultralytics import YOLO
# Configuração de warnings
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*') 
warnings.filterwarnings(action='default', module=r'torchao.quantization.pt2e')
_ = torch.manual_seed(191009) 

# --- Definições de Classes Auxiliares ---
class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n [6]
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = ' {name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# --- Funções Auxiliares ---
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions""" 
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size)) [2]
        return res

def evaluate(model, criterion, data_loader, device, neval_batches):
    torchao.quantization.pt2e.move_exported_model_to_eval(model) 
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1, image.size(0))
            top5.update(acc5, image.size(0))
            if cnt >= neval_batches: # Limita o número de batches para avaliação rápida
                break
    print ('')
    return top1, top5

def load_model(model_file):
    # Esta função carrega o ResNet18. VOCÊ DEVERÁ SUBSTITUÍ-LA PELA LÓGICA DE CARREGAMENTO DO YOLOv9.
    # Exemplo YOLOv9 (apenas para referência, o objeto retornado DEVE ser um torch.nn.Module para export):
    # from ultralytics import YOLO
    # model = YOLO("yolov9c.pt").model
    model = YOLO(model_file)
    state_dict = torch.load(model_file, weights_only=True)
    model.load_state_dict(state_dict)
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def prepare_data_loaders(data_path):
    """normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)"""
    # Source - https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
# Posted by Fábio Perez
# Retrieved 2025-11-06, License - CC BY-SA 4.0

    data_loader, data_loader_test = torch.utils.data.random_split("coco.yaml", [0.8, 0.2])

    return data_loader, data_loader_test

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches, nepoch):
    # Note: do not call model.train() here, since this doesn't work on an exported model.
    # Instead, call `torchao.quantization.pt2e.move_exported_model_to_train(model)`, which will
    # be added in the near future
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return
            
data_path = 'coco.yaml'
saved_model_dir = 'data/'
float_model_file = 'yolov9c.pt'

train_batch_size = 32
eval_batch_size = 32

data_loader, data_loader_test = prepare_data_loaders(data_path)
example_inputs = (next(iter(data_loader))[0])
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to("cuda")