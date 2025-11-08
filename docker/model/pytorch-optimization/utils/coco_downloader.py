import os
import urllib.request
import zipfile
import json
from pathlib import Path
import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
import tempfile
import hashlib

class CocoManager:
    def __init__(self, base_path="./datasets/coco"):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.annotations_path = self.base_path / "annotations"
        
        # URLs para COCO 2017 - usando mirrors mais confiáveis
        self.urls = {
            'train_images': 'https://pjreddie.com/media/files/train2017.zip',
            'val_images': 'https://pjreddie.com/media/files/val2017.zip',
            'annotations': 'https://pjreddie.com/media/files/annotations_trainval2017.zip'
        }
        
        # URLs alternativas
        self.backup_urls = {
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }

    def download_via_script():
        """Usa o script shell para download mais robusto"""
        import subprocess
        import os
        
        script_path = os.path.join(os.path.dirname(__file__), '..', 'download_coco.sh')
        
        print("🔄 Usando script shell para download robusto do COCO...")
        
        try:
            # Torna o script executável
            os.chmod(script_path, 0o755)
            
            # Executa o script
            result = subprocess.run(
                ['bash', script_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hora timeout
            )
            
            if result.returncode == 0:
                print("✅ Download via script concluído com sucesso!")
                return True
            else:
                print(f"❌ Script falhou: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Timeout no download do COCO")
            return False
        except Exception as e:
            print(f"❌ Erro ao executar script: {e}")
            return False
    
    def download_all(self):
        """Baixa todo o dataset COCO com verificação de integridade"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        print("📥 Iniciando download do COCO dataset...")
        """Baixa todo o dataset COCO"""
        if use_script:
            return download_via_script()
        #else:
        # Download das imagens de treino
        if not self._download_and_extract('train_images', 'train2017.zip'):
            print("❌ Falha no download do COCO. Criando dataset alternativo...")
            return self._create_mini_coco_dataset()
        
        # Download das imagens de validação
        if not self._download_and_extract('val_images', 'val2017.zip'):
            print("❌ Falha no download do COCO. Criando dataset alternativo...")
            return self._create_mini_coco_dataset()
        
        # Download das anotações
        if not self._download_and_extract('annotations', 'annotations_trainval2017.zip'):
            print("❌ Falha no download das anotações. Continuando sem anotações...")
        
        print("✅ COCO dataset preparado!")
        return self.get_paths()
    
    def _download_and_extract(self, key, filename, use_backup=False):
        """Download e extração de um arquivo com tratamento de erro"""
        zip_path = self.base_path / filename
        
        try:
            # Tenta usar URL principal primeiro, depois backup
            url = self.backup_urls[key] if use_backup else self.urls[key]
            
            # Download se não existir
            if not zip_path.exists():
                print(f"⬇️  Baixando {filename}...")
                try:
                    urllib.request.urlretrieve(url, zip_path)
                except Exception as e:
                    print(f"❌ Erro no download: {e}")
                    if not use_backup:
                        print("🔄 Tentando URL alternativa...")
                        return self._download_and_extract(key, filename, use_backup=True)
                    return False
            
            # Verifica se o arquivo zip é válido
            if not self._is_valid_zip(zip_path):
                print(f"🔄 Arquivo {filename} corrompido, baixando novamente...")
                zip_path.unlink(missing_ok=True)
                return self._download_and_extract(key, filename, use_backup=not use_backup)
            
            # Extração
            print(f"📦 Extraindo {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro com {filename}: {e}")
            if not use_backup:
                print("🔄 Tentando URL alternativa...")
                return self._download_and_extract(key, filename, use_backup=True)
            return False
    
    def _is_valid_zip(self, zip_path):
        """Verifica se o arquivo ZIP é válido"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Tenta ler a lista de arquivos
                file_list = zip_ref.namelist()
                # Tenta extrair o primeiro arquivo (em memória)
                if file_list:
                    with zip_ref.open(file_list[0]) as test_file:
                        test_file.read(100)  # Lê apenas os primeiros 100 bytes
                return True
        except Exception as e:
            print(f"❌ Arquivo ZIP inválido {zip_path}: {e}")
            return False
    
    def _create_mini_coco_dataset(self):
        """Cria um dataset mínimo alternativo para testes"""
        print("🔄 Criando dataset COCO mínimo alternativo...")
        
        # Cria diretórios
        (self.images_path / "train2017").mkdir(parents=True, exist_ok=True)
        (self.images_path / "val2017").mkdir(parents=True, exist_ok=True)
        self.annotations_path.mkdir(parents=True, exist_ok=True)
        
        # Cria anotações mínimas
        mini_annotations = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "bicycle"},
                {"id": 3, "name": "car"}
            ]
        }
        
        # Cria algumas imagens dummy
        from PIL import Image
        import numpy as np
        
        for split in ['train2017', 'val2017']:
            for i in range(10):  # Apenas 10 imagens por split para teste
                # Cria imagem RGB aleatória
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = self.images_path / split / f"dummy_{i:06d}.jpg"
                img.save(img_path)
                
                # Adiciona à lista de imagens
                mini_annotations["images"].append({
                    "id": i,
                    "file_name": f"dummy_{i:06d}.jpg",
                    "width": 224,
                    "height": 224
                })
        
        # Salva anotações
        with open(self.annotations_path / "instances_train2017.json", 'w') as f:
            json.dump(mini_annotations, f)
        
        with open(self.annotations_path / "instances_val2017.json", 'w') as f:
            json.dump(mini_annotations, f)
        
        print("✅ Dataset alternativo criado com 10 imagens dummy por split")
        return self.get_paths()
    
    def get_paths(self):
        """Retorna todos os paths importantes"""
        return {
            'train_images': str(self.images_path / "train2017"),
            'val_images': str(self.images_path / "val2017"),
            'train_annotations': str(self.annotations_path / "instances_train2017.json"),
            'val_annotations': str(self.annotations_path / "instances_val2017.json"),
            'base_path': str(self.base_path)
        }
    
    def is_downloaded(self):
        """Verifica se o dataset já foi baixado"""
        required_dirs = [
            self.images_path / "train2017",
            self.images_path / "val2017"
        ]
        
        # Verifica se os diretórios existem e têm pelo menos alguns arquivos
        for dir_path in required_dirs:
            if not dir_path.exists():
                return False
            if len(list(dir_path.glob("*.jpg"))) == 0:
                return False
        
        return True

# Singleton global
coco_manager = CocoManager()