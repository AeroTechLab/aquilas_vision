#!/usr/bin/env python3
"""
Verificação rápida do ambiente - sem downloads
"""
import sys
import importlib.util

def check_import(package_name, install_name=None):
    """Verifica se um pacote pode ser importado"""
    if install_name is None:
        install_name = package_name
    
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, f"❌ {package_name} não encontrado. Instale: pip install {install_name}"
        return True, f"✅ {package_name}"
    except ImportError as e:
        return False, f"❌ {package_name} - Erro: {e}"

def main():
    print("🔍 VERIFICAÇÃO RÁPIDA DO AMBIENTE")
    print("=" * 40)
    
    # Verifica pacotes principais
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"), 
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
        ("wandb", "wandb"),
        ("ultralytics", "ultralytics"),
    ]
    
    all_ok = True
    for package, install_name in packages:
        success, message = check_import(package, install_name)
        print(message)
        if not success:
            all_ok = False
    
    # Verifica módulos locais
    print("\n📁 MÓDULOS LOCAIS:")
    local_modules = [
        "utils.coco_downloader",
        "utils.coco_loader", 
        "configs.mobilenetv2_config",
        "configs.yolo_config",
    ]
    
    for module in local_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {e}")
            all_ok = False
    
    # Verifica CUDA
    print("\n⚡ ACELERAÇÃO GPU:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponível - {torch.cuda.get_device_name(0)}")
            print(f"   Versão CUDA: {torch.version.cuda}")
        else:
            print("⚠️  CUDA não disponível - usando CPU")
    except Exception as e:
        print(f"❌ Erro ao verificar CUDA: {e}")
    
    # Resumo
    print("\n" + "=" * 40)
    if all_ok:
        print("🎉 Ambiente configurado corretamente!")
        print("\n💡 Dica: Execute agora:")
        print("   python test_framework_fast.py")
    else:
        print("❌ Problemas encontrados no ambiente.")
        print("   Corrija as dependências acima.")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)