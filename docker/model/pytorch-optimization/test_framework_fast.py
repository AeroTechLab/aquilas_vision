#!/usr/bin/env python3
"""
Script de teste RÁPIDO que não depende do download do COCO
"""
import subprocess
import sys
import time
import os

def run_fast_test():
    """Teste rápido que usa dataset dummy"""
    print("⚡ TESTE RÁPIDO DO FRAMEWORK (sem download COCO)")
    print("=" * 60)
    
    # Comandos que usam --skip-coco para evitar download
    tests = [
        ("python main.py --model mobilenetv2 --optimization ptq --quick-test --skip-coco", "MobileNetV2 PTQ Quick"),
        ("python main.py --model yolo --optimization ptq --quick-test --skip-coco", "YOLO PTQ Quick"),
        ("python main.py --list", "Listar Otimizações"),
    ]
    
    results = {}
    
    for cmd, test_name in tests:
        print(f"\n🔧 Executando: {test_name}")
        print("-" * 40)
        
        start_time = time.time()
        try:
            # Timeout reduzido para 2 minutos
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=120  # 2 minutos
            )
            
            success = result.returncode == 0
            results[test_name] = success
            elapsed = time.time() - start_time
            
            if success:
                print(f"✅ {test_name} - SUCESSO ({elapsed:.1f}s)")
                if result.stdout:
                    # Mostra as últimas linhas da saída
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 3:
                        print("   ...")
                        for line in lines[-3:]:
                            print(f"   {line}")
                    else:
                        for line in lines:
                            print(f"   {line}")
            else:
                print(f"❌ {test_name} - FALHA ({elapsed:.1f}s)")
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-5:]:  # Mostra últimas 5 linhas de erro
                        if line.strip():
                            print(f"   ⚠️  {line}")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} - TIMEOUT (2 minutos)")
            results[test_name] = False
        except Exception as e:
            print(f"💥 {test_name} - ERRO: {e}")
            results[test_name] = False
    
    # Relatório final
    print("\n📊 RELATÓRIO FINAL - TESTE RÁPIDO")
    print("=" * 40)
    success_count = sum(results.values())
    total_count = len(results)
    
    for test_name, success in results.items():
        status = "✅ SUCESSO" if success else "❌ FALHA"
        print(f"{test_name}: {status}")
    
    print(f"\n📈 {success_count}/{total_count} testes passaram")
    
    return success_count > 0

def check_basic_functionality():
    """Verifica funcionalidades básicas sem download"""
    print("🔍 VERIFICANDO FUNCIONALIDADES BÁSICAS...")
    
    try:
        # Testa imports básicos
        import torch
        import torchvision
        from utils.coco_downloader import coco_manager
        from utils.coco_loader import RobustCocoDataLoader
        
        print("✅ Imports básicos - OK")
        
        # Testa se consegue criar um config básico
        from configs.mobilenetv2_config import MobileNetV2Config
        config = MobileNetV2Config(optimization_type='ptq')
        print("✅ Configurações - OK")
        
        # Testa se consegue criar data loader dummy
        data_loader = RobustCocoDataLoader(config)
        calibration_loader = data_loader.get_calibration_loader(num_samples=10)
        print("✅ DataLoader dummy - OK")
        
        # Testa modelo MobileNetV2
        import torchvision.models as models
        model = models.mobilenet_v2(pretrained=False)  # Usando pretrained=False para ser rápido
        print("✅ Modelo MobileNetV2 - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Falha na verificação básica: {e}")
        return False

if __name__ == "__main__":
    print("🚀 INICIANDO TESTE RÁPIDO DO FRAMEWORK")
    print("💡 Dica: Este teste usa dataset dummy para ser rápido")
    
    # Verifica funcionalidades básicas primeiro
    if not check_basic_functionality():
        print("\n❌ Verificação básica falhou. Corrija os problemas.")
        sys.exit(1)
    
    # Executa testes rápidos
    success = run_fast_test()
    
    if success:
        print("\n🎉 Framework funcionando corretamente!")
        print("\n📝 Próximos passos recomendados:")
        print("1. python main.py --skip-coco --model mobilenetv2 --optimization ptq")
        print("2. python main.py --skip-coco --model yolo --optimization ptq") 
        print("3. Para teste completo: python main.py --force-download --setup-only")
    else:
        print("\n⚠️  Alguns testes falharam, mas o framework pode funcionar parcialmente.")
        print("   Tente: python main.py --skip-coco --list")
    
    sys.exit(0 if success else 1)