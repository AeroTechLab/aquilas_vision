#!/usr/bin/env python3
"""
Script de teste robusto para o framework de otimização
"""
import subprocess
import sys
import time
import os

def run_simple_test():
    """Teste mais simples e robusto"""
    print("🧪 TESTE SIMPLES DO FRAMEWORK")
    print("=" * 50)
    
    tests = [
        # Testes básicos que devem funcionar mesmo sem COCO
        ("python main.py --model mobilenetv2 --optimization ptq --setup-only", "Setup MobileNetV2"),
        ("python main.py --model yolo --optimization ptq --setup-only", "Setup YOLO"),
    ]
    
    results = {}
    
    for cmd, test_name in tests:
        print(f"\n🔧 Executando: {test_name}")
        print("-" * 30)
        
        start_time = time.time()
        try:
            # Executa em subprocess para isolar erros
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minutos timeout
            )
            
            success = result.returncode == 0
            results[test_name] = success
            elapsed = time.time() - start_time
            
            if success:
                print(f"✅ {test_name} - SUCESSO ({elapsed:.1f}s)")
            else:
                print(f"❌ {test_name} - FALHA ({elapsed:.1f}s)")
                if result.stderr:
                    print(f"   Erro: {result.stderr[:500]}...")  # Mostra apenas os primeiros 500 chars
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} - TIMEOUT")
            results[test_name] = False
        except Exception as e:
            print(f"💥 {test_name} - ERRO: {e}")
            results[test_name] = False
    
    # Relatório final
    print("\n📊 RELATÓRIO FINAL")
    print("=" * 30)
    for test_name, success in results.items():
        status = "✅ SUCESSO" if success else "❌ FALHA"
        print(f"{test_name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n📈 {success_count}/{total_count} testes passaram")
    
    return success_count > 0

def check_imports():
    """Verifica se todos os imports básicos funcionam"""
    print("🔍 VERIFICANDO IMPORTS...")
    
    imports_to_check = [
        "torch",
        "torchvision", 
        "torch.nn",
        "utils.coco_downloader",
        "utils.coco_loader",
        "configs.mobilenetv2_config",
        "configs.yolo_config",
    ]
    
    for import_name in imports_to_check:
        try:
            if import_name.startswith("utils.") or import_name.startswith("configs."):
                __import__(import_name)
            else:
                # Para imports de bibliotecas externas
                exec(f"import {import_name}")
            print(f"✅ {import_name}")
        except Exception as e:
            print(f"❌ {import_name}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🚀 INICIANDO VERIFICAÇÃO DO FRAMEWORK")
    
    # Verifica imports primeiro
    if not check_imports():
        print("\n❌ Imports falharam. Corrija as dependências.")
        sys.exit(1)
    
    # Executa testes simples
    success = run_simple_test()
    
    if success:
        print("\n🎉 Framework verificado com sucesso!")
        print("\n📝 Próximos passos:")
        print("1. python main.py --list")
        print("2. python main.py --quick-test") 
        print("3. python main.py --model mobilenetv2 --optimization ptq")
    else:
        print("\n❌ Framework precisa de ajustes.")
    
    sys.exit(0 if success else 1)