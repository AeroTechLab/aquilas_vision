#!/bin/bash

# Script para download automático do dataset COCO
# Autor: Framework de Otimização
# Versão: 1.0

set -e  # Para em caso de erro

# Configurações
COCO_DIR="./datasets/coco"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TRAIN_IMAGES_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para log
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Função para verificar se o comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função para verificar integridade do ZIP
check_zip_integrity() {
    local zip_file="$1"
    log_info "Verificando integridade de $zip_file..."
    
    if ! unzip -t "$zip_file" >/dev/null 2>&1; then
        log_error "Arquivo ZIP corrompido: $zip_file"
        return 1
    fi
    return 0
}

# Função para baixar arquivo com retry
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        log_info "Baixando $output (tentativa $((retry_count + 1))/$max_retries)..."
        
        if command_exists wget; then
            if wget --progress=bar:force:noscroll -c -O "$output" "$url"; then
                if check_zip_integrity "$output"; then
                    log_success "Download completo: $output"
                    return 0
                else
                    log_warning "Arquivo corrompido, excluindo e tentando novamente..."
                    rm -f "$output"
                fi
            fi
        elif command_exists curl; then
            if curl -L -C - -o "$output" "$url"; then
                if check_zip_integrity "$output"; then
                    log_success "Download completo: $output"
                    return 0
                else
                    log_warning "Arquivo corrompido, excluindo e tentando novamente..."
                    rm -f "$output"
                fi
            fi
        else
            log_error "Nem wget nem curl encontrados. Instale um deles."
            return 1
        fi
        
        retry_count=$((retry_count + 1))
        log_warning "Tentativa $retry_count falhou. Tentando novamente em 5 segundos..."
        sleep 5
    done
    
    log_error "Falha no download após $max_retries tentativas: $url"
    return 1
}

# Função para extrair arquivo
extract_zip() {
    local zip_file="$1"
    local extract_dir="$2"
    
    log_info "Extraindo $zip_file..."
    
    if unzip -q "$zip_file" -d "$extract_dir"; then
        log_success "Extraído: $zip_file"
        return 0
    else
        log_error "Falha ao extrair: $zip_file"
        return 1
    fi
}

# Função para verificar se o dataset já existe
check_dataset_exists() {
    local required_files=(
        "$COCO_DIR/annotations/instances_train2017.json"
        "$COCO_DIR/annotations/instances_val2017.json"
        "$COCO_DIR/train2017"
        "$COCO_DIR/val2017"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -e "$file" ]; then
            return 1
        fi
    done
    
    # Verifica se há imagens nos diretórios
    local train_images_count=$(find "$COCO_DIR/train2017" -name "*.jpg" | wc -l 2>/dev/null)
    local val_images_count=$(find "$COCO_DIR/val2017" -name "*.jpg" | wc -l 2>/dev/null)
    
    if [ "$train_images_count" -eq 0 ] || [ "$val_images_count" -eq 0 ]; then
        log_warning "Diretórios existem mas estão vazios ou sem imagens .jpg"
        return 1
    fi
    
    log_success "Dataset COCO já existe e parece intacto"
    log_info "Train images: $train_images_count"
    log_info "Val images: $val_images_count"
    return 0
}

# Função principal
main() {
    log_info "Iniciando verificação e download do dataset COCO..."
    
    # Criar diretório se não existir
    mkdir -p "$COCO_DIR"
    
    # Verificar se dataset já existe
    if check_dataset_exists; then
        log_success "Dataset COCO já está disponível. Nada a fazer."
        exit 0
    fi
    
    log_warning "Dataset COCO não encontrado ou incompleto. Iniciando download..."
    
    # Download das anotações
    log_info "Baixando anotações..."
    if ! download_with_retry "$ANNOTATIONS_URL" "$COCO_DIR/annotations_trainval2017.zip"; then
        log_error "Falha no download das anotações"
        exit 1
    fi
    
    # Download das imagens de treino
    log_info "Baixando imagens de treino..."
    if ! download_with_retry "$TRAIN_IMAGES_URL" "$COCO_DIR/train2017.zip"; then
        log_error "Falha no download das imagens de treino"
        exit 1
    fi
    
    # Download das imagens de validação
    log_info "Baixando imagens de validação..."
    if ! download_with_retry "$VAL_IMAGES_URL" "$COCO_DIR/val2017.zip"; then
        log_error "Falha no download das imagens de validação"
        exit 1
    fi
    
    # Extrair arquivos
    log_info "Extraindo arquivos..."
    
    # Extrair anotações
    if ! extract_zip "$COCO_DIR/annotations_trainval2017.zip" "$COCO_DIR"; then
        log_error "Falha ao extrair anotações"
        exit 1
    fi
    
    # Extrair imagens de treino
    if ! extract_zip "$COCO_DIR/train2017.zip" "$COCO_DIR"; then
        log_error "Falha ao extrair imagens de treino"
        exit 1
    fi
    
    # Extrair imagens de validação
    if ! extract_zip "$COCO_DIR/val2017.zip" "$COCO_DIR"; then
        log_error "Falha ao extrair imagens de validação"
        exit 1
    fi
    
    # Limpar arquivos ZIP (opcional)
    log_info "Limpando arquivos temporários..."
    rm -f "$COCO_DIR/annotations_trainval2017.zip"
    rm -f "$COCO_DIR/train2017.zip" 
    rm -f "$COCO_DIR/val2017.zip"
    
    # Verificar estrutura final
    log_info "Verificando estrutura final..."
    if check_dataset_exists; then
        log_success "Dataset COCO baixado e extraído com sucesso!"
        log_info "Local: $COCO_DIR"
        log_info "Estrutura:"
        find "$COCO_DIR" -type d | sort | sed 's/^/    /'
    else
        log_error "Algo deu errado na verificação final"
        exit 1
    fi
}

# Verificar dependências
check_dependencies() {
    local missing_deps=()
    
    if ! command_exists unzip; then
        missing_deps+=("unzip")
    fi
    
    if ! command_exists wget && ! command_exists curl; then
        missing_deps+=("wget ou curl")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Dependências missing: ${missing_deps[*]}"
        log_info "Instale com:"
        if [[ " ${missing_deps[*]} " =~ "unzip" ]]; then
            echo "    sudo apt-get install unzip"
        fi
        if [[ " ${missing_deps[*]} " =~ "wget" ]]; then
            echo "    sudo apt-get install wget"
        fi
        exit 1
    fi
}

# Handler para interrupção
cleanup() {
    log_warning "Script interrompido. Limpando..."
    # Remove arquivos ZIP parciais se existirem
    find "$COCO_DIR" -name "*.zip" -size -10G -exec rm -f {} + 2>/dev/null || true
    exit 1
}

# Configurar trap para Ctrl+C
trap cleanup INT TERM

check_dependencies
main