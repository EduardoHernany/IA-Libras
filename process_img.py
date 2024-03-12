import cv2
import os
from pathlib import Path

def process_image(image_path, output_path):
    # Ler a imagem
    img = cv2.imread(image_path)
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar o filtro de Canny
    edges = cv2.Canny(gray, 100, 200)
    # Converter as bordas detectadas para uma máscara de 3 canais
    mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Remover o fundo
    img_no_bg = cv2.bitwise_and(img, mask)
    # Salvar a imagem processada
    cv2.imwrite(output_path, img_no_bg)

def process_dataset(input_dir, output_dir):
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    
    for class_dir in input_dir_path.iterdir():
        if class_dir.is_dir():
            # Criar o diretório correspondente no diretório de saída
            class_output_dir = output_dir_path / class_dir.name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            for image_path in class_dir.iterdir():
                if image_path.is_file():
                    # Definir o caminho de saída para a imagem processada
                    output_path = class_output_dir / image_path.name
                    # Processar a imagem
                    process_image(str(image_path), str(output_path))
                    print(f"Processed {image_path} -> {output_path}")

# Definir o diretório de entrada e de saída
input_dir = '/home/eduardo/Documentos/IA/test'
output_dir = '/home/eduardo/Documentos/IA/test_process'

process_dataset(input_dir, output_dir)