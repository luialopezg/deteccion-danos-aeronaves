import torch
import clip
from PIL import Image
import os
import time
import re
import matplotlib.pyplot as plt

# Verificar si tenemos GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el modelo CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# Ruta de las imágenes
image_path = "../data/raw/train/train"
image_files = [f for f in os.listdir(image_path) if f.endswith((".jpg", ".png"))]

if len(image_files) < 5:
    print("❌ No hay suficientes imágenes en la carpeta para la prueba (mínimo 5).")
    exit()

# Seleccionar 5 imágenes
selected_images = image_files[:5]

# Diccionario de traducción (palabras clave del nombre del archivo)
translation_dict = {
    "Cessna": "Cessna",
    "Beechcraft": "Beechcraft",
    "Boeing": "Boeing",
    "Wing": "Ala",
    "Structure": "Estructura",
    "Rudder": "Timón",
    "Dent": "Abolladura",
    "Damage": "Daño",
    "Trim": "Recorte",
    "Assembly": "Ensamble",
    "Fuselage": "Fuselaje",
    "Hailstorm": "Granizo",
    "Minus": "Menos",
    "Core": "Núcleo"
}

# Función para extraer palabras clave y traducirlas
def extract_keywords(filename):
    filename = filename.replace("_", " ").replace("-", " ")  # Normalizar separadores
    keywords = re.findall(r'\b(' + '|'.join(translation_dict.keys()) + r')\b', filename, re.IGNORECASE)
    translated_keywords = [translation_dict.get(k, k) for k in keywords]  # 🔹 Usa get() para evitar KeyError
    return " ".join(set(translated_keywords)) if translated_keywords else "Desconocido"


# Definir descripciones en español
text_descriptions = [
    "Abolladura en el fuselaje",
    "Grieta en el ala",
    "Corrosión en el motor",
    "Rasguño superficial",
    "Impacto con objeto externo",
    "Daño estructural severo"
]

# Tokenizar descripciones
text_tokens = clip.tokenize(text_descriptions).to(device)

# Medir tiempo de ejecución
start_time = time.time()

# Procesar imágenes
results = []
for img_name in selected_images:
    img_path = os.path.join(image_path, img_name)
    image = Image.open(img_path)
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)

    # Extraer y traducir palabras clave del nombre del archivo
    file_keywords = extract_keywords(img_name)

    # Ajustar las descripciones de CLIP con la información extraída
    enhanced_descriptions = [f"{desc} ({file_keywords})" for desc in text_descriptions]

    # Tokenizar nuevas descripciones
    enhanced_text_tokens = clip.tokenize(enhanced_descriptions).to(device)

    # Calcular similitud entre la imagen y las descripciones
    with torch.no_grad():
        image_features = model.encode_image(image_preprocessed)
        text_features = model.encode_text(enhanced_text_tokens)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    # Seleccionar la descripción más probable
    best_match = enhanced_descriptions[similarity.argmax()]
    
    # Guardar resultado
    results.append((img_name, best_match))

    # Mostrar imagen con su predicción
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Descripción generada: {best_match}", fontsize=12, color="blue")
    plt.savefig(f"../data/reports/{img_name}_output.png")
    plt.close()
    plt.show()

# Medir tiempo final
end_time = time.time()
execution_time = end_time - start_time

# Imprimir resultados en la terminal
print("\n✅ Resultados de la prueba con 5 imágenes (traducción al español):")
for img_name, desc in results:
    print(f"📌 {img_name}: {desc}")

print(f"\n⏳ Tiempo total de ejecución: {execution_time:.2f} segundos")
