import torch
import clip
from PIL import Image
import os
import matplotlib.pyplot as plt

# Verificar si tenemos GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el modelo CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# Ruta de la imagen de prueba
image_path = "../data/raw/train/train"
image_files = [f for f in os.listdir(image_path) if f.endswith((".jpg", ".png"))]

if not image_files:
    print("❌ No se encontraron imágenes en la carpeta:", image_path)
    exit()

image_file = os.path.join(image_path, image_files[0])  # Tomar la primera imagen

# Cargar y preprocesar la imagen
image = Image.open(image_file)
image_preprocessed = preprocess(image).unsqueeze(0).to(device)

# Definir posibles descripciones del daño
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

# Calcular similitud entre la imagen y las descripciones
with torch.no_grad():
    image_features = model.encode_image(image_preprocessed)
    text_features = model.encode_text(text_tokens)
    similarity = (image_features @ text_features.T).softmax(dim=-1)

# Seleccionar la descripción más probable
best_match = text_descriptions[similarity.argmax()]

# Mostrar la imagen con la descripción
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.title(f"Descripción generada: {best_match}", fontsize=12, color="blue")
plt.show()

# Imprimir resultado en la terminal también
print(f"✅ Descripción generada para {image_file}: {best_match}")
import torch
import clip
from PIL import Image
import os

# Verificar si tenemos GPU disponible (no la usaremos, pero lo dejamos por compatibilidad)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el modelo CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# Ruta de la imagen de prueba
image_path = "../data/raw/train/train"
image_files = [f for f in os.listdir(image_path) if f.endswith((".jpg", ".png"))]

if not image_files:
    print("❌ No se encontraron imágenes en la carpeta:", image_path)
    exit()

image_file = os.path.join(image_path, image_files[0])  # Tomar la primera imagen

# Cargar y preprocesar la imagen
image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)

# Definir posibles descripciones del daño
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

# Calcular similitud entre la imagen y las descripciones
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
    similarity = (image_features @ text_features.T).softmax(dim=-1)

# Seleccionar la descripción más probable
best_match = text_descriptions[similarity.argmax()]

# Mostrar resultado
print(f"✅ Descripción generada para {image_file}: {best_match}")
