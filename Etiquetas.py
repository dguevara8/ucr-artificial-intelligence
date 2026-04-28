from PIL import Image
import numpy as np
import os
import csv

input_folder = "fotos_originales"
output_folder = "fotos_procesadas"

# Crear subcarpetas de salida
os.makedirs(os.path.join(output_folder, "positivas"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "negativas"), exist_ok=True)

categorias = {
    "positivas": 1,
    "negativas": 0
}

rows = []

for carpeta, etiqueta in categorias.items():
    carpeta_input = os.path.join(input_folder, carpeta)
    carpeta_output = os.path.join(output_folder, carpeta)

    for filename in sorted(os.listdir(carpeta_input)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Abrir, convertir y guardar imagen procesada
        img = Image.open(os.path.join(carpeta_input, filename))
        img = img.convert("L")

        # Recorte central (zoom) — toma solo el 60% del centro
        w, h = img.size
        margen_w = int(w * 0.20)  # recorta 20% de cada lado horizontal
        margen_h = int(h * 0.20)  # recorta 20% de cada lado vertical
        img = img.crop((margen_w, margen_h, w - margen_w, h - margen_h))

        img = img.resize((128, 128))
        img = img.point(lambda x: 255 if x >= 130 else 0, "L")
        img.save(os.path.join(carpeta_output, filename))  # guarda en subcarpeta correcta

        # Convertir a vector binario
        arr = np.array(img)
        binary = (arr >= 130).astype(int)
        vector = binary.flatten().tolist() + [etiqueta]
        rows.append(vector)

# Guardar CSV
with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Dataset generado: {len(rows)} imágenes.")
print(f"  Positivas: {sum(1 for r in rows if r[-1] == 1)}")
print(f"  Negativas: {sum(1 for r in rows if r[-1] == 0)}")