import csv
import os
import sys

# Aumentar limite de lectura para filas muy grandes
csv.field_size_limit(10_000_000)

archivos = [
    "dataset.csv",
    "dataset1.csv",
    "dataset2.csv",
    "dataset3.csv",
    "dataset4.csv"
]

filas = []

for archivo in archivos:
    if not os.path.exists(archivo):
        print(f"No existe: {archivo}")
        continue

    print(f"Leyendo: {archivo}")

    with open(archivo, "r", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            # Ignorar encabezados como: pixel_0, pixel_1, ..., label
            try:
                etiqueta = int(float(row[-1]))
            except ValueError:
                print(f"Encabezado ignorado en: {archivo}")
                continue

            # Verificar que la etiqueta sea valida
            if etiqueta not in [0, 1]:
                print(f"Fila ignorada en {archivo}: etiqueta invalida {etiqueta}")
                continue

            # Verificar que tenga 16384 pixeles + 1 etiqueta
            if len(row) == 16385:
                filas.append(row)
            else:
                print(f"Fila ignorada en {archivo}, columnas: {len(row)}")

with open("dataset_grupo.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(filas)

print(f"\nDataset combinado generado: {len(filas)} imágenes.")
print(f"Positivas: {sum(1 for r in filas if int(float(r[-1])) == 1)}")
print(f"Negativas: {sum(1 for r in filas if int(float(r[-1])) == 0)}")
