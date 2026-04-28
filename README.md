# Proyecto 1 — Clasificación de Contaminaciones en Línea de Producción Simulada

Detección de granos de arroz en imágenes de una línea de producción simulada usando aprendizaje automático clásico.

---

## Estructura del repositorio

```
├── fotos_originales/
│   ├── positivas/        # Fotos originales con granos de arroz
│   └── negativas/        # Fotos originales sin arroz (clips, aros, vacías)
├── fotos_procesadas/
│   ├── positivas/        # Imágenes binarizadas 128x128
│   └── negativas/
├── Etiquetas.py          # Preprocesamiento y generación del CSV
├── dataset.csv           # Dataset final (30 filas × 16385 columnas)
├── ProyectoIA_C23562     # Documentación del conjunto de datos
└── README.md
```

---

## Descripción del problema

Se simula una línea de producción con una hoja blanca. Los objetos que caen sobre ella son:

- **Granos de arroz** → ejemplo **positivo** (contaminación, etiqueta = `1`)
- **Clips** → ejemplo **negativo** (etiqueta = `0`)
- **Hoja vacía** → ejemplo **negativo** (etiqueta = `0`)

---

## Preprocesamiento (`Etiquetas.py`)

Cada imagen pasa por el siguiente pipeline:

1. Conversión a escala de grises
2. Recorte del 20% de cada borde (zoom central para reducir ruido)
3. Redimensionado a **128×128 píxeles**
4. Binarización con umbral 130: píxel ≥ 130 → `1` (fondo), píxel < 130 → `0` (objeto)
5. Aplanado a vector de **16,384 valores** + etiqueta

### Ejecutar preprocesamiento

```bash
pip install pillow numpy
python Etiquetas.py
```

---

## Dataset

| Característica | Valor |
|---|---|
| Total de imágenes | 30 |
| Positivas (arroz) | 15 |
| Negativas | 15 |
| Filas en CSV | 30 |
| Columnas en CSV | 16,385 (16,384 píxeles + etiqueta) |

---

## Requisitos

```bash
pip install pillow numpy scikit-learn joblib
```
