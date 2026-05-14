# Proyecto 1 — Clasificación de Contaminaciones en Línea de Producción Simulada

Detección de granos de arroz en imágenes de una línea de producción simulada usando aprendizaje automático clásico.

---

## Estructura del repositorio

```text
├── fotos_originales/
│   ├── positivas/              # Fotos originales con granos de arroz
│   └── negativas/              # Fotos originales sin arroz
├── fotos_procesadas/
│   ├── positivas/              # Imágenes procesadas y binarizadas
│   └── negativas/
├── Etiquetas.py                # Preprocesamiento y generación del dataset individual
├── Datasets.py                 # Combinación de datasets del grupo
├── Modelos.py                  # Entrenamiento, evaluación y exportación del modelo
├── dataset.csv                 # Dataset individual
├── dataset_grupo.csv           # Dataset combinado del grupo
├── mejor_modelo.pkl            # Mejor modelo exportado
├── resultados_modelos.json     # Resultados de métricas e hiperparámetros
├── ProyectoIA_C23562.pdf       # Documentación del proyecto
└── README.md
```

---

## Descripción del problema

Se simula una línea de producción con una hoja blanca. Los objetos que caen sobre ella son clasificados como:

- **Granos de arroz** → ejemplo positivo, etiqueta = `1`
- **Objetos sin arroz o fondo vacío** → ejemplo negativo, etiqueta = `0`

El objetivo del proyecto es entrenar modelos de clasificación clásica capaces de distinguir imágenes con arroz y sin arroz.

---

## Preprocesamiento (`Etiquetas.py`)

Cada imagen pasa por el siguiente pipeline:

1. Conversión a escala de grises.
2. Recorte del 20% de cada borde para reducir ruido.
3. Redimensionamiento a **128×128 píxeles**.
4. Binarización con umbral 130:
   - píxel >= 130 → `1` / fondo
   - píxel < 130 → `0` / objeto
5. Aplanado a vector de **16,384 valores** más una etiqueta final.

Cada fila del CSV contiene:

```text
pixel_1, pixel_2, ..., pixel_16384, etiqueta
```

---

## Dataset

El dataset individual se genera con `Etiquetas.py`. Posteriormente, los datasets de las personas integrantes del grupo se combinan con `Datasets.py`.

| Característica | Valor |
|---|---:|
| Total de imágenes combinadas | 150 |
| Positivas, arroz | 75 |
| Negativas, sin arroz | 75 |
| Columnas en CSV | 16,385 |
| Características | 16,384 pixeles |
| Etiqueta final | 1 columna |

El archivo final utilizado para entrenamiento es:

```text
dataset_grupo.csv
```

---

## Entrenamiento (`Modelos.py`)

Para el entrenamiento se utilizaron modelos clásicos de clasificación supervisada:

- Árbol de Decisión
- Naive Bayes
- KNN
- SVM

Se aplicó una partición estratificada:

| Conjunto | Porcentaje | Cantidad |
|---|---:|---:|
| Entrenamiento | 80% | 120 imágenes |
| Prueba | 20% | 30 imágenes |

También se utilizó `GridSearchCV` para buscar la mejor combinación de hiperparámetros de cada modelo.

---

## Resultados

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Árbol de Decisión | 73.33% | 68.42% | 86.67% | 76.47% |
| Naive Bayes | 50.00% | 50.00% | 13.33% | 21.05% |
| KNN | 53.33% | 52.00% | 86.67% | 65.00% |
| SVM | 80.00% | 76.47% | 86.67% | 81.25% |

---

## Mejor modelo

El mejor modelo fue **SVM con kernel RBF**.

| Parámetro | Valor |
|---|---|
| Modelo | SVM |
| Kernel | `rbf` |
| C | `10` |
| Gamma | `scale` |
| F1-score | `81.25%` |

El modelo fue exportado como:

```text
mejor_modelo.pkl
```

Los resultados completos se guardaron en:

```text
resultados_modelos.json
```

---

## Reproducibilidad

Para reproducir el proceso completo, ejecutar los scripts en el siguiente orden:

```bash
python Etiquetas.py
python Datasets.py
python Modelos.py
```

El flujo genera primero el dataset individual, luego combina los datasets del grupo y finalmente entrena los modelos, calcula métricas y exporta el mejor modelo.

---

## Requisitos

Instalar las dependencias necesarias:

```bash
pip install pillow numpy scikit-learn joblib
```

---
