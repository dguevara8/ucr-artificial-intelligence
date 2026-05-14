# Model Card — Clasificador de Contaminaciones en Línea de Producción Simulada

## Información general

Este modelo fue desarrollado para clasificar imágenes de una línea de producción simulada y detectar la presencia de granos de arroz. El proyecto utiliza aprendizaje automático clásico y trabaja con imágenes procesadas como vectores binarios.

El objetivo principal es distinguir entre:

- **Clase positiva (`1`)**: imagen con granos de arroz.
- **Clase negativa (`0`)**: imagen sin arroz.

---

## Modelos evaluados

Se compararon cuatro modelos clásicos de clasificación supervisada: Árbol de Decisión, Naive Bayes, KNN y SVM. Estos algoritmos fueron elegidos porque representan distintas formas de aprender patrones a partir de datos etiquetados.

### Árbol de Decisión

Un Árbol de Decisión clasifica los datos mediante una estructura de preguntas o divisiones. En cada nodo, el modelo selecciona una característica que ayuda a separar mejor las clases. En este proyecto, las características corresponden a los píxeles binarizados de cada imagen.

Este modelo es fácil de interpretar, pero puede ser sensible a pequeñas variaciones en los datos. Si el árbol crece demasiado, puede memorizar detalles del conjunto de entrenamiento y perder capacidad de generalización.

### Naive Bayes

Naive Bayes es un clasificador probabilístico basado en el teorema de Bayes. Calcula la probabilidad de que una imagen pertenezca a una clase según los valores de sus características.

Se le llama “naive” porque asume que las características son independientes entre sí. En imágenes, esta suposición puede ser limitada, ya que los píxeles cercanos suelen estar relacionados. Por esta razón, en este proyecto Naive Bayes obtuvo un desempeño menor que los otros modelos.

### KNN

KNN, o K-Nearest Neighbors, clasifica una muestra observando las muestras más cercanas dentro del conjunto de entrenamiento. Si una imagen nueva se parece más a imágenes etiquetadas como positivas, el modelo la clasifica como positiva.

Este modelo depende mucho de la métrica de distancia y del número de vecinos seleccionados. En imágenes con muchos píxeles, la distancia entre vectores puede volverse menos representativa, especialmente si hay ruido o variaciones de iluminación.

### SVM

SVM, o Máquina de Vectores de Soporte, busca construir una frontera que separe las clases con el mayor margen posible. En este proyecto, el mejor resultado se obtuvo con SVM usando kernel RBF.

El kernel RBF permite que el modelo capture relaciones no lineales entre los datos. Esto es útil porque las imágenes con arroz y sin arroz no necesariamente se separan de forma simple cuando se representan como vectores de píxeles.

---

El mejor modelo encontrado fue:

| Elemento | Valor |
|---|---|
| Algoritmo | SVM |
| Kernel | `rbf` |
| C | `10` |
| Gamma | `scale` |

Este modelo fue seleccionado porque obtuvo el mejor desempeño general en comparación con Árbol de Decisión, Naive Bayes y KNN.

---

## Métricas de evaluación

Para comparar los modelos se utilizaron cuatro métricas: accuracy, precision, recall y F1-score.

### Accuracy

El accuracy mide el porcentaje total de predicciones correctas. Es decir, indica cuántas imágenes fueron clasificadas correctamente sobre el total de imágenes evaluadas.

Aunque es una métrica útil, puede ser insuficiente cuando existe desbalance entre clases. En este proyecto el dataset está balanceado, pero aun así se consideraron otras métricas para analizar mejor el comportamiento de los modelos.

### Precision

La precision mide qué tan confiables son las predicciones positivas del modelo. En este caso, indica cuántas de las imágenes clasificadas como “con arroz” realmente tenían arroz.

Una precision alta significa que el modelo comete pocos falsos positivos.

### Recall

El recall mide la capacidad del modelo para encontrar correctamente los casos positivos. En este proyecto, indica cuántas imágenes con arroz fueron detectadas correctamente.

Un recall alto significa que el modelo deja escapar pocos casos positivos. Esto es importante porque el objetivo principal es detectar la presencia de arroz.

### F1-score

El F1-score combina precision y recall en una sola métrica. Es útil cuando se desea encontrar un balance entre detectar correctamente los positivos y evitar clasificaciones positivas incorrectas.

En este proyecto se utilizó el F1-score como métrica principal para seleccionar el mejor modelo, ya que permite evaluar de forma más equilibrada el rendimiento general del clasificador.

---

## Datos utilizados

El entrenamiento se realizó con un dataset combinado del grupo.

| Característica | Valor |
|---|---:|
| Total de imágenes | 150 |
| Imágenes positivas | 75 |
| Imágenes negativas | 75 |
| Tamaño procesado | 128 x 128 píxeles |
| Características por imagen | 16,384 |
| Etiqueta | 1 columna final |

Cada imagen fue convertida a escala de grises, recortada, redimensionada y binarizada antes de ser incluida en el dataset.

---

## Preprocesamiento

Cada imagen siguió el siguiente proceso:

1. Conversión a escala de grises.
2. Recorte central, eliminando 20% de cada borde.
3. Redimensionamiento a 128 x 128 píxeles.
4. Binarización con umbral 130.
5. Conversión a vector de 16,384 valores.
6. Adición de etiqueta final:
   - `1` para arroz.
   - `0` para sin arroz.

---

## Modelos evaluados

Se compararon cuatro modelos clásicos de clasificación:

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Árbol de Decisión | 73.33% | 68.42% | 86.67% | 76.47% |
| Naive Bayes | 50.00% | 50.00% | 13.33% | 21.05% |
| KNN | 53.33% | 52.00% | 86.67% | 65.00% |
| SVM | 80.00% | 76.47% | 86.67% | 81.25% |

---

## Evaluación

El dataset se dividió en:

| Conjunto | Porcentaje | Cantidad |
|---|---:|---:|
| Entrenamiento | 80% | 120 imágenes |
| Prueba | 20% | 30 imágenes |

La métrica principal utilizada para seleccionar el modelo fue el **F1-score**, ya que permite balancear la precisión y el recall.

El modelo SVM obtuvo:

| Métrica | Valor |
|---|---:|
| Accuracy | 80.00% |
| Precision | 76.47% |
| Recall | 86.67% |
| F1-score | 81.25% |

---

## Matriz de confusión

La matriz de confusión del mejor modelo fue:

```text
[[11, 4],
 [2, 13]]
```

Interpretación:

| Resultado | Cantidad |
|---|---:|
| Negativas clasificadas correctamente | 11 |
| Negativas clasificadas como positivas | 4 |
| Positivas clasificadas como negativas | 2 |
| Positivas clasificadas correctamente | 13 |

---

## Hiperparámetros seleccionados
Los hiperparámetros son configuraciones que se definen antes del entrenamiento y que influyen en el comportamiento del modelo. A diferencia de los parámetros internos, que el modelo aprende durante el entrenamiento, los hiperparámetros deben ser seleccionados mediante prueba, criterio técnico o búsqueda automática.

En este proyecto se utilizó `GridSearchCV`, que permite probar varias combinaciones de hiperparámetros y seleccionar la mejor según una métrica definida. En este caso, se usó el F1-score como criterio principal.

Los mejores hiperparámetros encontrados para cada modelo fueron:

| Modelo | Hiperparámetros |
|---|---|
| Árbol de Decisión | `criterion=gini`, `max_depth=10`, `min_samples_split=5` |
| Naive Bayes | `alpha=0.5` |
| KNN | `metric=euclidean`, `n_neighbors=7`, `weights=distance` |
| SVM | `C=10`, `gamma=scale`, `kernel=rbf` |

---
### Interpretación de los hiperparámetros principales

En SVM, `C` controla qué tanto se penalizan los errores de clasificación. Un valor mayor intenta clasificar correctamente más ejemplos, aunque puede aumentar el riesgo de ajustarse demasiado al conjunto de entrenamiento. El parámetro `kernel` define la forma de la frontera de decisión. En este caso, `rbf` permite una separación no lineal. El parámetro `gamma` controla la influencia de cada punto sobre la frontera de decisión.

En KNN, `n_neighbors` define cuántos vecinos se toman en cuenta para clasificar una nueva imagen. El parámetro `weights=distance` hace que los vecinos más cercanos tengan mayor influencia. La métrica `euclidean` define cómo se mide la distancia entre vectores.

En Árbol de Decisión, `criterion=gini` indica la medida usada para decidir las divisiones del árbol. `max_depth=10` limita la profundidad máxima del árbol y `min_samples_split=5` indica el mínimo de muestras necesarias para dividir un nodo.

En Naive Bayes, `alpha=0.5` corresponde a un valor de suavizado que ayuda a evitar probabilidades extremas cuando ciertos valores aparecen poco en el dataset.

---

## Limitaciones

El desempeño del modelo puede verse afectado por:

- Cambios fuertes de iluminación.
- Fondos diferentes al usado en el entrenamiento.
- Objetos muy pequeños o parcialmente visibles.
- Diferencias de cámara entre integrantes.
- Posibles inconsistencias en el etiquetado de imágenes.
- Tamaño reducido del dataset.

Además, al convertir las imágenes a blanco y negro mediante binarización, se pierde información visual como textura, sombras y variaciones de intensidad.

---



El entrenamiento utiliza una partición estratificada de 80% entrenamiento y 20% prueba. Además, se utiliza `random_state=42` para mantener resultados consistentes entre ejecuciones.
