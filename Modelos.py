import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Leer dataset combinado del grupo
# Cada fila tiene 16384 pixeles + 1 etiqueta final
data = np.loadtxt("dataset_grupo.csv", delimiter=",")

X = data[:, :-1]
y = data[:, -1].astype(int)

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Validacion cruzada 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    "decision_tree": {
        "pipeline": Pipeline([
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "params": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10]
        }
    },

    "naive_bayes": {
        "pipeline": Pipeline([
            ("model", BernoulliNB())
        ]),
        "params": {
            "model__alpha": [0.1, 0.5, 1.0, 2.0]
        }
    },

    "knn": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]
        }
    },

    "svm": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42))
        ]),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        }
    }
}

resultados = []
mejor_modelo = None
mejor_nombre = None
mejor_f1 = -1


def porcentaje(valor):
    return f"{valor * 100:.2f}%"


def limpiar_parametros(parametros):
    parametros_limpios = {}

    for nombre, valor in parametros.items():
        nombre_limpio = nombre.replace("model__", "")
        parametros_limpios[nombre_limpio] = valor

    return parametros_limpios


def parametros_a_texto(parametros):
    parametros_limpios = limpiar_parametros(parametros)
    partes = []

    for nombre, valor in parametros_limpios.items():
        partes.append(f"{nombre}={valor}")

    return ", ".join(partes)

for nombre, config in modelos.items():
    grid = GridSearchCV(
        config["pipeline"],
        config["params"],
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)

    resultado = {
        "modelo": nombre,
        "mejores_parametros": grid.best_params_,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "matriz_confusion": matriz.tolist()
    }

    resultados.append(resultado)

    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_modelo = grid.best_estimator_
        mejor_nombre = nombre

# Exportar mejor modelo
joblib.dump(mejor_modelo, "mejor_modelo.pkl")

# Guardar resultados de los experimentos
with open("resultados_modelos.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)

print("\nResultados")
print("-" * 78)
print(f"{'Modelo':<18} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-score':>12}")
print("-" * 78)

for resultado in resultados:
    print(
        f"{resultado['modelo']:<18} "
        f"{porcentaje(resultado['accuracy']):>12} "
        f"{porcentaje(resultado['precision']):>12} "
        f"{porcentaje(resultado['recall']):>12} "
        f"{porcentaje(resultado['f1']):>12}"
    )

print("-" * 78)

print("\nMejores hiperparametros por modelo")
print("-" * 92)
print(f"{'Modelo':<18} {'Hiperparametros seleccionados':<70}")
print("-" * 92)

for resultado in resultados:
    print(
        f"{resultado['modelo']:<18} "
        f"{parametros_a_texto(resultado['mejores_parametros']):<70}"
    )

print("-" * 92)

print("\nMatriz de confusion por modelo")
print("-" * 50)

for resultado in resultados:
    matriz = resultado["matriz_confusion"]

    print(f"\n{resultado['modelo']}")
    print(f"[[{matriz[0][0]}, {matriz[0][1]}],")
    print(f" [{matriz[1][0]}, {matriz[1][1]}]]")

print("-" * 50)

print("\nMejor modelo:", mejor_nombre)
print("Mejor F1-score:", porcentaje(mejor_f1))
print("Modelo guardado en mejor_modelo.pkl")
print("Resultados guardados en resultados_modelos.json")
