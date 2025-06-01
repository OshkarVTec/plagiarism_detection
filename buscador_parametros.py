import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from comparador import normalize_pairs, update_predicted_label
import plagiarism_clusters
import itertools
import random
import os

DATASET_PATH = "dataset_4"
CSV_ETIQUETADO = "plagiarism_pairs.csv"
NUMBER_OF_EXPERIMENTS = 10

# Parámetros a explorar
min_nodes_list = [20, 30, 40]
window_list = [3, 5, 7]
stride_list = [3, 5]
radius_list = [0.01, 0.05, 0.1]
length_tol_list = [0.1, 0.2]

# Cargar etiquetas verdaderas
df_true = pd.read_csv(CSV_ETIQUETADO)
df_true = normalize_pairs(df_true)

resultados = []


def run_detector_and_evaluate(params):
    files = plagiarism_clusters.collect_py_files(DATASET_PATH)
    file_map, line_ranges, clones, labels = plagiarism_clusters.detect_clones(
        files,
        min_nodes=params["min_nodes"],
        window=params["window"],
        stride=params["stride"],
        radius=params["radius"],
        length_tol=params["length_tol"],
    )
    rows = []
    for i, j in clones:
        f1 = file_map[i]
        s1, e1 = line_ranges[i]
        f2 = file_map[j]
        s2, e2 = line_ranges[j]
        f1 = "/".join(f1.split("/")[1:])
        f2 = "/".join(f2.split("/")[1:])
        rows.append(
            {
                "file1": f1,
                "line_start1": s1,
                "line_end1": e1,
                "file2": f2,
                "line_start2": s2,
                "line_end2": e2,
                "predicted_label": 1,
            }
        )
    df_pred = pd.DataFrame(rows)
    df_pred = normalize_pairs(df_pred)
    df_pred = df_pred.drop_duplicates(subset=["file1", "file2"]).reset_index(drop=True)
    df_pred = update_predicted_label(df_pred)

    # Evaluación
    merged = pd.merge(
        df_true, df_pred, on=["file1", "file2"], how="left", suffixes=("_true", "_pred")
    )
    merged["predicted_label"] = merged["predicted_label"].fillna(0).astype(int)
    y_true = merged["true_label"]
    y_pred = merged["predicted_label"]
    labels = [0, 1, 2, 3]
    f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return f1, acc, cm, y_true, y_pred


# Seleccionar solo 10 combinaciones aleatorias de parámetros
param_grid = list(
    itertools.product(
        min_nodes_list, window_list, stride_list, radius_list, length_tol_list
    )
)
random.seed(42)  # Para reproducibilidad
sampled_params = random.sample(param_grid, NUMBER_OF_EXPERIMENTS)

# Búsqueda de parámetros (máximo 10 experimentos)
exp_id = 0
for min_nodes, window, stride, radius, length_tol in sampled_params:
    params = {
        "min_nodes": min_nodes,
        "window": window,
        "stride": stride,
        "radius": radius,
        "length_tol": length_tol,
    }
    print(f"Ejecutando experimento {exp_id} con parámetros: {params}")
    f1, acc, cm, y_true, y_pred = run_detector_and_evaluate(params)
    resultados.append(
        {
            "exp_id": exp_id,
            "min_nodes": min_nodes,
            "window": window,
            "stride": stride,
            "radius": radius,
            "length_tol": length_tol,
            "f1_weighted": f1,
            "accuracy": acc,
        }
    )
    # Graficar matriz de confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[0, 1, 2, 3],
        yticklabels=[0, 1, 2, 3],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    output_dir = "confusion_matrices"
    os.makedirs(output_dir, exist_ok=True)
    plt.title(f"Confusion Matrix Exp {exp_id}")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_exp_{exp_id}.png"))
    plt.close()
    exp_id += 1

# Guardar resultados en DataFrame
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_parametros.csv", index=False)

# Graficar F1 weighted y Accuracy para cada experimento
plt.figure(figsize=(10, 5))
plt.plot(
    df_resultados["exp_id"],
    df_resultados["f1_weighted"],
    marker="o",
    label="F1 Weighted",
)
plt.plot(
    df_resultados["exp_id"], df_resultados["accuracy"], marker="x", label="Accuracy"
)
plt.xlabel("Experimento")
plt.ylabel("Score")
plt.title("F1 Weighted y Accuracy por Experimento")
plt.legend()
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "f1_accuracy_parametros.png"))
plt.close()
