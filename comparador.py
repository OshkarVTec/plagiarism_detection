import pandas as pd
import time
import tracemalloc
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
    accuracy_score,
)
import subprocess
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from plagiarism_difflib import detect_clone_type

DATASET_PATH = "dataset_test"


# Función para medir tiempo y memoria de una función dada
def run_with_profiler(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = end_time - start_time
    mem_peak_mb = peak / 10**6
    return result, time_taken, mem_peak_mb


# Funciones adaptadas para ejecutar ambos detectores y devolver DataFrame con pares detectados
# **Debes adaptar deckard.py y Nuestro.py para exportar funciones que hagan lo siguiente**
def normalize_pairs(df):
    file1_new = []
    file2_new = []
    for f1, f2 in zip(df["file1"], df["file2"]):
        if f1 < f2:
            file1_new.append(f1)
            file2_new.append(f2)
        else:
            file1_new.append(f2)
            file2_new.append(f1)
    df["file1"] = file1_new
    df["file2"] = file2_new
    return df


def extract_code_fragment(filepath, start, end):
    filepath = DATASET_PATH + "/" + filepath
    with open(filepath, "r") as f:
        lines = f.readlines()
    # Ajusta los índices si tus líneas son 1-based
    return "".join(lines[start - 1 : end])


def update_predicted_label(df):
    types = []
    for idx, row in df.iterrows():
        code1 = extract_code_fragment(
            row["file1"], row["line_start1"], row["line_end1"]
        )
        code2 = extract_code_fragment(
            row["file2"], row["line_start2"], row["line_end2"]
        )
        tipo = detect_clone_type(code1, code2)
        types.append(tipo)
    df["predicted_label"] = types
    return df


def run_deckard_detector(DATASET_PATH):
    """
    Ejecuta el detector deckard.py y devuelve DataFrame con columnas:
    file1, line_start1, line_end1, file2, line_start2, line_end2, predicted_label=1
    """
    import deckard  # asumiendo deckard.py está modularizado

    clones = deckard.run_deckard_on_directory(
        DATASET_PATH, min_size=5, window_size=3, min_dist=5.0, k=5, L=10, w=4.0
    )
    rows = []
    # print(f"Tipo de clones: {type(clones)}")
    for idx, item in enumerate(clones):
        # print(f"Clone #{idx}: {item}")
        try:
            (file1, line1), (file2, line2), dist = item
            # print( f"file1={file1}, line1={line1}, file2={file2}, line2={line2}, dist={dist}")
            file1 = "/".join(file1.split("/")[1:])
            file2 = "/".join(file2.split("/")[1:])
            rows.append(
                {
                    "file1": file1,
                    "line_start1": line1,
                    "line_end1": line1,
                    "file2": file2,
                    "line_start2": line2,
                    "line_end2": line2,
                    "predicted_label": 1,
                }
            )
        except Exception as e:
            # print(f"Error al desempaquetar clone #{idx}: {e}")
            break
    df = pd.DataFrame(rows)
    return df


def run_nuestro_detector(DATASET_PATH):
    """
    Ejecuta el detector Nuestro.py y devuelve DataFrame con columnas similares.
    """
    import plagiarism_clusters  # asumiendo nuestro.py modularizado

    files = plagiarism_clusters.collect_py_files(DATASET_PATH)
    # rint("NUESTRO")
    # print(files)
    file_map, line_ranges, clones, labels = plagiarism_clusters.detect_clones(
        files, min_nodes=30, window=1, stride=1, radius=0.05, length_tol=0.2
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
    df = pd.DataFrame(rows)
    df = update_predicted_label(df)
    return df


def evaluate_deckard(df_deckard, df_true):
    df_deckard = normalize_pairs(df_deckard)
    df_true = normalize_pairs(df_true)
    # Eliminar duplicados donde (file1, file2) == (file2, file1)
    df_deckard["pair"] = df_deckard.apply(
        lambda row: tuple(sorted([row["file1"], row["file2"]])), axis=1
    )
    df_deckard = df_deckard.drop_duplicates(subset="pair").drop(columns="pair")
    merge_cols = [
        "file1",
        "file2",
    ]
    merged = pd.merge(
        df_true, df_deckard, on=merge_cols, how="left", suffixes=("_true", "_pred")
    )
    # Si no hay predicción, es 0 (no detectado)
    merged["predicted_label"] = merged["predicted_label"].fillna(0).astype(int)
    y_true = merged["true_label"]
    y_pred = merged["predicted_label"]
    print(y_pred)
    print(y_true)
    y_true = y_true.apply(lambda x: 1 if x > 1 else x)
    labels = [0, 1]  # 0 = no plagio, 1 = plagio
    evaluate(y_true, y_pred, labels, "deckard")


def evaluate_nuestro(df_nuestro, df_true):
    df_nuestro = normalize_pairs(df_nuestro)
    df_true = normalize_pairs(df_true)
    idx = df_nuestro.groupby(["file1", "file2"])["predicted_label"].idxmax()
    df_nuestro = df_nuestro.loc[idx].reset_index(drop=True)
    merge_cols = [
        "file1",
        "file2",
    ]
    merged = pd.merge(
        df_true, df_nuestro, on=merge_cols, how="left", suffixes=("_true", "_pred")
    )
    # Si no hay predicción, es 0 (no detectado)
    merged["predicted_label"] = merged["predicted_label"].fillna(0).astype(int)
    y_true = merged["true_label"]
    y_pred = merged["predicted_label"]
    labels = [0, 1, 2, 3]
    evaluate(y_true, y_pred, labels, "nuestro")


def evaluate(y_true, y_pred, labels, name):
    print("Matriz de Confusión:")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()
    print(f"F1 weighted: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Reporte completo:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


def main():
    csv_etiquetado = "plagiarism_pairs.csv"

    # Carga etiquetas verdaderas
    df_true = pd.read_csv(csv_etiquetado)

    # Ejecutar Deckard
    print("Ejecutando Deckard...")
    deckard_df, deckard_time, deckard_mem = run_with_profiler(
        run_deckard_detector, DATASET_PATH
    )
    print(f"Deckard tiempo: {deckard_time:.2f} s, pico memoria: {deckard_mem:.2f} MB")

    # Ejecutar Nuestro
    print("Ejecutando Nuestro...")
    nuestro_df, nuestro_time, nuestro_mem = run_with_profiler(
        run_nuestro_detector, DATASET_PATH
    )
    print(f"Nuestro tiempo: {nuestro_time:.2f} s, pico memoria: {nuestro_mem:.2f} MB")

    print("Resultados de Deckard:")
    print(deckard_df.head())
    print("Resultados de Nuestro:")
    print(nuestro_df.head())

    evaluate_deckard(deckard_df, df_true)
    evaluate_nuestro(nuestro_df, df_true)


if __name__ == "__main__":
    main()
