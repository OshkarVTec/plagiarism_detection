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


def run_deckard_detector(dataset_path):
    """
    Ejecuta el detector deckard.py y devuelve DataFrame con columnas:
    file1, line_start1, line_end1, file2, line_start2, line_end2, predicted_label=1
    """
    import deckard  # asumiendo deckard.py está modularizado

    clones = deckard.run_deckard_on_directory(
        dataset_path, min_size=5, window_size=3, min_dist=5.0, k=5, L=10, w=4.0
    )
    rows = []
    # print(f"Tipo de clones: {type(clones)}")
    for idx, item in enumerate(clones):
        # print(f"Clone #{idx}: {item}")
        try:
            (file1, line1), (file2, line2), dist = item
            # print( f"file1={file1}, line1={line1}, file2={file2}, line2={line2}, dist={dist}")

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


def run_nuestro_detector(dataset_path):
    """
    Ejecuta el detector Nuestro.py y devuelve DataFrame con columnas similares.
    """
    import plagiarism_clusters  # asumiendo nuestro.py modularizado

    files = plagiarism_clusters.collect_py_files(dataset_path)
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
    return df


def evaluate(y_true, y_pred):
    labels = [0, 1, 2, 3]
    print("Matriz de Confusión:")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(f"F1 weighted: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Reporte completo:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


def main():
    # Ruta dataset y CSV con etiquetas verdaderas
    dataset_path = "dataset_test"  # ajusta a tu ruta
    csv_etiquetado = "plagiarism_pairs.csv"

    # Carga etiquetas verdaderas
    df_true = pd.read_csv(csv_etiquetado)

    # Ejecutar Deckard
    print("Ejecutando Deckard...")
    deckard_df, deckard_time, deckard_mem = run_with_profiler(
        run_deckard_detector, dataset_path
    )
    print(f"Deckard tiempo: {deckard_time:.2f} s, pico memoria: {deckard_mem:.2f} MB")

    # Ejecutar Nuestro
    print("Ejecutando Nuestro...")
    nuestro_df, nuestro_time, nuestro_mem = run_with_profiler(
        run_nuestro_detector, dataset_path
    )
    print(f"Nuestro tiempo: {nuestro_time:.2f} s, pico memoria: {nuestro_mem:.2f} MB")

    # Para simplificar, haremos matching simple con merge de pares etiquetados vs detectados
    # Asegúrate que columnas y formatos coinciden

    def match_and_evaluate(df_detected, name):
        print(f"\nEvaluando resultados de {name}...")
        merge_cols = [
            "file1",
            "file2",
        ]

        # Para pares detectados, etiqueta predicha=1, no detectados=0
        # Primero combinamos con pares etiquetados para evaluar
        df_merge = pd.merge(
            df_true,
            df_detected[merge_cols + ["predicted_label"]],
            on=merge_cols,
            how="left",
        )

        df_merge["predicted_label"] = df_merge["predicted_label"].fillna(0).astype(int)

        y_true = df_merge["true_label"].values
        y_pred = df_merge["predicted_label"].values

        evaluate(y_true, y_pred)

    print("Resultados de Deckard:")
    print(deckard_df.head())
    print("Resultados de Nuestro:")
    print(nuestro_df.head())

    match_and_evaluate(deckard_df, "Deckard")
    match_and_evaluate(nuestro_df, "Nuestro")


if __name__ == "__main__":
    main()
