#!/usr/bin/env python3
"""
Detector de plagio basado en AST + LSH con normalización, métrica coseno,
filtro de longitud de fragmento, muestra de snippets y clustering via LSH
- QVG: characteristic vectors por sub-árbol
- WVG: fenestrado de vectores para clones parciales (opcional)
- LSH: detección aproximada usando distancia coseno
- Clustering: agrupamiento en componentes conexas del grafo LSH
"""

import ast
import os
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
import json


def collect_py_files(root_dir):
    py_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".py"):
                py_files.append(os.path.join(dirpath, fn))
    return py_files


def index_node_types():
    types = {
        cls: i
        for i, cls in enumerate(
            [
                getattr(ast, name)
                for name in dir(ast)
                if isinstance(getattr(ast, name), type)
                and issubclass(getattr(ast, name), ast.AST)
            ]
        )
    }
    return types


NODE2IDX = index_node_types()
DIM = len(NODE2IDX)


def qvg(node):
    vec = np.zeros(DIM, dtype=int)
    count = 1
    for child in ast.iter_child_nodes(node):
        v_c, c_c = qvg(child)
        vec += v_c
        count += c_c
    vec[NODE2IDX[type(node)]] += 1
    return vec, count


def extract_qvg_vectors(tree, min_nodes=30):
    vectors = []
    for node in ast.walk(tree):
        v, count = qvg(node)
        if count >= min_nodes:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is not None:
                vectors.append((v.astype(float), (start, end)))
    if not vectors:
        start = 1
        end = (tree.body and tree.body[-1].lineno) or 1
        vec, _ = qvg(tree)
        vectors.append((vec.astype(float), (start, end)))
    return vectors


def wvg(vectors, window=5, stride=1):
    pooled = []
    vecs = [v for v, _ in vectors]
    lines = [rng for _, rng in vectors]
    for i in range(0, len(vecs) - window + 1, stride):
        win_vec = np.sum(vecs[i : i + window], axis=0)
        win_start = min(r[0] for r in lines[i : i + window])
        win_end = max(r[1] for r in lines[i : i + window])
        pooled.append((win_vec, (win_start, win_end)))
    return pooled


def detect_clones(file_paths, min_nodes, window, stride, radius, length_tol):
    all_vecs, line_ranges, file_map = [], [], []

    for fpath in file_paths:
        print("FILE", fpath)
        src = open(fpath, "r", encoding="utf-8").read()
        raw = open(fpath, "r", encoding="utf-8").read()
        src = raw.expandtabs(4)
        tree = ast.parse(src)
        qv = extract_qvg_vectors(tree, min_nodes)
        qv_sorted = sorted(qv, key=lambda x: x[1][0])
        wv = wvg(qv_sorted, window, stride) if window > 1 else qv_sorted
        for vec, (s, e) in wv:
            all_vecs.append(vec)
            line_ranges.append((s, e))
            file_map.append(fpath)

    print(f"Total vectores extraídos: {len(all_vecs)}")
    if not all_vecs:
        print("⚠️  No se extrajeron vectores.")
        return file_map, line_ranges, [], None

    X = np.stack(all_vecs)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms != 0, norms, 1)

    # construir grafo LSH y extraer pares aproximados
    nbrs = NearestNeighbors(radius=radius, metric="cosine").fit(X_norm)
    adj = nbrs.radius_neighbors_graph(X_norm, mode="connectivity").tocoo()
    raw_pairs = {(i, j) for i, j in zip(adj.row, adj.col) if i < j}
    print(f"Pares crudos encontrados: {len(raw_pairs)}")

    # filtrar pares entre ficheros distintos
    cross = [(i, j) for i, j in raw_pairs if file_map[i] != file_map[j]]
    print(f"Pares entre ficheros distintos: {len(cross)}")

    # filtro de longitud similar
    final = []
    for i, j in cross:
        si, ei = line_ranges[i]
        sj, ej = line_ranges[j]
        l1, l2 = ei - si, ej - sj
        if max(l1, l2) == 0:
            continue
        if abs(l1 - l2) / max(l1, l2) <= length_tol:
            final.append((i, j))
    print(f"Pares tras filtro de longitud similar: {len(final)}")

    # clustering sobre pares finales (componentes conexas)
    if final:
        N = len(file_map)
        rows = [i for i, j in final] + [j for i, j in final]
        cols = [j for i, j in final] + [i for i, j in final]
        data = [1] * len(rows)
        adj_final = coo_matrix((data, (rows, cols)), shape=(N, N))
        n_comp, labels = connected_components(adj_final, directed=False)
        print(f"Componentes conexas finales: {n_comp}")
    else:
        labels = np.arange(len(file_map))

    return file_map, line_ranges, final, labels


def filter_overlapping_segments(file_map, line_ranges, members):
    filtered, by_file = [], {}
    for idx in members:
        by_file.setdefault(file_map[idx], []).append(idx)
    for f, idxs in by_file.items():
        idxs_sorted = sorted(
            idxs, key=lambda i: -(line_ranges[i][1] - line_ranges[i][0])
        )
        sel = []
        for i in idxs_sorted:
            s, e = line_ranges[i]
            if all(e <= line_ranges[j][0] or s >= line_ranges[j][1] for j in sel):
                sel.append(i)
        filtered.extend(sel)
    return filtered


def merge_intervals(intervals):
    """
    Merge overlapping intervals in a list of (start, end) tuples.
    """
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged = [intervals_sorted[0]]
    for start, end in intervals_sorted[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:  # overlap
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def write_merge_clusters(clusters):
    """
    Merge clusters with overlapping intervals in the same file and save the results to a JSON file.
    """
    merged_clusters = {}
    merged_cluster_index = 0

    processed_clusters = set()
    for lab, members in clusters.items():
        if lab in processed_clusters or len(members) == 1:
            continue

        merged = False
        files = members.keys()
        for other_lab, other_members in clusters.items():
            if other_lab == lab or other_lab in processed_clusters:
                continue
            if set(other_members.keys()) == set(files):
                for f in files:
                    # Combine intervals from both clusters
                    combined_intervals = merge_intervals(members[f] + other_members[f])

                    # Filter existing intervals in merged_clusters
                    existing_intervals = merged_clusters.get(
                        str(merged_cluster_index), {}
                    ).get(f, [])

                    filtered_intervals = set(
                        merge_intervals(existing_intervals + combined_intervals)
                    )

                    merged_clusters.setdefault(str(merged_cluster_index), {})[f] = list(
                        filtered_intervals
                    )
                processed_clusters.add(lab)
                processed_clusters.add(other_lab)
                merged = True

        if not merged:
            merged_clusters[str(merged_cluster_index)] = members

        processed_clusters.add(lab)
        merged_cluster_index += 1

    # Save the merged clusters to a JSON file
    with open("merged_clusters.json", "w", encoding="utf-8") as json_file:
        json.dump(merged_clusters, json_file, indent=4, ensure_ascii=False)

    print("Merged clusters saved to merged_clusters.json")
    return merged_clusters


def write_clustering_report(clusters, labels, filename="output.txt"):

    with open(filename, "w", encoding="utf-8") as out:
        out.write(f"Clustering de {len(labels)} fragmentos (pares exactos)\n\n")
        for lab, members in sorted(clusters.items()):
            if (
                not members
                or sum(len(intervals) for intervals in members.values()) <= 1
            ):
                continue
            out.write(f"Cluster {lab}:\n")
            total_members = 0
            for f, intervals in members.items():
                for s, e in intervals:
                    out.write(f"  - {f} líneas {s}-{e}\n")
                total_members += len(intervals)
            out.write(f"  Total miembros tras merge: {total_members}\n\n")
    print(f"Análisis de clustering escrito en {filename}")


def group_clusters(file_map, line_ranges, labels):
    clusters = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, {}).setdefault(file_map[idx], []).append(
            line_ranges[idx]
        )

    for lab, members in clusters.items():
        for f, intervals in members.items():
            clusters[lab][f] = merge_intervals(intervals)

    return clusters


def main():
    parser = argparse.ArgumentParser(
        description="Detector de clones AST+LSH con clustering de pares exactos"
    )
    parser.add_argument("--root", type=str, help="directorio raíz con .py")
    parser.add_argument("paths", nargs="*", help="archivos o directorios .py")
    parser.add_argument("--min-nodes", type=int, default=30)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--length-tol", type=float, default=0.2)
    args = parser.parse_args()

    files = collect_py_files(args.root) if args.root else []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(collect_py_files(p))
        elif p.endswith(".py"):
            files.append(p)

    file_map, line_ranges, clones, labels = detect_clones(
        files, args.min_nodes, args.window, args.stride, args.radius, args.length_tol
    )

    clusters = group_clusters(file_map, line_ranges, labels)
    write_clustering_report(clusters, labels)
    write_merge_clusters(clusters)

    if not clones:
        print("No se encontraron fragmentos similares.")
        return

    print("\nFragmentos similares detectados:\n")
    for i, j in clones:
        fi, fj = file_map[i], file_map[j]
        si, ei = line_ranges[i]
        sj, ej = line_ranges[j]
        print(f"{fi} líneas {si}–{ei} ≈ {fj} líneas {sj}–{ej}\n")


if __name__ == "__main__":
    main()
