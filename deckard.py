import ast
import os
import numpy as np
import logging
from collections import defaultdict, Counter
import itertools

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ASTVectorizer(ast.NodeVisitor):
    def __init__(self, vocab):
        self.vocab = vocab
        self.counter = Counter()
        self.vectors = []
        self.locations = []

    def visit(self, node):
        self.counter[type(node).__name__] += 1
        super().visit(node)

    def extract_vectors(self, tree, min_size=5):
        self.vectors.clear()
        self.locations.clear()
        for node in ast.walk(tree):
            self.counter.clear()
            self.visit(node)
            size = sum(self.counter.values())
            if size >= min_size:
                vec = np.array([self.counter[t] for t in self.vocab], dtype=float)
                self.vectors.append(vec)
                self.locations.append(getattr(node, 'lineno', None))
        return self.vectors, self.locations


class LSHIndex:
    def __init__(self, dim, k=5, L=10, w=4.0):
        self.k, self.L, self.w, self.dim = k, L, w, dim
        self.tables = [defaultdict(list) for _ in range(L)]
        self.projections = [np.random.randn(k, dim) for _ in range(L)]
        self.offsets = [np.random.rand(k) * w for _ in range(L)]

    def _hash(self, v):
        hashes = []
        for proj, offset in zip(self.projections, self.offsets):
            bins = np.floor((proj.dot(v) + offset) / self.w).astype(int)
            hashes.append(tuple(bins))
        return hashes

    def insert(self, vec, loc):
        for table, h in zip(self.tables, self._hash(vec)):
            table[h].append((vec, loc))

    def query(self, vec):
        cands = set()
        for table, h in zip(self.tables, self._hash(vec)):
            for v, loc in table.get(h, []):
                cands.add((tuple(v), loc))
        return [(np.array(v), loc) for v, loc in cands]


def sliding_window_merge(vectors, locations, window_size=3):
    merged, merged_locs = [], []
    for i in range(len(vectors) - window_size + 1):
        window_sum = np.sum(vectors[i:i+window_size], axis=0)
        merged.append(window_sum)
        merged_locs.append(locations[i])
    return merged, merged_locs


def detect_clones_in_file(path, lsh_index, vocab, min_size=5, window_size=3):
    logging.info("Parsing and vectorizing %s", path)
    with open(path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=path)
    vecs, locs = ASTVectorizer(vocab).extract_vectors(tree, min_size=min_size)
    logging.info("  → %d subárboles ≥ %d nodos", len(vecs), min_size)
    merged_vecs, merged_locs = sliding_window_merge(vecs, locs, window_size=window_size)
    logging.info("  → %d vectores tras ventana deslizante de tamaño %d", len(merged_vecs), window_size)
    for vec, loc in zip(merged_vecs, merged_locs):
        lsh_index.insert(vec, (path, loc))


def find_clones(lsh_index, min_dist=5.0):
    clones = []
    for idx, table in enumerate(lsh_index.tables):
        logging.info("Revisando tabla LSH %d/%d", idx+1, lsh_index.L)
        for bucket in table.values():
            for (v1, loc1), (v2, loc2) in itertools.combinations(bucket, 2):
                dist = np.linalg.norm(v1 - v2)
                if dist < min_dist:
                    clones.append((loc1, loc2, dist))
    return clones


def run_deckard_on_directory(directory,
                             min_size=5,
                             window_size=3,
                             min_dist=5.0,
                             k=5, L=10, w=4.0):
    logging.info("Iniciando análisis en directorio: %s", directory)

    # 1) Construir vocabulario global de tipos de nodo
    node_types = set()
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.py'):
                full = os.path.join(root, f)
                logging.debug("  → escaneando nodos de %s", full)
                tree = ast.parse(open(full, 'r', encoding='utf-8').read())
                for node in ast.walk(tree):
                    node_types.add(type(node).__name__)
    if not node_types:
        raise FileNotFoundError(f"No se encontró ningún .py en {directory}")
    vocab = sorted(node_types)
    logging.info("Vocabulario construido con %d tipos de nodo", len(vocab))

    # 2) Inicializar LSH
    dim = len(vocab)
    lsh = LSHIndex(dim=dim, k=k, L=L, w=w)
    logging.info("Índice LSH inicializado (dim=%d, k=%d, L=%d, w=%.1f)", dim, k, L, w)

    # 3) Procesar archivos
    total_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                total_files += 1
                path = os.path.join(root, file)
                detect_clones_in_file(path, lsh, vocab,
                                      min_size=min_size,
                                      window_size=window_size)
    logging.info("Procesados %d archivos .py", total_files)

    # 4) Detectar clones
    clones = find_clones(lsh, min_dist=min_dist)
    logging.info("Detección completa: se encontraron %d pares de clones (dist < %.1f)", len(clones), min_dist)
    return clones


if __name__ == "__main__":
    base_dir = "dataset_4"  # ajusta a tu ruta
    clones = run_deckard_on_directory(
        base_dir,
        min_size=5,
        window_size=3,
        min_dist=5.0,
        k=5,
        L=10,
        w=4.0
    )
    for loc1, loc2, dist in clones:
        print(f"Clone between {loc1} and {loc2} (distance: {dist:.2f})")
