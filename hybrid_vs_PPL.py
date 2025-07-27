# ============================================
# Hybrid vs Pruned Landmark Labeling (PLL)
# ============================================

import networkx as nx
import torch
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
import random
import numpy as np
import time
import pandas as pd
import community.community_louvain as community_louvain
from collections import Counter

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_dataset(name):
    dataset = Planetoid(root=f'/tmp/{name}', name=name)
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        G.add_edge(int(u), int(v), weight=1)
    labels = data.y.numpy()
    return G, labels

# -------------------------------
# Optimized Pruned PLL
# -------------------------------
class PrunedPLLIndex:
    def __init__(self, G):
        self.G = G
        self.labels = {}

    def build(self):
        nodes = list(self.G.nodes())
        self.labels = {v: {} for v in nodes}
        for v in tqdm(nodes, desc="Building Pruned PLL"):
            dist = nx.single_source_dijkstra_path_length(self.G, v)
            for u, d in dist.items():
                if any(self.labels[u].get(k, float("inf")) + self.labels[v].get(k, float("inf")) <= d
                       for k in self.labels[u].keys() & self.labels[v].keys()):
                    continue
                self.labels[u][v] = d
                self.labels[v][u] = d

    def query(self, u, v):
        return min([self.labels[u][k] + self.labels[v][k]
                    for k in self.labels[u].keys() & self.labels[v].keys()],
                   default=float('inf'))

# -------------------------------
# Hybrid with Louvain Clustering
# -------------------------------
def build_hybrid_index(G, clusters):
    cluster_ids = set(clusters.values())
    cluster_graphs = {cid: G.subgraph([n for n in G if clusters[n] == cid]) for cid in cluster_ids}
    cluster_pll = {}
    centroids = {}
    for cid, subg in cluster_graphs.items():
        pll = PrunedPLLIndex(subg)
        pll.build()
        cluster_pll[cid] = pll
        ecc = nx.eccentricity(subg)
        centroids[cid] = min(ecc, key=ecc.get)

    centroid_graph = nx.Graph()
    for cid in cluster_ids:
        centroid_graph.add_node(cid)
    for cid1 in cluster_ids:
        for cid2 in cluster_ids:
            if cid1 < cid2:
                try:
                    d = nx.shortest_path_length(G, source=centroids[cid1], target=centroids[cid2])
                    centroid_graph.add_edge(cid1, cid2, weight=d)
                except:
                    continue
    centroid_dist = dict(nx.all_pairs_dijkstra_path_length(centroid_graph))
    return cluster_pll, centroids, centroid_dist

def query_hybrid(u, v, clusters, cluster_pll, centroids, centroid_dist):
    cu, cv = clusters[u], clusters[v]
    if cu == cv:
        return cluster_pll[cu].query(u, v)
    try:
        d1 = cluster_pll[cu].query(u, centroids[cu])
        d2 = centroid_dist[cu][cv]
        d3 = cluster_pll[cv].query(centroids[cv], v)
        return d1 + d2 + d3
    except:
        return float('inf')

# -------------------------------
# Evaluation
# -------------------------------
def evaluate(G, labels, name):
    clusters = community_louvain.best_partition(G)

    pll = PrunedPLLIndex(G)
    start = time.time()
    pll.build()
    pll_time = time.time() - start

    start = time.time()
    cluster_pll, centroids, centroid_dist = build_hybrid_index(G, clusters)
    hybrid_time = time.time() - start

    # クラスタ数の表示
    cluster_ids = set(clusters.values())
    print(f"クラスタ数: {len(cluster_ids)}")

    # オプション：クラスタごとのサイズ分布を表示
    cluster_sizes = Counter(clusters.values())
    print(f"クラスタサイズ分布: {sorted(cluster_sizes.values(), reverse=True)}")

    nodes = list(G.nodes)
    pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(1000)]

    result = []
    for method_name, method in [
        ('PLL', lambda u, v: pll.query(u, v)),
        ('Hybrid', lambda u, v: query_hybrid(u, v, clusters, cluster_pll, centroids, centroid_dist))
    ]:
        correct = 0
        total = 0
        mae = 0.0
        start = time.time()
        for u, v in pairs:
            try:
                gt = nx.shortest_path_length(G, u, v)
                est = method(u, v)
                if est == gt:
                    correct += 1
                if est != float("inf"):
                    mae += abs(est - gt)
                    total += 1
            except:
                continue
        end = time.time()
        result.append({
            "method": method_name,
            "query_time_sec": end - start,
            "mae": mae / total if total > 0 else float('inf'),
            "samples": total,
            "exact_matches": correct,
            "build_time_sec": pll_time if method_name == "PLL" else hybrid_time
        })
    return pd.DataFrame(result)

# -------------------------------
# Execute
# -------------------------------
G, labels = load_dataset("Cora")
df = evaluate(G, labels, "Cora")
print(df)

