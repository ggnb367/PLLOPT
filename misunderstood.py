"""
Hybrid PLL approach with Louvain communities. Large communities are assessed
for scale-free-ness using power-law fitting. Communities whose fit score is
below a threshold choose their centroid by betweenness centrality; others use
eccentricity. Small communities always select the highest betweenness node in a
local neighbourhood. The rest follows the logic of `hybrid_betweenness.py`.
The evaluation reports build times, query accuracy, and the number of labels
stored in each PLL index.
"""

import time
import random
from collections import defaultdict, Counter
import warnings
import contextlib
import io

import networkx as nx
import pandas as pd
from torch_geometric.datasets import Planetoid
import community.community_louvain as community_louvain
from tqdm import tqdm
import numpy as np
import powerlaw

# -------------------------------
# Load Cora dataset
# -------------------------------

def load_cora():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    for u, v in zip(edge_index[0], edge_index[1]):
        G.add_edge(int(u), int(v), weight=1)
    return G

# -------------------------------
# Pruned Landmark Labeling (basic)
# -------------------------------
class PrunedPLLIndex:
    def __init__(self, G):
        self.G = G
        self.labels = {}

    def build(self):
        for v in tqdm(self.G.nodes(), desc="Building PLL index"):
            dist = nx.single_source_dijkstra_path_length(self.G, v)
            label = {}
            for u in dist:
                if u in self.labels:
                    prune = False
                    for k in self.labels[u]:
                        if k in label and label[k] + self.labels[u][k] <= dist[u]:
                            prune = True
                            break
                    if prune:
                        continue
                label[u] = dist[u]
            self.labels[v] = label

    def query(self, u, v):
        dist = float('inf')
        if u not in self.labels or v not in self.labels:
            return dist
        for k in self.labels[u]:
            if k in self.labels[v]:
                d = self.labels[u][k] + self.labels[v][k]
                if d < dist:
                    dist = d
        return dist

# -------------------------------
# Utility functions
# -------------------------------

def compute_centroid_ecc(G, nodes):
    subg = G.subgraph(nodes)
    ecc = nx.eccentricity(subg)
    return min(ecc, key=ecc.get)


def compute_centroid_bc(subg, nodes):
    bc = nx.betweenness_centrality(subg)
    return max(nodes, key=lambda n: bc.get(n, 0.0))


def internal_degrees(G, nodes):
    subg = G.subgraph(nodes)
    return [subg.degree(n) for n in nodes]


def powerlaw_score(degs):
    if len(degs) < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            fit = powerlaw.Fit(degs, verbose=False)
            _, p = fit.distribution_compare("power_law", "lognormal")
    return p

# -------------------------------
# Hybrid index
# -------------------------------

def build_hybrid_index(G, clusters, size_threshold=25, sf_threshold=0.1):
    cluster_nodes = defaultdict(list)
    for node, cid in clusters.items():
        cluster_nodes[cid].append(node)

    cluster_sizes = {cid: len(nodes) for cid, nodes in cluster_nodes.items()}
    large_clusters = {cid for cid, sz in cluster_sizes.items() if sz >= size_threshold}
    small_clusters = set(cluster_nodes) - large_clusters

    centroids = {}
    cluster_pll = {}

    # large clusters
    for cid in large_clusters:
        nodes = cluster_nodes[cid]
        degs = internal_degrees(G, nodes)
        score = powerlaw_score(degs)
        subg = G.subgraph(nodes)
        if score < sf_threshold:
            centroid = compute_centroid_bc(subg, nodes)
        else:
            centroid = compute_centroid_ecc(G, nodes)
        centroids[cid] = centroid
        pll = PrunedPLLIndex(subg)
        pll.build()
        cluster_pll[cid] = pll

    # small clusters: betweenness in local neighbourhood
    for cid in small_clusters:
        nodes = cluster_nodes[cid]
        local_nodes = set(nodes)
        for n in nodes:
            for nb in G.neighbors(n):
                if clusters[nb] != cid:
                    local_nodes.add(nb)
        local_subg = G.subgraph(local_nodes)
        centroid = compute_centroid_bc(local_subg, nodes)
        centroids[cid] = centroid
        subg = G.subgraph(nodes)
        pll = PrunedPLLIndex(subg)
        pll.build()
        cluster_pll[cid] = pll

    centroid_nodes = list(centroids.values())
    centroid_dist = {}
    for c in centroid_nodes:
        lengths = nx.single_source_dijkstra_path_length(G, c)
        centroid_dist[c] = {d: l for d, l in lengths.items() if d in centroid_nodes}

    return {
        'large_clusters': large_clusters,
        'cluster_nodes': cluster_nodes,
        'cluster_pll': cluster_pll,
        'centroids': centroids,
        'centroid_dist': centroid_dist,
        'clusters': clusters,
        'centroid_nodes': centroid_nodes,
    }

# -------------------------------
# Hybrid query
# -------------------------------

def query_hybrid(G, u, v, index):
    clusters = index['clusters']
    centroids = index['centroids']
    cluster_pll = index['cluster_pll']
    centroid_dist = index['centroid_dist']

    cid_u = clusters[u]
    cid_v = clusters[v]

    if cid_u == cid_v:
        return cluster_pll[cid_u].query(u, v)

    cu = centroids[cid_u]
    cv = centroids[cid_v]
    du = cluster_pll[cid_u].query(u, cu)
    dv = cluster_pll[cid_v].query(v, cv)
    dcv = centroid_dist.get(cu, {}).get(cv, float('inf'))
    return du + dcv + dv

# -------------------------------
# Evaluation
# -------------------------------

def evaluate(G, size_threshold=25, sf_threshold=0.1, samples=1000):
    clusters = community_louvain.best_partition(G)

    pll = PrunedPLLIndex(G)
    start = time.time()
    pll.build()
    pll_time = time.time() - start
    pll_size = sum(len(lbls) for lbls in pll.labels.values())

    start = time.time()
    index = build_hybrid_index(G, clusters, size_threshold, sf_threshold)
    hybrid_time = time.time() - start
    hybrid_size = sum(
        sum(len(lbls) for lbls in pll_idx.labels.values())
        for pll_idx in index['cluster_pll'].values()
    )

    cluster_sizes = Counter(clusters.values())
    print(f"クラスタ数: {len(cluster_sizes)}")
    print(f"クラスタサイズ分布: {sorted(cluster_sizes.values(), reverse=True)}")

    nodes = list(G.nodes)
    query_pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(samples)]

    result = []
    for method_name, method in [
        ('PLL', lambda x, y: pll.query(x, y)),
        ('Hybrid', lambda x, y: query_hybrid(G, x, y, index)),
    ]:
        correct = 0
        total = 0
        mae = 0.0
        start = time.time()
        for u, v in query_pairs:
            try:
                gt = nx.shortest_path_length(G, u, v)
                est = method(u, v)
                if est == gt:
                    correct += 1
                if est != float('inf'):
                    mae += abs(est - gt)
                    total += 1
            except Exception:
                continue
        end = time.time()
        result.append({
            'method': method_name,
            'query_time_sec': end - start,
            'mae': mae / total if total > 0 else float('inf'),
            'samples': total,
            'exact_matches': correct,
            'build_time_sec': pll_time if method_name == 'PLL' else hybrid_time,
            'index_size': pll_size if method_name == 'PLL' else hybrid_size,
        })
    return pd.DataFrame(result)

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    G = load_cora()
    df = evaluate(G)
    print(df)
