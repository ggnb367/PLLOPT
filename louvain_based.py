import time
import random
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
import community.community_louvain as community_louvain
from tqdm import tqdm

# -------------------------------
# データセットの読み込み
# -------------------------------
def load_dataset(name):
    if name == "Cora":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
        edge_index = data.edge_index.numpy()
        G = nx.Graph()
        for u, v in zip(edge_index[0], edge_index[1]):
            G.add_edge(int(u), int(v), weight=1)
        return G, data.y.numpy()
    else:
        raise NotImplementedError

# -------------------------------
# Pruned Landmark Labeling
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
                        if k in label:
                            if label[k] + self.labels[u][k] <= dist[u]:
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
# クラスタごとの重心決定 (eccentricity最小)
# -------------------------------
def compute_centroid(G, nodes):
    ecc = {}
    for node in nodes:
        try:
            length = nx.single_source_dijkstra_path_length(G, node)
            ecc[node] = max([length[n] for n in nodes if n in length])
        except:
            continue
    return min(ecc.items(), key=lambda x: x[1])[0] if ecc else random.choice(nodes)

# -------------------------------
# ハイブリッドインデックス構築
# -------------------------------
def build_hybrid_index(G, initial_clusters, size_threshold=25):
    # ステップ1: クラスタをサイズで分離
    cluster_sizes = Counter(initial_clusters.values())
    large_clusters = {k for k, v in cluster_sizes.items() if v >= size_threshold}
    node_to_cluster = {}
    for node, cid in initial_clusters.items():
        if cid in large_clusters:
            node_to_cluster[node] = cid

    # ステップ2: 重心ノードを大クラスタについて決定
    cluster_to_nodes = defaultdict(list)
    for node, cid in node_to_cluster.items():
        cluster_to_nodes[cid].append(node)
    centroids = {cid: compute_centroid(G, nodes) for cid, nodes in cluster_to_nodes.items()}

    # ステップ3: 小クラスタのノードを最近傍重心に割り当てる
    for node in G.nodes():
        if node not in node_to_cluster:
            min_dist = float('inf')
            best_cid = None
            for cid, centroid in centroids.items():
                try:
                    d = nx.shortest_path_length(G, node, centroid)
                    if d < min_dist:
                        min_dist = d
                        best_cid = cid
                except:
                    continue
            if best_cid is not None:
                node_to_cluster[node] = best_cid

    # ステップ4: クラスタごとにPLL構築
    cluster_pll = {}
    for cid, nodes in cluster_to_nodes.items():
        subgraph = G.subgraph([n for n in node_to_cluster if node_to_cluster[n] == cid])
        pll = PrunedPLLIndex(subgraph)
        pll.build()
        cluster_pll[cid] = pll

    # ステップ5: 重心間の距離事前計算
    centroid_dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    return cluster_pll, centroids, centroid_dist

# -------------------------------
# ハイブリッドクエリ
# -------------------------------
def query_hybrid(u, v, clusters, cluster_pll, centroids, centroid_dist):
    cid_u = clusters[u]
    cid_v = clusters[v]
    if cid_u == cid_v:
        return cluster_pll[cid_u].query(u, v)
    else:
        cu = centroids[cid_u]
        cv = centroids[cid_v]
        d1 = cluster_pll[cid_u].query(u, cu)
        d2 = cluster_pll[cid_v].query(v, cv)
        d3 = centroid_dist.get(cu, {}).get(cv, float('inf'))
        return d1 + d2 + d3

# -------------------------------
# 評価
# -------------------------------
def evaluate(G, labels, name):
    initial_clusters = community_louvain.best_partition(G)

    pll = PrunedPLLIndex(G)
    start = time.time()
    pll.build()
    pll_time = time.time() - start

    start = time.time()
    cluster_pll, centroids, centroid_dist = build_hybrid_index(G, initial_clusters)
    hybrid_time = time.time() - start

    # クラスタ数の表示
    print(f"クラスタ数: {len(set(cluster_pll.keys()))}")

    # クラスタサイズ分布を表示
    sizes = Counter(initial_clusters.values())
    print(f"クラスタサイズ分布: {sorted(sizes.values(), reverse=True)}")

    # サンプル生成
    nodes = list(G.nodes)
    pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(1000)]

    result = []
    for method_name, method in [
        ('PLL', lambda u, v: pll.query(u, v)),
        ('Hybrid', lambda u, v: query_hybrid(u, v, initial_clusters, cluster_pll, centroids, centroid_dist))
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
# 実行
# -------------------------------
G, labels = load_dataset("Cora")
df = evaluate(G, labels, "Cora")
print(df)
