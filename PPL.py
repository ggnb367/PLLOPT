import networkx as nx
import heapq
import random
import matplotlib.pyplot as plt
from collections import Counter

class PrunedLandmarkLabeling:
    def __init__(self, G, min_landmarks=20, sample_pairs_num=100, zero_error_threshold=2, error_epsilon=1e-8):
        self.G = G
        self.labels = {v: [] for v in G.nodes}
        self.node_degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
        self.min_landmarks = min_landmarks
        self.sample_pairs = self._generate_sample_pairs(sample_pairs_num)
        self.zero_error_threshold = zero_error_threshold
        self.error_epsilon = error_epsilon
        self.order = []
        self.build_index()

    def _generate_sample_pairs(self, num):
        nodes = list(self.G.nodes)
        pairs = []
        for _ in range(num):
            u, v = random.sample(nodes, 2)
            pairs.append((u,v))
        return pairs

    def _estimate_error(self):
        error_sum = 0
        for u,v in self.sample_pairs:
            est = self.query_distance(u,v)
            try:
                true_dist = nx.shortest_path_length(self.G, u, v)
            except nx.NetworkXNoPath:
                true_dist = float('inf')
            if est == float('inf') and true_dist == float('inf'):
                dist_err = 0
            elif est == float('inf') or true_dist == float('inf'):
                dist_err = float('inf')
            else:
                dist_err = abs(est - true_dist)
            error_sum += dist_err
        return error_sum / len(self.sample_pairs)

    def build_index(self):
        total_nodes = len(self.G)
        print(f"Start landmark selection, total nodes: {total_nodes}")

        consecutive_zero_error = 0
        i = 0
        while i < total_nodes:
            landmark, deg = self.node_degrees[i]
            self.order.append(landmark)
            self._bfs_label(landmark)
            i += 1

            error = self._estimate_error()
            #print(f"Landmarks used: {i}, average distance estimation error: {error:.6f}")

            if error < self.error_epsilon:
                consecutive_zero_error += 1
            else:
                consecutive_zero_error = 0

            if i >= self.min_landmarks and consecutive_zero_error >= self.zero_error_threshold:
                #print(f"Average distance estimation error below {self.error_epsilon} for {self.zero_error_threshold} consecutive times; stopping.")
                break

    def _bfs_label(self, landmark):
        dist = {landmark: 0}
        visited = set([landmark])
        queue = [(0, landmark)]

        while queue:
            cur_dist, u = heapq.heappop(queue)

            if self.query_distance(u, landmark) <= cur_dist:
                continue

            self.labels[u].append((landmark, cur_dist))

            for v in self.G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    heapq.heappush(queue, (cur_dist + 1, v))
                    dist[v] = cur_dist + 1

    def query_distance(self, u, v):
        labels_u = dict(self.labels[u])
        labels_v = dict(self.labels[v])
        common = set(labels_u.keys()) & set(labels_v.keys())
        if not common:
            return float('inf')
        return min(labels_u[l] + labels_v[l] for l in common)

    def get_landmark_count(self):
        return len(self.order)

def generate_random_graph(n_nodes=1000, n_edges=8000):
    p = n_edges / (n_nodes*(n_nodes-1)/2)
    print(f"Generating Erdős-Rényi random graph with p={p:.6f}")
    G = nx.erdos_renyi_graph(n_nodes, p)
    return G

def generate_scale_free_graph(n_nodes=1000, n_edges=8000):
    m = max(1, n_edges // n_nodes)
    print(f"Generating Barabási-Albert scale-free graph with m={m}")
    G = nx.barabasi_albert_graph(n_nodes, m)
    return G

def plot_degree_distribution(G, title):
    degrees = [deg for _, deg in G.degree()]
    degree_count = Counter(degrees)
    x = list(degree_count.keys())
    y = list(degree_count.values())

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title(title)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()

def main():
    print("=== Random Graph ===")
    n_node = 5000
    n_edge = n_node * 5
    G_random = generate_random_graph(n_node, n_edge)
    print(f"Random graph: Nodes={G_random.number_of_nodes()}, Edges={G_random.number_of_edges()}")
    plot_degree_distribution(G_random, "Random Graph Degree Distribution")

    print("Building PLL index for random graph...")
    start = time.time()
    pll_random = PrunedLandmarkLabeling(G_random)
    end = time.time()
    print(f"Random graph landmark count: {pll_random.get_landmark_count()}")
    print(f"PLL construction time: {end - start:.2f} seconds")

    print("\n=== Scale-Free Graph ===")
    G_sf = generate_scale_free_graph(n_node, n_edge)
    print(f"Scale-free graph: Nodes={G_sf.number_of_nodes()}, Edges={G_sf.number_of_edges()}")
    plot_degree_distribution(G_sf, "Scale-Free Graph Degree Distribution")

    print("Building PLL index for scale-free graph...")
    start = time.time()
    pll_sf = PrunedLandmarkLabeling(G_sf)
    end = time.time()
    print(f"Scale-free graph landmark count: {pll_sf.get_landmark_count()}")
    print(f"PLL construction time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
