# PLLOPT

Graph Distance Query: Hybrid vs. Pruned Landmark Labeling (PLL)
项目概述
本项目旨在探索和比较在大型图上高效计算任意两点间最短路径的两种主要策略：Pruned Landmark Labeling (PLL) 和一种基于社区划分的混合（Hybrid）方法。我们通过在真实世界数据集（如Cora）上进行实验，分析它们在索引构建时间、查询速度以及查询准确性方面的性能权衡。

在图论和许多实际应用（如社交网络分析、路由、推荐系统等）中，高效地查找节点之间的最短路径距离至关重要。传统的图遍历算法（如BFS或Dijkstra）虽然精确，但对于每次查询都需要重新计算，在大型图上效率低下。PLL通过预计算和存储部分距离信息来加速查询，而混合方法则进一步利用图的社区结构来优化这一过程。

解决的问题
大规模图上的最短路径查询效率：如何在大规模图中快速响应任意两点间的最短路径查询。
PLL的构建和存储成本：纯PLL在某些图结构（如随机图）上可能面临高昂的构建时间和大量的标签存储。
兼顾速度与精度：如何在追求查询速度的同时，保持合理的距离估计精度。
方法论
本项目实现了以下两种核心方法：
Pruned Landmark Labeling (PLL):
原理: PLL是一种预计算索引方法。它为图中的每个节点存储一组“地标”及其到这些地标的最短距离。在构建过程中，利用剪枝技术避免存储冗余信息。
查询: 通过查找两个查询节点共享的地标，并结合地标到各自节点的距离来计算最短路径。
Hybrid Approach (基于 Louvain 社区发现):
原理: 该方法旨在结合PLL的优点并解决其在大图上的扩展性问题。
核心步骤:
社区发现: 利用 Louvain 社区发现算法将图划分为多个紧密连接的社区（聚类）。
局部 PLL 构建: 对每个独立的社区子图，分别构建一个 Pruned Landmark Labeling 索引。这使得社区内部的查询非常高效。
重心选择: 对于每个较大的社区，识别一个“重心”节点（通常选择离心率最小的节点）。
重心间距离预计算: 计算并存储所有重心节点之间的最短路径距离，形成一个“重心图”的距离表。
小社区处理: 代码中将不属于任何大型社区的节点（或属于小型社区的节点）简单地吸收到最近的大型社区中。
查询:
如果两个查询节点位于同一个社区内，直接使用该社区的局部PLL进行查询。
如果两个查询节点位于不同社区，则通过“节点A -> 其社区重心 -> 另一个社区重心 -> 节点B”的路径进行近似计算。
改进点（待实现或未来工作）
当前混合实现中，对于不属于大集群的点（或属于小集群的点），以及跨集群的距离计算，存在进一步优化的空间：
非无标度区域处理: 那些不适合被现有大集群吸收的点，以及小集群本身，在原论文中通常被视为“非无标度区域”。这些点应该通过**中介中心性（Betweenness Centrality）**等指标，单独选择额外的地标，并构建另一个独立的PLL索引来处理，而不是简单地吸收到最近的大集群。
跨集群距离精度: 目前跨集群的距离计算是基于“重心间的距离”进行近似的，这可能导致距离估算偏大。更精确的方法可以引入第二个PLL索引，该索引选择具有高中介中心性的节点作为地标，专门用于处理跨集群的路径。
安装
本项目依赖于以下Python库：
networkx
torch
torch_geometric
tqdm
numpy
pandas
python-louvain (community 库)
您可以使用 pip 安装所有依赖项：
pip install networkx torch torch_geometric tqdm numpy pandas python-louvain
