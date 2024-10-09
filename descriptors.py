import numpy as np
from networkit.centrality import Betweenness, DegreeCentrality, Closeness, LocalClusteringCoefficient, PageRank, EigenvectorCentrality
from networkit.distance import APSP, AlgebraicDistance, Diameter
from networkit.graph import Graph
from networkit.linkprediction import JaccardIndex, AdjustedRandIndex, AdamicAdarIndex, CommonNeighborsIndex, KatzIndex, PreferentialAttachmentIndex
from networkit.sparsification import LocalDegreeScore, LocalSimilarityScore, TriangleEdgeScore, SCANStructuralSimilarityScore
from networkit.graphtools import GraphTools


def get_triangles(G: Graph):
    edge_triangles = TriangleEdgeScore(G)
    edge_triangles.run()
    return edge_triangles.scores()


def calculate_degree_centrality(graph: Graph) -> np.array:
    dc = DegreeCentrality(graph).run().scores()
    scores = np.array(dc, dtype=np.float32)
    return scores


def calculate_closeness(graph: Graph) -> np.array:
    c = Closeness(graph, normalized=True, checkConnectedness=False).run().scores()
    scores = np.array(c, dtype=np.float32)
    return scores


def calculate_local_clustering_coefficient(graph: Graph) -> np.array:
    lcc = LocalClusteringCoefficient(graph).run().scores()
    scores = np.array(lcc, dtype=np.float32)
    return scores


def calculate_pagerank(graph: Graph) -> np.array:
    pr = PageRank(graph).run().scores()
    scores = np.array(pr, dtype=np.float32)
    return scores


def calculate_eigenvector_centrality(graph: Graph) -> np.array:
    ec = EigenvectorCentrality(graph).run().scores()
    scores = np.array(ec, dtype=np.float32)
    return scores


def calculate_algebraic_distance(graph: Graph) -> np.array:
    ad = AlgebraicDistance(graph).getEdgeScores()
    scores = np.array(ad, dtype=np.float32)
    return scores


def calculate_diameter(graph: Graph) -> np.array:
    pr = Diameter(graph).run()
    return pr


def calculate_common_neighbor_index(graph: Graph) -> np.array:
    scores = [CommonNeighborsIndex(graph).run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    return scores


def calculate_katz_index(graph: Graph) -> np.array:
    scores = [KatzIndex(graph).run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    return scores


def calculate_preferential_attachment_index(graph: Graph) -> np.array:
    scores = [PreferentialAttachmentIndex(graph).run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    return scores


def calculate_scan(graph: Graph) -> np.array:
    pr = SCANStructuralSimilarityScore(graph, get_triangles(graph)).run().scores()
    scores = np.array(pr, dtype=np.float32)
    return scores


def calculate_density(graph: Graph) -> np.array:
    gt = GraphTools()
    d = gt.density(graph)
    return d


def calculate_shortest_paths(graph: Graph) -> np.array:
    # Networkit is faster than NetworkX for large graphs
    apsp = APSP(graph)
    apsp.run()
    path_lengths = apsp.getDistances(asarray=True)

    path_lengths = path_lengths.ravel()

    # filter out 0 length "paths" from node to itself
    path_lengths = path_lengths[np.nonzero(path_lengths)]

    # Networkit assigns extremely high values (~1e308) to mark infinite
    # distances for disconnected components, so we simply drop them
    path_lengths = path_lengths[path_lengths < 1e100]

    return path_lengths


def calculate_edge_betweenness(graph: Graph) -> np.ndarray:
    betweeness = Betweenness(graph, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    scores = np.array(scores, dtype=np.float32)
    return scores


def calculate_jaccard_index(graph: Graph) -> np.ndarray:
    jaccard_index = JaccardIndex(graph)
    scores = [jaccard_index.run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    return scores


def calculate_adjusted_rand_index(graph: Graph) -> np.ndarray:
    adjusted_rand_index = AdjustedRandIndex(graph)
    scores = [adjusted_rand_index.run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    return scores


def calculate_adamic_adar_index(graph: Graph) -> np.ndarray:
    adamic_adar_index = AdamicAdarIndex(graph)
    scores = [adamic_adar_index.run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    return scores


def calculate_local_degree_score(graph: Graph) -> np.ndarray:
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return np.array(scores, dtype=np.float32)


def calculate_local_similarity_score(graph: Graph) -> np.ndarray:
    lss = LocalSimilarityScore(graph, get_triangles(graph))
    lss.run()
    scores = lss.scores()
    return np.array(scores, dtype=np.float32)

