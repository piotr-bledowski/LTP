import os
from pathlib import Path
from typing import Optional

import pandas as pd

FEATURES_CACHE_DIR = Path("features_cache")


def try_loading_cached_features(
    dataset_name: str,
    degree_sum: bool = False,
    shortest_paths: bool = False,
    edge_betweenness: bool = False,
    degree_centrality: bool = False,
    closeness: bool = False,
    local_clustering_coefficient: bool = False,
    pagerank: bool = False,
    eigenvector_centrality: bool = False,
    algebraic_distance: bool = False,
    diameter: bool = False,
    density: bool = False,
    preferential_attachment: bool = False,
    common_neighbor: bool = False,
    katz_index: bool = False,
    jaccard_index: bool = False,
    adjusted_rand: bool = False,
    adamic_adar: bool = False,
    local_degree_score: bool = False,
    local_similarity_score: bool = False,
    scan: bool = False
) -> Optional[pd.DataFrame]:
    if not os.path.exists(FEATURES_CACHE_DIR):
        return None

    filename = _get_file_name(
        dataset_name,
        degree_sum,
        shortest_paths,
        edge_betweenness,
        degree_centrality,
        closeness,
        local_clustering_coefficient,
        pagerank,
        eigenvector_centrality,
        algebraic_distance,
        diameter,
        density,
        preferential_attachment,
        common_neighbor,
        katz_index,
        jaccard_index,
        adjusted_rand,
        adamic_adar,
        local_degree_score,
        local_similarity_score,
        scan
    )
    filepath = FEATURES_CACHE_DIR / filename

    try:
        return pd.read_pickle(filepath, compression="zstd")
    except FileNotFoundError:
        return None


def cache_features(
    features: pd.DataFrame,
    dataset_name: str,
    degree_sum: bool = False,
    shortest_paths: bool = False,
    edge_betweenness: bool = False,
    degree_centrality: bool = False,
    closeness: bool = False,
    local_clustering_coefficient: bool = False,
    pagerank: bool = False,
    eigenvector_centrality: bool = False,
    algebraic_distance: bool = False,
    diameter: bool = False,
    density: bool = False,
    preferential_attachment: bool = False,
    common_neighbor: bool = False,
    katz_index: bool = False,
    jaccard_index: bool = False,
    adjusted_rand: bool = False,
    adamic_adar: bool = False,
    local_degree_score: bool = False,
    local_similarity_score: bool = False,
    scan: bool = False
) -> None:
    FEATURES_CACHE_DIR.mkdir(exist_ok=True)

    filename = _get_file_name(
        dataset_name,
        degree_sum,
        shortest_paths,
        edge_betweenness,
        degree_centrality,
        closeness,
        local_clustering_coefficient,
        pagerank,
        eigenvector_centrality,
        algebraic_distance,
        diameter,
        density,
        preferential_attachment,
        common_neighbor,
        katz_index,
        jaccard_index,
        adjusted_rand,
        adamic_adar,
        local_degree_score,
        local_similarity_score,
        scan
    )
    filepath = FEATURES_CACHE_DIR / filename

    features.to_pickle(
        filepath, compression={"method": "zstd", "threads": -1}, protocol=5
    )


def _get_file_name(
    dataset_name: str,
    degree_sum: bool = False,
    shortest_paths: bool = False,
    edge_betweenness: bool = False,
    degree_centrality: bool = False,
    closeness: bool = False,
    local_clustering_coefficient: bool = False,
    pagerank: bool = False,
    eigenvector_centrality: bool = False,
    algebraic_distance: bool = False,
    diameter: bool = False,
    density: bool = False,
    preferential_attachment: bool = False,
    common_neighbor: bool = False,
    katz_index: bool = False,
    jaccard_index: bool = False,
    adjusted_rand: bool = False,
    adamic_adar: bool = False,
    local_degree_score: bool = False,
    local_similarity_score: bool = False,
    scan: bool = False
) -> str:
    filename = "_".join(
        [
            dataset_name,
            str(int(degree_sum)),
            str(int(shortest_paths)),
            str(int(edge_betweenness)),
            str(int(degree_centrality)),
            str(int(closeness)),
            str(int(local_clustering_coefficient)),
            str(int(pagerank)),
            str(int(eigenvector_centrality)),
            str(int(algebraic_distance)),
            str(int(diameter)),
            str(int(density)),
            str(int(preferential_attachment)),
            str(int(common_neighbor)),
            str(int(katz_index)),
            str(int(jaccard_index)),
            str(int(adjusted_rand)),
            str(int(adamic_adar)),
            str(int(local_degree_score)),
            str(int(local_similarity_score)),
            str(int(scan))
        ]
    )
    return f"{filename}.zst"
