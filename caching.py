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


def create_features_table(
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
    
    features = {
        degree_sum : False,
        shortest_paths : False,
        edge_betweenness :False,
        degree_centrality : False,
        closeness : False,
        local_clustering_coefficient : False,
        pagerank :False,
        eigenvector_centrality :False,
        algebraic_distance : False,
        diameter : False,
        density :False,
        preferential_attachment : False,
        common_neighbor : False,
        katz_index : False,
        jaccard_index :False,
        adjusted_rand : False,
        adamic_adar : False,
        local_degree_score : False,
        local_similarity_score : False,
        scan : False
    }
    
    # create dataframe, iterate through features, open files and add data
    df = pd.DataFrame() # TODO
    for feature in features.keys():
        features[feature] = True
        filename = _get_file_name(
            features[dataset_name],
            features[degree_sum],
            features[shortest_paths],
            features[edge_betweenness],
            features[degree_centrality],
            features[closeness],
            features[local_clustering_coefficient],
            features[pagerank],
            features[eigenvector_centrality],
            features[algebraic_distance],
            features[diameter],
            features[density],
            features[preferential_attachment],
            features[common_neighbor],
            features[katz_index],
            features[jaccard_index],
            features[adjusted_rand],
            features[adamic_adar],
            features[local_degree_score],
            features[local_similarity_score],
            features[scan]
        )
        features[feature] = False
        filepath = FEATURES_CACHE_DIR / filename

        try:
            feature_data = pd.read_pickle(filepath, compression="zstd")
            df.append(feature_data) # TODO
        except FileNotFoundError:
            pass
    return df


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
