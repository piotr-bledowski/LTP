import os
from pathlib import Path
from typing import Optional

import pandas as pd

FEATURES_CACHE_DIR = Path("f_cache")


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
    **kwargs
) -> Optional[pd.DataFrame]:
    if not os.path.exists(FEATURES_CACHE_DIR):
        return None

    true_features = [str(k) for k, v in kwargs.items() if v]
    list_of_features = ["degree_sum", "shortest_paths", "edge_betweenness", "degree_centrality", "closeness", "local_clustering_coefficient", "pagerank", "eigenvector_centrality", "algebraic_distance", "diameter", "density", "preferential_attachment", "common_neighbor", "katz_index", "jaccard_index", "adjusted_rand", "adamic_adar", "local_degree_score", "local_similarity_score", "scan"]
    dict_of_features = {k: False for k in list_of_features}
    df = pd.DataFrame() 

    # create filenames for each single feature from true_features, this feature should be true and the rest false. order of list_of_features is good, for example name_dataset_1_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0.zst
    for feature in true_features:
        dict_of_features[feature] = True
        filename = _get_file_name(
            dataset_name,
            dict_of_features["degree_sum"],
            dict_of_features["shortest_paths"],
            dict_of_features["edge_betweenness"],
            dict_of_features["degree_centrality"],
            dict_of_features["closeness"],
            dict_of_features["local_clustering_coefficient"],
            dict_of_features["pagerank"],
            dict_of_features["eigenvector_centrality"],
            dict_of_features["algebraic_distance"],
            dict_of_features["diameter"],
            dict_of_features["density"],
            dict_of_features["preferential_attachment"],
            dict_of_features["common_neighbor"],
            dict_of_features["katz_index"],
            dict_of_features["jaccard_index"],
            dict_of_features["adjusted_rand"],
            dict_of_features["adamic_adar"],
            dict_of_features["local_degree_score"],
            dict_of_features["local_similarity_score"],
            dict_of_features["scan"]
        )
        dict_of_features[feature] = False
        filepath = FEATURES_CACHE_DIR / filename
        try:
            feature_data = pd.read_pickle(filepath, compression="zstd")
            log_features = ['deg', 'deg_min', 'deg_max', 'deg_mean', 'deg_stddev']
            if "deg" in df.columns:
                df = pd.concat([df, feature_data.drop(columns=log_features)], axis=1)
            else:
                df = pd.concat([df, feature_data], axis=1)
        except FileNotFoundError:
            pass
    return df

    
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
