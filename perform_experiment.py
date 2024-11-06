import gc
import time

import numpy as np
from sklearn.metrics import accuracy_score
from feature_engine.selection import DropConstantFeatures
import matplotlib.pyplot as plt
import wandb
from caching import try_loading_cached_features, cache_features
from data_loading import load_dataset, load_dataset_splits
from feature_extraction import extract_features, calculate_features_matrix
from models import get_model
import pandas as pd


def perform_experiment(
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
        scan: bool = False,
        n_bins: int = 50,
        normalization: str = "none",
        aggregation: str = "histogram",
        log_degree: bool = False,
        model_type: str = "RandomForest",
        use_features_cache: bool = True,
        verbose: bool = False,
        plots_dir: str = "plots"
):
    start = time.time()

    dataset = load_dataset(dataset_name)

    if use_features_cache:
        features = try_loading_cached_features(
            dataset_name,
            degree_sum=degree_sum,
            shortest_paths=shortest_paths,
            edge_betweenness=edge_betweenness,
            degree_centrality=degree_centrality,
            closeness=closeness,
            local_clustering_coefficient=local_clustering_coefficient,
            pagerank=pagerank,
            eigenvector_centrality=eigenvector_centrality,
            algebraic_distance=algebraic_distance,
            diameter=diameter,
            density=density,
            preferential_attachment=preferential_attachment,
            common_neighbor=common_neighbor,
            katz_index=katz_index,
            jaccard_index=jaccard_index,
            adjusted_rand=adjusted_rand,
            adamic_adar=adamic_adar,
            local_degree_score=local_degree_score,
            local_similarity_score=local_similarity_score,
            scan=scan
        )
    else:
        features = None

    if not use_features_cache or features is None:
        features = extract_features(
            dataset,
            degree_sum=degree_sum,
            shortest_paths=shortest_paths,
            edge_betweenness=edge_betweenness,
            degree_centrality=degree_centrality,
            closeness=closeness,
            local_clustering_coefficient=local_clustering_coefficient,
            pagerank=pagerank,
            eigenvector_centrality=eigenvector_centrality,
            algebraic_distance=algebraic_distance,
            diameter=diameter,
            density=density,
            preferential_attachment=preferential_attachment,
            common_neighbor=common_neighbor,
            katz_index=katz_index,
            jaccard_index=jaccard_index,
            adjusted_rand=adjusted_rand,
            adamic_adar=adamic_adar,
            local_degree_score=local_degree_score,
            local_similarity_score=local_similarity_score,
            scan=scan
        )

        if use_features_cache:
            cache_features(
                features,
                dataset_name=dataset_name,
                degree_sum=degree_sum,
                shortest_paths=shortest_paths,
                edge_betweenness=edge_betweenness,
                degree_centrality=degree_centrality,
                closeness=closeness,
                local_clustering_coefficient=local_clustering_coefficient,
                pagerank=pagerank,
                eigenvector_centrality=eigenvector_centrality,
                algebraic_distance=algebraic_distance,
                diameter=diameter,
                density=density,
                preferential_attachment=preferential_attachment,
                common_neighbor=common_neighbor,
                katz_index=katz_index,
                jaccard_index=jaccard_index,
                adjusted_rand=adjusted_rand,
                adamic_adar=adamic_adar,
                local_degree_score=local_degree_score,
                local_similarity_score=local_similarity_score,
                scan=scan
            )

    print("Features shape:", features.shape)
    y = np.array(dataset.data.y)
    # del dataset
    gc.collect()

    splits = load_dataset_splits(dataset_name)
    nodes_nums = [data.num_nodes for split in splits for data in dataset[split.train_idxs]]
    # del dataset
    n_bins = int(np.median(nodes_nums))
    print(n_bins)
    test_metrics = []

    importances = []
    for i, split in enumerate(splits):
        if verbose:
            print("Starting computing split", i)

        train_idxs = split.train_idxs
        test_idxs = split.test_idxs
        features_train = features.iloc[train_idxs, :]
        features_test = features.iloc[test_idxs, :]
        y_train = y[train_idxs]
        y_test = y[test_idxs]

        nodes_nums = [data.num_nodes for data in dataset[train_idxs]]
        n_bins = int(np.median(nodes_nums))
        n_bins = 60

        ldp_params = {
            "n_bins": n_bins,
            "normalization": normalization,
            "aggregation": aggregation,
            "log_degree": log_degree,
        }

        X_train = calculate_features_matrix(features_train, **ldp_params)
        X_test = calculate_features_matrix(features_test, **ldp_params)

        columns = []
        columns.extend([f"deg {i}" for i in range(n_bins)])
        columns.extend([f"deg_min {i}" for i in range(n_bins)])
        columns.extend([f"deg_max {i}" for i in range(n_bins)])
        columns.extend([f"deg_mean {i}" for i in range(n_bins)])
        columns.extend([f"deg_stddev {i}" for i in range(n_bins)])
        columns.extend([f"degree_sum {i}" for i in range(n_bins)])
        columns.extend([f"shortest_paths {i}" for i in range(n_bins)])
        columns.extend([f"edge_betweenness {i}" for i in range(n_bins)])
        columns.extend([f"degree_centrality {i}" for i in range(n_bins)])
        columns.extend([f"local_clustering_coefficient {i}" for i in range(n_bins)])
        columns.extend([f"pagerank {i}" for i in range(n_bins)])
        columns.extend([f"eigenvector_centrality {i}" for i in range(n_bins)])
        columns.extend([f"algebraic_distance {i}" for i in range(n_bins)])
        columns.extend([f"diameter {i}" for i in range(n_bins)])
        columns.extend([f"density {i}" for i in range(n_bins)])
        columns.extend([f"preferential_attachment {i}" for i in range(n_bins)])
        columns.extend([f"common_neighbor {i}" for i in range(n_bins)])
        columns.extend([f"katz_index {i}" for i in range(n_bins)])
        columns.extend([f"jaccard_index {i}" for i in range(n_bins)])
        columns.extend([f"adjusted_rand {i}" for i in range(n_bins)])
        columns.extend([f"adamic_adar {i}" for i in range(n_bins)])
        columns.extend([f"local_degree_score {i}" for i in range(n_bins)])
        columns.extend([f"local_similarity_score {i}" for i in range(n_bins)])
        columns.extend([f"scan {i}" for i in range(n_bins)])

        # df_train = pd.DataFrame(X_train, columns=columns)
        # print(X_train.shape)
        # dropper = DropConstantFeatures()
        # X_train = dropper.fit_transform(df_train).values
        # print(X_train.shape)

        model = get_model(model_type=model_type, verbose=verbose)
        model.fit(X_train, y_train)

        # y_pred = model.predict(X_test)
        # acc = accuracy_score(y_test, y_pred)
        # test_metrics.append(acc)
        importances.append(model.feature_importances_)

    importances = [np.ravel(imp) for imp in importances]

    max_length = max(len(imp) for imp in importances)

    padded_importances = [
        np.pad(imp, (0, max_length - len(imp)), 'constant', constant_values=0)
        for imp in importances
    ]
    importances = np.mean(np.array(padded_importances), axis=0).tolist()

    # total importance of each feature group
    # columns = dropper.get_feature_names_out()
    columns = [col.split(" ")[0].replace("_", " ") for col in columns]
    print(f"Len columns: {len(columns)}")
    print(f"Len importances: {len(importances)}")
    df = pd.DataFrame({"column": columns, "value": importances})
    importances = df.groupby("column").sum().transpose()
    print("xd")
    columns = [
        "deg",
        "deg min",
        "deg max",
        "deg mean",
        "deg stddev",
        "degree sum",
        "shortest paths",
        "edge betweenness",
        "degree centrality",
        "local clustering coefficient",
        "pagerank",
        "eigenvector centrality",
        "algebraic distance",
        "diameter",
        "density",
        "preferential attachment",
        "common neighbor",
        "katz index",
        "jaccard index",
        "adjusted rand",
        "adamic adar",
        "local degree score",
        "local similarity score",
        "scan",
    ]

    importances = importances[columns]
    importances.columns = columns
    importances.index = [""]

    plt.figure(figsize=(12, 8))  # Adjust these values as needed

    ax = importances.plot.bar(rot=0)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.subplots_adjust(right=0.75)

    filename = dataset_name.removeprefix("ogbg-mol")
    plt.savefig(plots_dir / f"{filename}.pdf", bbox_inches='tight', dpi=300)

    return importances