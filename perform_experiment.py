import gc
import time

import numpy as np
from sklearn.metrics import accuracy_score
from feature_engine.selection import DropConstantFeatures
import matplotlib.pyplot as plt
import wandb
from caching import try_loading_cached_features, cache_features, create_features_table
from data_loading import load_dataset, load_dataset_splits
from feature_extraction import extract_features, calculate_features_matrix
from models import get_model
import pandas as pd


def perform_experiment_calculate_importance(
        dataset_name: str,
        atom_features: bool = False,
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

    #dataset = load_dataset(dataset_name)

    features = create_features_table(
        dataset_name,
        atom_features=atom_features,
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

    print("Features")
    print(features.shape)

    #print("Features shape:", features.shape)
    #y = np.array(dataset.data.y)
    y = np.load(f'y/{dataset_name}.npy')
    # del dataset
    #gc.collect()

    splits = load_dataset_splits(dataset_name)
    nodes_nums = [data.num_nodes for split in splits for data in dataset[split.train_idxs]]
    # del dataset
    n_bins = int(np.median(nodes_nums))
    #print(n_bins)
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
        if atom_features:
            columns.extend([f"atom_features {i}" for i in range(n_bins)])
        if degree_sum:
            columns.extend([f"degree_sum {i}" for i in range(n_bins)])
        if shortest_paths:
            columns.extend([f"shortest_paths {i}" for i in range(n_bins)])
        if edge_betweenness:
            columns.extend([f"edge_betweenness {i}" for i in range(n_bins)])
        if degree_centrality:
            columns.extend([f"degree_centrality {i}" for i in range(n_bins)])
        if local_clustering_coefficient:
            columns.extend([f"local_clustering_coefficient {i}" for i in range(n_bins)])
        if pagerank:
            columns.extend([f"pagerank {i}" for i in range(n_bins)])
        if eigenvector_centrality:
            columns.extend([f"eigenvector_centrality {i}" for i in range(n_bins)])
        if algebraic_distance:
            columns.extend([f"algebraic_distance {i}" for i in range(n_bins)])
        if diameter:
            columns.extend([f"diameter {i}" for i in range(n_bins)])
        if density:
            columns.extend([f"density {i}" for i in range(n_bins)])
        if preferential_attachment:
            columns.extend([f"preferential_attachment {i}" for i in range(n_bins)])
        if common_neighbor:
            columns.extend([f"common_neighbor {i}" for i in range(n_bins)])
        if katz_index:
            columns.extend([f"katz_index {i}" for i in range(n_bins)])
        if jaccard_index:
            columns.extend([f"jaccard_index {i}" for i in range(n_bins)])
        if adjusted_rand:
            columns.extend([f"adjusted_rand {i}" for i in range(n_bins)])
        if adamic_adar:
            columns.extend([f"adamic_adar {i}" for i in range(n_bins)])
        if local_degree_score:
            columns.extend([f"local_degree_score {i}" for i in range(n_bins)])
        if local_similarity_score:
            columns.extend([f"local_similarity_score {i}" for i in range(n_bins)])
        if scan:
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
    #print(f"Len columns: {len(columns)}")
    #print(f"Len importances: {len(importances)}")
    print(f"Columns: {len(columns)}, values: {np.array(importances).shape}")
    df = pd.DataFrame({"column": columns, "value": importances})
    importances = df.groupby("column").sum().transpose()
    print("xd")
    columns = [
        "deg",
        "deg min",
        "deg max",
        "deg mean",
        "deg stddev",
    ]
    #     "degree sum",
    #     "shortest paths",
    #     "edge betweenness",
    #     "degree centrality",
    #     "local clustering coefficient",
    #     "pagerank",
    #     "eigenvector centrality",
    #     "algebraic distance",
    #     "diameter",
    #     "density",
    #     "preferential attachment",
    #     "common neighbor",
    #     "katz index",
    #     "jaccard index",
    #     "adjusted rand",
    #     "adamic adar",
    #     "local degree score",
    #     "local similarity score",
    #     "scan",
    # ]
    if atom_features:
        columns.append("atom features")
    if degree_sum:
        columns.append("degree sum")
    if shortest_paths:
        columns.append("shortest paths")
    if edge_betweenness:
        columns.append("edge betweenness")
    if degree_centrality:
        columns.append("degree centrality")
    if local_clustering_coefficient:
        columns.append("local clustering coefficient")
    if pagerank:
        columns.append("pagerank")
    if eigenvector_centrality:
        columns.append("eigenvector centrality")
    if algebraic_distance:
        columns.append("algebraic distance")
    if diameter:
        columns.append("diameter")
    if density:
        columns.append("density")
    if preferential_attachment:
        columns.append("preferential attachment")
    if common_neighbor:
        columns.append("common neighbor")
    if katz_index:
        columns.append("katz index")
    if jaccard_index:
        columns.append("jaccard index")
    if adjusted_rand:
        columns.append("adjusted rand")
    if adamic_adar:
        columns.append("adamic adar")
    if local_degree_score:
        columns.append("local degree score")
    if local_similarity_score:
        columns.append("local similarity score")
    if scan:
        columns.append("scan")
    
    importances = importances[columns]
    importances.columns = columns
    importances.index = [""]

    # filename = dataset_name.removeprefix("ogbg-mol")

    # importances.to_pickle(plots_dir / f'{filename}.pkl')

    # plt.figure(figsize=(12, 8))  # Adjust these values as needed

    # ax = importances.plot.bar(rot=0)

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # plt.tight_layout()

    # plt.subplots_adjust(right=0.75)

    # plt.savefig(plots_dir / f"{filename}.pdf", bbox_inches='tight', dpi=300)

    return importances


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
        model_type: str = "LightGBM",
        use_features_cache: bool = True,
        verbose: bool = False,
        plots_dir: str = "plots"
):
    start = time.time()

    features = create_features_table(
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

    #print("Features shape:", features.shape)
    path = f"y/{dataset_name}.npy"
    y = np.load(path)
    #gc.collect()

    splits = load_dataset_splits(dataset_name)
    n_bins = 60

    #print(n_bins)
    test_metrics = []

    for i, split in enumerate(splits):
        if verbose:
            print("Starting computing split", i)

        train_idxs = split.train_idxs
        test_idxs = split.test_idxs
        features_train = features.iloc[train_idxs, :]
        features_test = features.iloc[test_idxs, :]
        y_train = y[train_idxs]
        y_test = y[test_idxs]

        ldp_params = {
            "n_bins": n_bins,
            "normalization": normalization,
            "aggregation": aggregation,
            "log_degree": log_degree,
        }

        X_train = calculate_features_matrix(features_train, **ldp_params)
        X_test = calculate_features_matrix(features_test, **ldp_params)


        model = get_model(model_type=model_type, verbose=verbose)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        test_metrics.append(acc)

    acc_mean = np.mean(test_metrics)
    acc_std = np.std(test_metrics)

    print(f'{dataset_name}: acc_mean: {acc_mean}, acc_std: {acc_std}')

    return acc_mean, acc_std
