import gc
import time

import numpy as np
from sklearn.metrics import accuracy_score

import wandb
from caching import try_loading_cached_features, cache_features
from data_loading import load_dataset, load_dataset_splits
from feature_extraction import extract_features, calculate_features_matrix
from models import get_model


def perform_experiment(
        dataset_name: str,
        degree_sum: bool = False,
        shortest_paths: bool = False,
        edge_betweenness: bool = False,
        jaccard_index: bool = False,
        adjusted_rand: bool = False,
        adamic_adar: bool = False,
        local_degree_score: bool = False,
        local_similarity_score: bool = False,
        n_bins: int = 50,
        normalization: str = "none",
        aggregation: str = "histogram",
        log_degree: bool = False,
        model_type: str = "RandomForest",
        use_features_cache: bool = True,
        verbose: bool = False,
):
    start = time.time()

    dataset = load_dataset(dataset_name)

    if use_features_cache:
        features = try_loading_cached_features(
            dataset_name,
            degree_sum=degree_sum,
            shortest_paths=shortest_paths,
            edge_betweenness=edge_betweenness,
            jaccard_index=jaccard_index,
            adjusted_rand=adjusted_rand,
            adamic_adar=adamic_adar,
            local_degree_score=local_degree_score,
            local_similarity_score=local_similarity_score
        )
    else:
        features = None

    if not use_features_cache or features is None:
        features = extract_features(
            dataset,
            degree_sum=degree_sum,
            shortest_paths=shortest_paths,
            edge_betweenness=edge_betweenness,
            jaccard_index=jaccard_index,
            adjusted_rand=adjusted_rand,
            adamic_adar=adamic_adar,
            local_degree_score=local_degree_score,
            local_similarity_score=local_similarity_score,
            verbose=verbose,
        )

        if use_features_cache:
            cache_features(
                features,
                dataset_name=dataset_name,
                degree_sum=degree_sum,
                shortest_paths=shortest_paths,
                edge_betweenness=edge_betweenness,
                jaccard_index=jaccard_index,
                adjusted_rand=adjusted_rand,
                adamic_adar=adamic_adar,
                local_degree_score=local_degree_score,
                local_similarity_score=local_similarity_score
            )

    y = np.array(dataset.data.y)
    del dataset
    gc.collect()

    splits = load_dataset_splits(dataset_name)
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
    acc_stddev = np.std(test_metrics)
    total_time = time.time() - start

    wandb.log({
        'time': round(total_time, 2),
    })

    return acc_mean, acc_stddev
