from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
import torch_geometric.utils
from networkit.nxadapter import nx2nk
from torch_geometric.data import Data, Dataset
from torch_scatter import (
    scatter_min,
    scatter_max,
    scatter_mean,
    scatter_std,
    scatter_sum,
)
from tqdm import tqdm

from descriptors import *


def _extract_single_graph_features(
    data: Data,
    degree_sum: bool,
    shortest_paths: bool,
    edge_betweenness: bool,
    degree_centrality: bool,
    closeness: bool,
    local_clustering_coefficient: bool,
    pagerank: bool,
    eigenvector_centrality: bool,
    algebraic_distance: bool,
    diameter: bool,
    density: bool,
    preferential_attachment: bool,
    common_neighbor: bool,
    katz_index: bool,
    jaccard_index: bool,
    adjusted_rand: bool,
    adamic_adar: bool,
    local_degree_score: bool,
    local_similarity_score: bool,
    scan: bool
) -> np.array:
    # adapted from PyTorch Geometric
    row, col = data.edge_index
    N = data.num_nodes

    deg = torch_geometric.utils.degree(row, N, dtype=torch.float)
    deg_col = deg[col]

    deg_min, _ = scatter_min(deg_col, row, dim_size=N)
    deg_min[deg_min > 10000] = 0
    deg_max, _ = scatter_max(deg_col, row, dim_size=N)
    deg_max[deg_max < -10000] = 0
    deg_mean = scatter_mean(deg_col, row, dim_size=N)
    deg_stddev = scatter_std(deg_col, row, dim_size=N)

    ldp_features = [
        deg.numpy(),
        deg_min.numpy(),
        deg_max.numpy(),
        deg_mean.numpy(),
        deg_stddev.numpy(),
    ]

    if degree_sum:
        deg_sum = scatter_sum(deg_col, row, dim_size=N)
        deg_sum = deg_sum.numpy()
        ldp_features.append(deg_sum)

    if any(
        [
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
        ]
    ):
        graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
        graph = nx2nk(graph)
        graph.indexEdges()

    if shortest_paths:
        sp_lengths = calculate_shortest_paths(graph)
        ldp_features.append(sp_lengths)

    if edge_betweenness:
        eb = calculate_edge_betweenness(graph)
        ldp_features.append(eb)

    if degree_centrality:
        dc = calculate_degree_centrality(graph)
        ldp_features.append(dc)

    if closeness:
        c = calculate_closeness(graph)
        ldp_features.append(c)

    if local_clustering_coefficient:
        lcc = calculate_local_clustering_coefficient(graph)
        ldp_features.append(lcc)

    if pagerank:
        pr = calculate_pagerank(graph)
        ldp_features.append(pr)

    if eigenvector_centrality:
        ec = calculate_eigenvector_centrality(graph)
        ldp_features.append(ec)

    if algebraic_distance:
        ad = calculate_algebraic_distance(graph)
        ldp_features.append(ad)

    if diameter:
        d = calculate_diameter(graph)
        ldp_features.append(d)

    if density:
        d = calculate_density(graph)
        ldp_features.append(d)

    if preferential_attachment:
        pa = calculate_preferential_attachment_index(graph)
        ldp_features.append(pa)

    if common_neighbor:
        cn = calculate_common_neighbor_index(graph)
        ldp_features.append(cn)

    if katz_index:
        ki = calculate_katz_index(graph)
        ldp_features.append(ki)

    if jaccard_index:
        ji = calculate_jaccard_index(graph)
        ldp_features.append(ji)

    if adjusted_rand:
        ar = calculate_adjusted_rand_index(graph)
        ldp_features.append(ar)

    if adamic_adar:
        aa = calculate_adamic_adar_index(graph)
        ldp_features.append(aa)

    if local_degree_score:
        lds = calculate_local_degree_score(graph)
        ldp_features.append(lds)

    if local_similarity_score:
        lss = calculate_local_similarity_score(graph)
        ldp_features.append(lss)

    if scan:
        scan = calculate_scan(graph)
        ldp_features.append(scan)

    # make sure that all features have the same dtype
    ldp_features = [feature.astype(np.float32) for feature in ldp_features]

    return ldp_features


def extract_features(
    dataset: Dataset,
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
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calculates LDP features distributions for each graph, i.e. neighborhood
    degrees, mins, maxes, means and standard deviations. Optionally also
    calculates additional features.

    Returns Pandas DataFrame, where each column is of dtype np.ndarray, holding
    feature distribution.
    """
    if verbose:
        print("Extracting LDP features")

    iterable = tqdm(dataset) if verbose else dataset
    data = [
        _extract_single_graph_features(
            data,
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
            scan,
        )
        for data in iterable
    ]

    columns = [
        "deg",
        "deg_min",
        "deg_max",
        "deg_mean",
        "deg_stddev",
    ]
    if degree_sum:
        columns.append("deg_sum")
    if shortest_paths:
        columns.append("shortest_paths")
    if edge_betweenness:
        columns.append("edge_betweenness")
    if degree_centrality:
        columns.append("degree_centrality")
    if closeness:
        columns.append("closeness")
    if local_clustering_coefficient:
        columns.append("local_clustering_coefficient")
    if pagerank:
        columns.append("pagerank")
    if eigenvector_centrality:
        columns.append("eigenvector_centrality")
    if algebraic_distance:
        columns.append("algebraic_distance")
    if diameter:
        columns.append("diameter")
    if density:
        columns.append("density")
    if preferential_attachment:
        columns.append("preferential_attachment")
    if common_neighbor:
        columns.append("common_neighbor")
    if katz_index:
        columns.append("katz_index")
    if jaccard_index:
        columns.append("jaccard_index")
    if adjusted_rand:
        columns.append("adjusted_rand_index")
    if adamic_adar:
        columns.append("adamic_adar_index")
    if local_degree_score:
        columns.append("local_degree_score")
    if local_similarity_score:
        columns.append("local_similarity_score")
    if scan:
        columns.append("scan_structural_similarity_score")

    return pd.DataFrame(data, columns=columns)


def process_row(
    row,
    columns: list[str],
    n_bins: int,
    normalization: Literal["none", "graph", "dataset"] = "graph",
    aggregation: Literal["histogram", "EDF"] = "histogram",
    log_degree: bool = False,
):
    x = np.empty(len(columns) * n_bins, dtype=np.float32)

    # features that use logarithm of values if log_degree is True
    log_features = [
        "deg",
        "deg_min",
        "deg_max",
        "deg_mean",
    ]

    col_start = 0
    col_end = n_bins

    for col_idx, col_name in enumerate(columns):
        values = row[col_idx]

        if log_degree is True and col_name in log_features:
            # add small value to avoid problems with degree 0
            values = np.log(values + 1e-3)

        density = True if normalization == "graph" else None

        # assume "histogram" aggregation by default, since we have to use some
        # aggregation, and EDF is just used on top of histogram
        values, _ = np.histogram(values, bins=n_bins, density=density)

        if aggregation == "EDF":
            # calculate empirical CDF from histogram bins
            values = np.cumsum(values)
            if density:
                # normalize again after summation if needed
                values /= values.max()

        x[col_start:col_end] = values
        col_start += n_bins
        col_end += n_bins

    return x


def calculate_features_matrix(
    ldp_features: pd.DataFrame,
    n_bins: int,
    normalization: Literal["none", "graph", "dataset"] = "graph",
    aggregation: Literal["histogram", "EDF"] = "histogram",
    log_degree: bool = False,
) -> np.ndarray:
    if normalization == "dataset":
        # turn off Pandas warning about setting with copy, we know what we're doing
        pd.options.mode.chained_assignment = None

        for col_name in ldp_features.columns:
            # select max absolute feature value among all values of that feature
            # in the dataset, and normalize using it
            feature_values = ldp_features[col_name].to_numpy()
            max_value = np.max([np.max(np.abs(vals)) for vals in feature_values])
            feature_values = [vals / max_value for vals in feature_values]
            ldp_features[col_name] = feature_values

        pd.options.mode.chained_assignment = "warn"

    args = dict(
        columns=ldp_features.columns,
        n_bins=n_bins,
        normalization=normalization,
        aggregation=aggregation,
        log_degree=log_degree,
    )

    n_jobs = joblib.cpu_count(only_physical_cores=True)
    parallel = joblib.Parallel(n_jobs=n_jobs, backend="loky")
    jobs = (joblib.delayed(process_row)(row, **args) for row in ldp_features.to_numpy())
    rows = parallel(jobs)

    X = np.stack(rows)

    return X
