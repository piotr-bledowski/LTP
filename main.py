import argparse
import os
import sys
import warnings
import pickle
from pathlib import Path
from typing import Union
from time import time
import matplotlib.pyplot as plt
from feature_extraction import extract_features
from data_loading import DATASET_NAMES
from perform_experiment import perform_experiment, perform_experiment_calculate_importance, perform_experiment_calculate_importance_single_split, perform_experiment_single_split
import pandas as pd
from data_loading import load_dataset, load_dataset_splits
from caching import cache_features
import numpy as np
# the only warning raised is ConvergenceWarning for linear SVM, which is
# acceptable (max_iter is already higher than default); unfortunately, we
# have to do this globally for all warnings to affect child processes in
# cross-validation
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # also affect subprocesses


def ensure_bool(data: Union[bool, str]) -> bool:
    if isinstance(data, bool):
        return data
    elif data.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif data.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Local Topological Profile")
    parser.add_argument(
        "--dataset_name",
        choices=[
            "all",
            "DD",
            "NCI1",
            "PROTEINS_full",
            "ENZYMES",
            "IMDB-BINARY",
            "IMDB-MULTI",
            "REDDIT-BINARY",
            "REDDIT-MULTI-5K",
            "COLLAB",
        ],
        default="all",
        help="Dataset name, use 'all' to run the entire benchmark.",
    )
    parser.add_argument(
        "--iterative_feature_selection",
        type=ensure_bool,
        default=False,
        help="Use iterative feature selection instead of importance-based selection?",
    )
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.001,
        help="Minimum accuracy improvement for iterative feature selection.",
    )
    parser.add_argument(
        "--max_features_iterative",
        type=int,
        default=None,
        help="Maximum features for iterative selection (None for no limit).",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of times to run the entire iterative selection process per dataset.",
    )
    parser.add_argument(
        "--degree_sum",
        type=ensure_bool,
        default=False,
        help="Add degree sum feature from LDP?",
    )
    parser.add_argument(
        "--shortest_paths",
        type=ensure_bool,
        default=False,
        help="Add shortest paths feature from LDP?",
    )
    parser.add_argument(
        "--edge_betweenness",
        type=ensure_bool,
        default=True,
        help="Add edge betweenness centrality proposed in LTP?",
    )
    parser.add_argument(
        "--jaccard_index",
        type=ensure_bool,
        default=True,
        help="Add Jaccard Index proposed in LTP?",
    )
    parser.add_argument(
        "--local_degree_score",
        type=ensure_bool,
        default=True,
        help="Add Local Degree Score proposed in LTP?",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="Number of bins for aggregation.",
    )
    parser.add_argument(
        "--normalization",
        choices=[
            "none",
            "graph",
            "dataset",
        ],
        default="none",
        help="Normalization scheme.",
    )
    parser.add_argument(
        "--aggregation",
        choices=[
            "histogram",
            "EDF",
        ],
        default="histogram",
        help="Aggregation scheme.",
    )
    parser.add_argument(
        "--log_degree",
        type=bool,
        default=False,
        help="Use log scale for degree features from LDP?",
    )
    parser.add_argument(
        "--model_type",
        choices=[
            "LinearSVM",
            "KernelSVM",
            "RandomForest",
        ],
        default="RandomForest",
        help="Classification algorithm to use.",
    )
    parser.add_argument(
        "--tune_feature_extraction_hyperparams",
        type=bool,
        default=False,
        help="Perform hyperparameter tuning for feature extraction?",
    )
    parser.add_argument(
        "--tune_model_hyperparams",
        type=bool,
        default=False,
        help="Perform hyperparameter tuning for classification model?",
    )
    parser.add_argument(
        "--use_features_cache",
        type=bool,
        default=True,
        help="Cache calculated features to speed up subsequent experiments?",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Should print out verbose output?",
    )
    parser.add_argument(
        "--single_split_selection",
        type=ensure_bool,
        default=True,
        help="Use single split for importance-based feature selection? When enabled, runs algorithm on ALL splits sequentially and aggregates results.",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="[DEPRECATED - not used in current implementation] Index of the split to use for single-split feature selection.",
    )

    return parser.parse_args()

def save_dataset_labels(dataset):
    path = f"y/{dataset.name}.npy"
    if not os.path.exists(path):
        y = np.array(dataset.data.y)
        np.save(path, y)

def save_node_num(dataset):
    path = f"y/{dataset.name}_node_num.npy"
    if not os.path.exists(path):
        y = np.array(dataset.data.y)
        np.save(path, y)

def create_cached_features(datasets):
    #if not os.path.exists('features_cache') or len(os.listdir('features_cache')) == 0:
    Path("y").mkdir(exist_ok=True)

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        save_dataset_labels(dataset)
        
        # First extract and cache base features (LDP + node statistics)
        base_features = extract_features(dataset=dataset)
        cache_features(base_features, dataset_name, is_base=True)

        # Then extract and cache each topological feature separately
        params = {
            'degree_sum': False, 'shortest_paths': False, 'edge_betweenness': False,
            'degree_centrality': False, 'local_clustering_coefficient': False,
            'pagerank': False, 'eigenvector_centrality': False, 'algebraic_distance': False,
            'diameter': False, 'density': False, 'preferential_attachment': False,
            'common_neighbor': False, 'katz_index': False, 'jaccard_index': False,
            'adjusted_rand': False, 'adamic_adar': False, 'local_degree_score': False,
            'local_similarity_score': False, 'scan': False,
        }

        for feature_name in params.keys():
            params[feature_name] = True
            extracted_data = extract_features(dataset=dataset, **params)
            
            # Get only the topological feature columns
            feature_cols = [col for col in extracted_data.columns if col.startswith(feature_name)]
            feature_data = extracted_data[feature_cols]
            
            cache_features(feature_data, dataset_name, feature_name=feature_name)
            params[feature_name] = False



if __name__ == "__main__":
    plots_dir = Path("plots") / "feature_importance"
    plots_dir.mkdir(parents=True, exist_ok=True)
    args = parse_args()

    datasets = ['DD', 'NCI1', 'PROTEINS_full', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']
    ldp_features = ['deg max', 'deg', 'deg min', 'deg mean', 'deg stddev']

    #create_cached_features(datasets)

    # Handle dataset selection
    if args.dataset_name == "all":
        selected_datasets = datasets
    else:
        selected_datasets = [args.dataset_name]

    # Check if iterative feature selection is requested
    if args.iterative_feature_selection:
        print("Running iterative feature selection...")
        from iterative_feature_selection import run_iterative_feature_selection_experiment
        
        run_iterative_feature_selection_experiment(
            datasets=selected_datasets,
            model_type=args.model_type,
            min_improvement=args.min_improvement,
            max_features=args.max_features_iterative,
            verbose=args.verbose
        )
        print("Iterative feature selection completed!")
        
    else:
        # Original importance-based feature selection
        print("Running importance-based feature selection...")
        print(f"Available modes:")
        print(f"  - Multi-split averaging (default): Averages importance across all splits, evaluates on all splits")
        print(f"  - Single-split mode: Runs algorithm on each split separately, aggregates results across all runs")
        print(f"Current mode: {'Single-split sequential' if args.single_split_selection else 'Multi-split averaging'}")
        print()
        
        # Check if single-split mode is enabled
        if args.single_split_selection:
            print("Using single-split mode - running on ALL splits sequentially")
            
            # Prepare results collection for CSV output
            all_results = []
            
            for dataset_name in selected_datasets:
                print(f"\nProcessing {dataset_name} - running feature selection on all splits")
                
                # Load dataset splits to determine how many splits we have
                splits = load_dataset_splits(dataset_name)
                num_splits = len(splits)
                print(f"Dataset {dataset_name} has {num_splits} splits")
                
                # Collect results from all splits
                split_results = []
                split_times = []
                split_accuracies = []
                split_feature_selections = []
                
                dataset_start_time = time()
                
                # Run feature selection on each split
                for split_idx in range(num_splits):
                    print(f"\n--- Split {split_idx + 1}/{num_splits} ---")
                    
                    # Initialize parameters for this split
                    best_params = {
                        'degree sum': False,
                        'shortest paths': False,
                        'edge betweenness': False,
                        'degree centrality': False,
                        'local clustering coefficient': False,
                        'pagerank': False,
                        'eigenvector centrality': False,
                        'algebraic distance': False,
                        'diameter': False,
                        'density': False,
                        'preferential attachment': False,
                        'common neighbor': False,
                        'katz index': False,
                        'jaccard index': False,
                        'adjusted rand': False,
                        'adamic adar': False,
                        'local degree score': False,
                        'local similarity score': False,
                        'scan': False,
                    }

                    best_acc = 0
                    split_start = time()

                    # Use pre-computed feature importance instead of recalculating
                    if args.verbose:
                        print(f"Loading pre-computed feature importance for {dataset_name}...")
                    
                    try:
                        # Load pre-computed importance (same as multi-split approach)
                        with open(os.path.join('plots', 'feature_importance', f'{dataset_name}.pkl'), 'rb') as handle:
                            importance_data = pickle.load(handle)
                            d = importance_data.to_dict('records')[0]
                            d = sorted(d.items(), key=lambda x: x[1], reverse=True)
                            imp = [x for x in d if x[0] not in ldp_features]

                        if args.verbose:
                            print(f"Top 5 features by importance: {[x[0] for x in imp[:5]]}")

                        # Greedy selection using single split evaluation
                        for i in range(len(imp)):
                            params = best_params.copy()
                            next_descriptor = imp[i][0]
                            params[next_descriptor] = True

                            acc = perform_experiment_single_split(
                                model_type=args.model_type,
                                dataset_name=dataset_name,
                                split_idx=split_idx,
                                verbose=args.verbose,
                                degree_sum=params['degree sum'],
                                shortest_paths=params['shortest paths'],
                                edge_betweenness=params['edge betweenness'],
                                degree_centrality=params['degree centrality'],
                                local_clustering_coefficient=params['local clustering coefficient'],
                                pagerank=params['pagerank'],
                                eigenvector_centrality=params['eigenvector centrality'],
                                algebraic_distance=params['algebraic distance'],
                                diameter=params['diameter'],
                                density=params['density'],
                                preferential_attachment=params['preferential attachment'],
                                common_neighbor=params['common neighbor'],
                                katz_index=params['katz index'],
                                jaccard_index=params['jaccard index'],
                                adjusted_rand=params['adjusted rand'],
                                adamic_adar=params['adamic adar'],
                                local_degree_score=params['local degree score'],
                                local_similarity_score=params['local similarity score'],
                                scan=params['scan']
                            )

                            if acc > best_acc:
                                best_acc = acc
                                best_params = params
                                if args.verbose:
                                    true_best_params = [k for k, v in best_params.items() if v]
                                    print(f"New best: {best_acc:.4f} with {true_best_params}")

                        split_time = round(time() - split_start, 2)
                        
                        # Store results for this split
                        selected_features = [k for k, v in best_params.items() if v]
                        split_results.append({
                            'split_idx': split_idx,
                            'accuracy': best_acc,
                            'time': split_time,
                            'selected_features': selected_features,
                            'num_features': len(selected_features),
                            'feature_params': best_params.copy()
                        })
                        
                        split_accuracies.append(best_acc)
                        split_times.append(split_time)
                        split_feature_selections.append(selected_features)
                        
                        print(f"Split {split_idx}: Acc={best_acc:.4f}, Time={split_time}s, Features={len(selected_features)}")
                        
                    except FileNotFoundError:
                        print(f"ERROR: Pre-computed importance file not found for {dataset_name}")
                        print(f"Expected file: plots/feature_importance/{dataset_name}.pkl")
                        print("Please ensure you have pre-computed feature importance files.")
                        break
                    except Exception as e:
                        print(f"Error processing split {split_idx}: {e}")
                        if args.verbose:
                            import traceback
                            print(f"Full traceback: {traceback.format_exc()}")
                        continue
                
                # Calculate aggregated statistics
                if split_accuracies:  # Only if we have successful results
                    total_dataset_time = round(time() - dataset_start_time, 2)
                    acc_mean = np.mean(split_accuracies)
                    acc_std = np.std(split_accuracies)
                    time_mean = np.mean(split_times)
                    time_std = np.std(split_times)
                    
                    # Calculate feature selection frequency
                    all_possible_features = [
                        'degree sum', 'shortest paths', 'edge betweenness', 'degree centrality',
                        'local clustering coefficient', 'pagerank', 'eigenvector centrality',
                        'algebraic distance', 'diameter', 'density', 'preferential attachment',
                        'common neighbor', 'katz index', 'jaccard index', 'adjusted rand',
                        'adamic adar', 'local degree score', 'local similarity score', 'scan'
                    ]
                    
                    feature_frequencies = {}
                    for feature in all_possible_features:
                        count = sum(1 for selected in split_feature_selections if feature in selected)
                        feature_frequencies[feature] = count / len(split_feature_selections)
                    
                    # Find most commonly selected features
                    most_common_features = sorted(feature_frequencies.items(), key=lambda x: x[1], reverse=True)
                    selected_features_str = '; '.join([f"{feat}({freq:.2f})" for feat, freq in most_common_features if freq > 0])
                    
                    # Get baseline accuracy (assuming it's the LDP-only performance)
                    baseline_acc, _ = perform_experiment(
                        model_type=args.model_type,
                        dataset_name=dataset_name,
                        verbose=False,
                        degree_sum=False, shortest_paths=False, edge_betweenness=False,
                        degree_centrality=False, local_clustering_coefficient=False,
                        pagerank=False, eigenvector_centrality=False, algebraic_distance=False,
                        diameter=False, density=False, preferential_attachment=False,
                        common_neighbor=False, katz_index=False, jaccard_index=False,
                        adjusted_rand=False, adamic_adar=False, local_degree_score=False,
                        local_similarity_score=False, scan=False
                    )
                    
                    improvement = acc_mean - baseline_acc
                    
                    # Prepare row for CSV
                    csv_row = {
                        'Dataset': dataset_name,
                        'Accuracy_Mean': acc_mean,
                        'Accuracy_Std': acc_std,
                        'Baseline_Accuracy': baseline_acc,
                        'Total_Improvement': improvement,
                        'Time_Mean_Seconds': time_mean,
                        'Time_Std_Seconds': time_std,
                        'Total_Time_Seconds': total_dataset_time,
                        'Num_Splits_Processed': len(split_accuracies),
                        'Selected_Features_Frequencies': selected_features_str,
                        'Model_Type': args.model_type
                    }
                    
                    # Add individual feature frequencies
                    for feature in all_possible_features:
                        csv_row[f'Feature_{feature.replace(" ", "_")}'] = feature_frequencies[feature]
                    
                    all_results.append(csv_row)
                    
                    print(f"\n=== {dataset_name} Summary ===")
                    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
                    print(f"Baseline: {baseline_acc:.4f} (improvement: +{improvement:.4f})")
                    print(f"Time per split: {time_mean:.1f} ± {time_std:.1f} seconds")
                    print(f"Total time: {total_dataset_time:.1f} seconds")
                    print(f"Splits processed: {len(split_accuracies)}/{num_splits}")
                    print(f"Most common features: {[feat for feat, freq in most_common_features[:3] if freq > 0.3]}")

            # Save aggregated results to CSV
            if all_results:
                os.makedirs('results', exist_ok=True)
                results_df = pd.DataFrame(all_results)
                csv_filename = f'single_split_results_{args.model_type.lower()}.csv'
                csv_path = os.path.join('results', csv_filename)
                results_df.to_csv(csv_path, index=False)
                
                print(f"\n=== FINAL RESULTS ===")
                print(f"Results saved to: {csv_filename}")
                print(f"Processed {len(all_results)} datasets")
                print("\nSummary:")
                for _, row in results_df.iterrows():
                    print(f"{row['Dataset']}: {row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f} "
                          f"(+{row['Total_Improvement']:.4f} vs baseline)")
                
                # Also save detailed split results for debugging
                detailed_filename = f'single_split_detailed_{args.model_type.lower()}.pkl'
                with open(os.path.join('results', detailed_filename), 'wb') as f:
                    pickle.dump({
                        'summary': all_results,
                        'split_details': split_results if 'split_results' in locals() else []
                    }, f)
                print(f"Detailed results saved to: {detailed_filename}")
            else:
                print("No results to save - all datasets failed processing")
        
        else:
            # Original multi-split averaging approach
            print("Using multi-split averaging approach...")
        
        # for dataset_name in datasets:
        #     print(dataset_name)
        #     importances = perform_experiment_calculate_importance(
        #         dataset_name=dataset_name,
        #         verbose=False,
        #         atom_features=True,
        #         degree_sum=True,
        #         shortest_paths=True,
        #         edge_betweenness=True,
        #         degree_centrality=True,
        #         local_clustering_coefficient=True,
        #         pagerank=True,
        #         eigenvector_centrality=True,
        #         algebraic_distance=True,
        #         diameter=True,
        #         density=True,
        #         preferential_attachment=True,
        #         common_neighbor=True,
        #         katz_index=True,
        #         jaccard_index=True,
        #         adjusted_rand=True,
        #         adamic_adar=True,
        #         local_degree_score=True,
        #         local_similarity_score=True,
        #         scan=True,
        #         plots_dir=plots_dir
        #     )
        #     all_feature_importances.append(importances)
        #
        # df = pd.concat(all_feature_importances, ignore_index=True)
        # df = pd.DataFrame(df.mean(axis=0)).transpose()
        # df.index = [""]
        # df.to_pickle(plots_dir / "DD.pkl")
        # df.plot.bar(rot=0)
        # plt.tight_layout()
        # plt.savefig(plots_dir / "DD.pdf")
        for dataset_name in selected_datasets:
            print(dataset_name)
            best_params = {
                'degree sum': False,
                'shortest paths': False,
                'edge betweenness': False,
                'degree centrality': False,
                'local clustering coefficient': False,
                'pagerank': False,
                'eigenvector centrality': False,
                'algebraic distance': False,
                'diameter': False,
                'density': False,
                'preferential attachment': False,
                'common neighbor': False,
                'katz index': False,
                'jaccard index': False,
                'adjusted rand': False,
                'adamic adar': False,
                'local degree score': False,
                'local similarity score': False,
                'scan': False,
            }

            best_acc = 0
            best_acc_std = 0

            start = time()

            with open(os.path.join('plots', 'feature_importance', f'{dataset_name}.pkl'), 'rb') as handle:
                b = pickle.load(handle)
                d = b.to_dict('records')[0]
                d = sorted(d.items(), key=lambda x: x[1], reverse=True)

                imp = [x for x in d if x[0] not in ldp_features]

                for i in range(len(imp)):
                    params = best_params.copy()
                    next_descriptor = imp[i][0]
                    params[next_descriptor] = True

                    acc_mean, acc_std = perform_experiment(
                        model_type=args.model_type,
                        dataset_name=dataset_name,
                        verbose=args.verbose,
                        degree_sum=params['degree sum'],
                        shortest_paths=params['shortest paths'],
                        edge_betweenness=params['edge betweenness'],
                        degree_centrality=params['degree centrality'],

                        local_clustering_coefficient=params['local clustering coefficient'],
                        pagerank=params['pagerank'],
                        eigenvector_centrality=params['eigenvector centrality'],
                        algebraic_distance=params['algebraic distance'],
                        diameter=params['diameter'],
                        density=params['density'],
                        preferential_attachment=params['preferential attachment'],
                        common_neighbor=params['common neighbor'],
                        katz_index=params['katz index'],
                        jaccard_index=params['jaccard index'],
                        adjusted_rand=params['adjusted rand'],
                        adamic_adar=params['adamic adar'],
                        local_degree_score=params['local degree score'],
                        local_similarity_score=params['local similarity score'],
                        scan=params['scan'],
                        plots_dir=plots_dir
                    )

                    if acc_mean > best_acc:
                        best_acc = acc_mean
                        best_params = params
                        best_acc_std = acc_std
                        true_best_params = [k for k, v in best_params.items() if v]
                        print(true_best_params)

            total_time = round(time() - start, 2)

            best_params['time'] = total_time
            best_params['acc_mean'] = best_acc
            best_params['acc_std'] = best_acc_std

            os.makedirs('results', exist_ok=True)

            with open(os.path.join('results', f'{dataset_name}_sum_rf.pkl'), 'wb') as f:
                pickle.dump(best_params, f)

    # all_feature_importances = []
    #
    # for dataset_name in datasets:
    #     print(dataset_name)
    #     importances = perform_experiment_calculate_importance(
    #         dataset_name=dataset_name,
    #         verbose=False,
    #         degree_sum=True,
    #         shortest_paths=True,
    #         edge_betweenness=True,
    #         degree_centrality=True,
    #         local_clustering_coefficient=True,
    #         pagerank=True,
    #         eigenvector_centrality=True,
    #         algebraic_distance=True,
    #         diameter=True,
    #         density=True,
    #         preferential_attachment=True,
    #         common_neighbor=True,
    #         katz_index=True,
    #         jaccard_index=True,
    #         adjusted_rand=True,
    #         adamic_adar=True,
    #         local_degree_score=True,
    #         local_similarity_score=True,
    #         scan=True,
    #         plots_dir=plots_dir
    #     )
    #     all_feature_importances.append(importances)
    #
    # df = pd.concat(all_feature_importances, ignore_index=True)
    # df = pd.DataFrame(df.mean(axis=0)).transpose()
    # df.index = [""]
    # df.to_pickle(plots_dir / "DD.pkl")
    # df.plot.bar(rot=0)
    # plt.tight_layout()
    # plt.savefig(plots_dir / "DD.pdf")
