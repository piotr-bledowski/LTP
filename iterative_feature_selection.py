import os
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from perform_experiment import perform_experiment
from caching import create_features_table
from data_loading import load_dataset_splits, load_dataset
from feature_extraction import calculate_features_matrix
from models import get_model

# All available topological features (excluding base LDP features which are always included)
TOPOLOGICAL_FEATURES = [
    'degree_sum', 'shortest_paths', 'edge_betweenness', 'degree_centrality',
    'local_clustering_coefficient', 'pagerank', 'eigenvector_centrality', 'algebraic_distance',
    'diameter', 'density', 'preferential_attachment', 'common_neighbor', 'katz_index',
    'jaccard_index', 'adjusted_rand', 'adamic_adar', 'local_degree_score',
    'local_similarity_score', 'scan'
]

def iterative_feature_selection(
    dataset_name: str,
    model_type: str = 'RandomForest',
    min_improvement: float = 0.001,
    max_features: Optional[int] = None,
    verbose: bool = True,
    plots_dir: str = "plots"
) -> Dict:
    """
    Performs iterative forward feature selection starting with no topological features.
    Iterates over different dataset splits and reports statistics across splits.
    
    Args:
        dataset_name: Name of the dataset
        model_type: Type of model to use for evaluation
        min_improvement: Minimum accuracy improvement to continue adding features
        max_features: Maximum number of features to add (None for no limit)
        verbose: Whether to print progress
        plots_dir: Directory for plots
        
    Returns:
        Dictionary with results including best feature set, accuracies, and timing
    """
    if verbose:
        print(f"Starting iterative feature selection for {dataset_name}")
        print(f"Model: {model_type}, Min improvement: {min_improvement}")
    
    start_time = time.time()
    
    # Load dataset splits
    splits = load_dataset_splits(dataset_name)
    n_splits = len(splits)
    
    if verbose:
        print(f"Number of splits: {n_splits}")
    
    # Store results from all splits
    all_split_results = []
    
    for split_idx, split in enumerate(splits):
        if verbose:
            print(f"\n{'='*40}")
            print(f"Split {split_idx + 1}/{n_splits}")
            print(f"{'='*40}")
        
        split_result = _single_iterative_selection_split(
            dataset_name=dataset_name,
            model_type=model_type,
            min_improvement=min_improvement,
            max_features=max_features,
            verbose=verbose,
            plots_dir=plots_dir,
            split_idx=split_idx,
            split=split
        )
        
        all_split_results.append(split_result)
        
        if verbose:
            print(f"Split {split_idx + 1} completed:")
            print(f"  Final accuracy: {split_result['final_accuracy']:.4f}")
            print(f"  Baseline accuracy: {split_result['baseline_acc']:.4f}")
            print(f"  Features selected: {len(split_result['selected_features'])}")
            print(f"  Selected: {sorted(list(split_result['selected_features']))}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics across all splits
    final_accuracies = [result['final_accuracy'] for result in all_split_results]
    baseline_accuracies = [result['baseline_acc'] for result in all_split_results]
    acc_mean = np.mean(final_accuracies)
    acc_std = np.std(final_accuracies, ddof=1)  # Sample standard deviation
    baseline_acc_mean = np.mean(baseline_accuracies)
    
    # Find the most common feature set or the one with best accuracy
    best_split_idx = np.argmax(final_accuracies)
    best_split = all_split_results[best_split_idx]
    
    # Count feature selection frequency across splits
    feature_counts = {feature: 0 for feature in TOPOLOGICAL_FEATURES}
    for result in all_split_results:
        for feature in result['selected_features']:
            feature_counts[feature] += 1
    
    # Create final feature set (features selected in majority of splits)
    majority_threshold = n_splits // 2
    consensus_features = {feature for feature, count in feature_counts.items() 
                         if count > majority_threshold}
    
    # Use best split's features if no consensus, or consensus features if they exist
    final_selected_features = consensus_features if consensus_features else best_split['selected_features']
    
    # Prepare final results with statistics across splits
    final_params = {feature: feature in final_selected_features for feature in TOPOLOGICAL_FEATURES}
    final_params.update({
        'time': total_time,
        'acc_mean': acc_mean,
        'acc_std': acc_std,
        'num_features': len(final_selected_features),
        'baseline_acc': baseline_acc_mean,
        'total_improvement': acc_mean - baseline_acc_mean,
        'n_splits': n_splits,
        'best_split_accuracy': best_split['final_accuracy'],
        'worst_split_accuracy': min(final_accuracies),
        'all_split_accuracies': final_accuracies,
        'all_baseline_accuracies': baseline_accuracies,
        'feature_selection_frequency': feature_counts,
        'consensus_features': sorted(list(consensus_features)),
        'best_split_features': sorted(list(best_split['selected_features'])),
        'all_split_results': all_split_results  # Detailed results from each split
    })
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS ACROSS {n_splits} SPLITS")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Mean accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"Best split accuracy: {max(final_accuracies):.4f}")
        print(f"Worst split accuracy: {min(final_accuracies):.4f}")
        print(f"Mean baseline accuracy: {baseline_acc_mean:.4f}")
        print(f"Mean improvement: {acc_mean - baseline_acc_mean:+.4f}")
        
        print(f"\nFeature Selection Frequency:")
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features:
            if count > 0:
                percentage = (count / n_splits) * 100
                print(f"  {feature}: {count}/{n_splits} splits ({percentage:.1f}%)")
        
        print(f"\nConsensus features (>50% of splits): {sorted(list(consensus_features)) if consensus_features else 'None'}")
        print(f"Best split features: {sorted(list(best_split['selected_features']))}")
        print(f"Final selected features: {sorted(list(final_selected_features))}")
    
    return final_params

def _single_iterative_selection_split(
    dataset_name: str,
    model_type: str,
    min_improvement: float,
    max_features: Optional[int],
    verbose: bool,
    plots_dir: str,
    split_idx: int,
    split
) -> Dict:
    """
    Run a single iteration of the iterative feature selection process on a specific split.
    
    IMPORTANT: This function now uses cross-validation on ONLY the training data
    to select features, keeping the test set completely separate until final evaluation.
    
    Returns:
        Dictionary with results from this single split
    """
    # Load dataset and labels for this specific evaluation
    dataset = load_dataset(dataset_name)
    y = np.load(f'y/{dataset_name}.npy')
    
    # Get train/test indices for this split
    train_idxs = split.train_idxs
    test_idxs = split.test_idxs
    y_train = y[train_idxs]
    y_test = y[test_idxs]
    
    # Initialize with no topological features (only base LDP features)
    selected_features = set()
    remaining_features = set(TOPOLOGICAL_FEATURES)
    
    # Fixed parameters for feature extraction
    n_bins = 60
    ldp_params = {
        "n_bins": n_bins,
        "normalization": "none",
        "aggregation": "histogram",
        "log_degree": False,
    }
    
    # Helper function to evaluate a feature set using cross-validation on TRAINING DATA ONLY
    def evaluate_feature_set_cv(feature_params: Dict[str, bool]) -> float:
        """
        Evaluate feature set using cross-validation on training data only.
        This prevents data leakage by keeping test set completely separate.
        """
        # Get features with the specified configuration
        features = create_features_table(dataset_name, **feature_params)
        if features is None:
            # Extract features if not cached
            from feature_extraction import extract_features
            features = extract_features(dataset, **feature_params, verbose=False)
        
        # Use ONLY training data for feature selection
        features_train = features.iloc[train_idxs, :]
        
        # Calculate feature matrices for training data only
        X_train = calculate_features_matrix(features_train, **ldp_params)
        
        # Normalize features for SVM models (fit on training data only)
        if model_type in ['LinearSVM', 'KernelSVM']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        
        # Perform cross-validation on training data only to evaluate feature set
        model = get_model(model_type=model_type, verbose=False)
        
        # Use 5-fold cross-validation on training data
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=1  # Avoid nested parallelization issues
        )
        
        # Return mean cross-validation score
        return np.mean(cv_scores)
    
    # Helper function to evaluate final selected features on test set (for final evaluation only)
    def evaluate_on_test_set(feature_params: Dict[str, bool]) -> float:
        """
        Evaluate the final selected feature set on the test set.
        This should ONLY be called once at the end for final evaluation.
        """
        # Get features with the specified configuration
        features = create_features_table(dataset_name, **feature_params)
        if features is None:
            # Extract features if not cached
            from feature_extraction import extract_features
            features = extract_features(dataset, **feature_params, verbose=False)
        
        # Split features into train and test
        features_train = features.iloc[train_idxs, :]
        features_test = features.iloc[test_idxs, :]
        
        # Calculate feature matrices
        X_train = calculate_features_matrix(features_train, **ldp_params)
        X_test = calculate_features_matrix(features_test, **ldp_params)
        
        # Normalize features for SVM models
        if model_type in ['LinearSVM', 'KernelSVM']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train model on full training set and evaluate on test set
        model = get_model(model_type=model_type, verbose=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    # Evaluate baseline with only base features using cross-validation
    baseline_params = {feature: False for feature in TOPOLOGICAL_FEATURES}
    baseline_acc_cv = evaluate_feature_set_cv(baseline_params)
    
    best_acc_cv = baseline_acc_cv
    iteration = 1
    
    # Track history for this split
    selection_history = []
    selection_history.append({
        'features': list(selected_features),
        'accuracy_cv': baseline_acc_cv,
        'improvement': 0.0
    })
    
    if verbose:
        print(f"  Baseline CV accuracy: {baseline_acc_cv:.4f}")
    
    while remaining_features and (max_features is None or len(selected_features) < max_features):
        if verbose:
            print(f"  Iteration {iteration}: Testing {len(remaining_features)} remaining features")
        
        best_candidate = None
        best_candidate_acc_cv = best_acc_cv
        
        # Test each remaining feature using cross-validation on training data only
        for feature in remaining_features:
            if verbose:
                print(f"    Testing feature: {feature}")
            
            # Create parameters with current selected features + candidate
            test_params = {f: False for f in TOPOLOGICAL_FEATURES}
            for selected_feature in selected_features:
                test_params[selected_feature] = True
            test_params[feature] = True
            
            # Evaluate with this feature set using cross-validation
            acc_cv = evaluate_feature_set_cv(test_params)
            
            if verbose:
                improvement = acc_cv - best_acc_cv
                print(f"      CV Accuracy: {acc_cv:.4f} (improvement: {improvement:+.4f})")
            
            # Check if this is the best candidate so far
            if acc_cv > best_candidate_acc_cv:
                best_candidate = feature
                best_candidate_acc_cv = acc_cv
        
        # Check if best candidate provides sufficient improvement
        improvement = best_candidate_acc_cv - best_acc_cv
        
        if improvement >= min_improvement:
            # Add the best candidate to selected features
            selected_features.add(best_candidate)
            remaining_features.remove(best_candidate)
            best_acc_cv = best_candidate_acc_cv
            
            selection_history.append({
                'features': list(selected_features),
                'accuracy_cv': best_acc_cv,
                'improvement': improvement,
                'added_feature': best_candidate
            })
            
            if verbose:
                print(f"    ✓ Added feature: {best_candidate}")
                print(f"      New best CV accuracy: {best_acc_cv:.4f} (improvement: {improvement:+.4f})")
        else:
            if verbose:
                print(f"    ✗ No feature provides sufficient improvement (best: {improvement:+.4f} < {min_improvement})")
            break
        
        iteration += 1
    
    # NOW evaluate the final selected features on the test set (this is the only time we touch test data)
    final_params = {f: False for f in TOPOLOGICAL_FEATURES}
    for feature in selected_features:
        final_params[feature] = True
    
    final_test_accuracy = evaluate_on_test_set(final_params)
    baseline_test_accuracy = evaluate_on_test_set(baseline_params)
    
    if verbose:
        print(f"  Final evaluation on test set:")
        print(f"    Selected features test accuracy: {final_test_accuracy:.4f}")
        print(f"    Baseline test accuracy: {baseline_test_accuracy:.4f}")
        print(f"    Test set improvement: {final_test_accuracy - baseline_test_accuracy:+.4f}")
    
    return {
        'selected_features': selected_features,
        'final_accuracy': final_test_accuracy,  # Test set accuracy with selected features
        'baseline_acc': baseline_test_accuracy,  # Test set accuracy with baseline features
        'final_cv_accuracy': best_acc_cv,  # Cross-validation accuracy used for selection
        'baseline_cv_accuracy': baseline_acc_cv,  # Cross-validation baseline accuracy
        'improvement': final_test_accuracy - baseline_test_accuracy,
        'cv_improvement': best_acc_cv - baseline_acc_cv,
        'iterations': iteration - 1,
        'selection_history': selection_history
    }

def save_iterative_selection_summary(
    datasets: List[str],
    model_type: str = 'RandomForest',
    output_file: str = None
) -> None:
    """
    Create and save a comprehensive summary of iterative feature selection results.
    
    Args:
        datasets: List of dataset names to include in summary
        model_type: Model type used for the experiments
        output_file: Optional custom output file path
    """
    results_dir = Path("results") / "iterative_feature_selection"
    
    if output_file is None:
        output_file = results_dir / f"iterative_selection_summary_{model_type.lower()}.txt"
    
    summary_data = []
    detailed_results = {}
    
    # Collect results from individual files
    for dataset_name in datasets:
        result_filename = f"{dataset_name}_iterative_{model_type.lower()}.pkl"
        result_path = results_dir / result_filename
        
        if result_path.exists():
            try:
                with open(result_path, 'rb') as f:
                    results = pickle.load(f)
                
                # Extract selected features
                selected_features = [feature for feature in TOPOLOGICAL_FEATURES 
                                   if results.get(feature, False)]
                
                # Create summary entry
                summary_entry = {
                    'dataset': dataset_name,
                    'accuracy_mean': results.get('acc_mean', 0.0),
                    'accuracy_std': results.get('acc_std', 0.0),
                    'time_seconds': results.get('time', 0.0),
                    'num_features_selected': len(selected_features),
                    'selected_features': selected_features,
                    'baseline_accuracy': results.get('baseline_acc', 0.0),
                    'total_improvement': results.get('total_improvement', 0.0),
                    'iterations': results.get('iterations', 0)
                }
                
                summary_data.append(summary_entry)
                detailed_results[dataset_name] = results
                
            except Exception as e:
                print(f"Warning: Could not load results for {dataset_name}: {e}")
        else:
            print(f"Warning: No results file found for {dataset_name}")
    
    # Write comprehensive summary
    with open(output_file, 'w') as f:
        f.write("ITERATIVE FEATURE SELECTION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Number of Datasets: {len(summary_data)}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        if summary_data:
            avg_accuracy = np.mean([entry['accuracy_mean'] for entry in summary_data])
            avg_time = np.mean([entry['time_seconds'] for entry in summary_data])
            avg_features = np.mean([entry['num_features_selected'] for entry in summary_data])
            avg_improvement = np.mean([entry['total_improvement'] for entry in summary_data])
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Average Time: {avg_time:.2f} seconds\n")
            f.write(f"Average Features Selected: {avg_features:.1f}\n")
            f.write(f"Average Improvement: {avg_improvement:.4f}\n\n")
        
        # Dataset-by-dataset results
        f.write("DATASET RESULTS\n")
        f.write("-" * 20 + "\n")
        
        for entry in summary_data:
            f.write(f"\nDataset: {entry['dataset']}\n")
            f.write(f"  Accuracy: {entry['accuracy_mean']:.4f} ± {entry['accuracy_std']:.4f}\n")
            f.write(f"  Baseline: {entry['baseline_accuracy']:.4f}\n")
            f.write(f"  Improvement: {entry['total_improvement']:+.4f}\n")
            f.write(f"  Time: {entry['time_seconds']:.2f} seconds\n")
            f.write(f"  Features Selected ({entry['num_features_selected']}): {', '.join(entry['selected_features']) if entry['selected_features'] else 'None'}\n")
            f.write(f"  Iterations: {entry['iterations']}\n")
        
        # Feature frequency analysis
        f.write(f"\n\nFEATURE SELECTION FREQUENCY\n")
        f.write("-" * 30 + "\n")
        
        feature_counts = {}
        for feature in TOPOLOGICAL_FEATURES:
            count = sum(1 for entry in summary_data if feature in entry['selected_features'])
            if count > 0:
                feature_counts[feature] = count
        
        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        for feature, count in sorted_features:
            percentage = (count / len(summary_data)) * 100 if summary_data else 0
            f.write(f"  {feature}: {count}/{len(summary_data)} datasets ({percentage:.1f}%)\n")
        
        # Features never selected
        never_selected = [f for f in TOPOLOGICAL_FEATURES if f not in feature_counts]
        if never_selected:
            f.write(f"\n  Features never selected: {', '.join(never_selected)}\n")
    
    # Also save as CSV for easy analysis
    csv_file = output_file.with_suffix('.csv')
    import pandas as pd
    
    # Create DataFrame for easy analysis
    df_data = []
    for entry in summary_data:
        row = {
            'Dataset': entry['dataset'],
            'Accuracy_Mean': entry['accuracy_mean'],
            'Accuracy_Std': entry['accuracy_std'],
            'Baseline_Accuracy': entry['baseline_accuracy'],
            'Total_Improvement': entry['total_improvement'],
            'Time_Seconds': entry['time_seconds'],
            'Num_Features_Selected': entry['num_features_selected'],
            'Selected_Features': '; '.join(entry['selected_features']),
            'Iterations': entry['iterations']
        }
        # Add individual feature columns
        for feature in TOPOLOGICAL_FEATURES:
            row[f'Feature_{feature}'] = feature in entry['selected_features']
        
        df_data.append(row)
    
    if df_data:
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False)
        
        print(f"Summary saved to:")
        print(f"  Text: {output_file}")
        print(f"  CSV:  {csv_file}")
    else:
        print("No data to save in summary")

def run_iterative_feature_selection_experiment(
    datasets: List[str],
    model_type: str = 'RandomForest',
    min_improvement: float = 0.001,
    max_features: Optional[int] = None,
    verbose: bool = True
) -> None:
    """
    Run iterative feature selection on multiple datasets and save results.
    
    Args:
        datasets: List of dataset names
        model_type: Type of model to use
        min_improvement: Minimum improvement threshold
        max_features: Maximum number of features to select
        verbose: Whether to print progress
    """
    plots_dir = Path("plots") / "iterative_feature_selection"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path("results") / "iterative_feature_selection"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    completed_datasets = []
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Run iterative feature selection
            results = iterative_feature_selection(
                dataset_name=dataset_name,
                model_type=model_type,
                min_improvement=min_improvement,
                max_features=max_features,
                verbose=verbose,
                plots_dir=str(plots_dir)
            )
            
            # Save individual results
            result_filename = f"{dataset_name}_iterative_{model_type.lower()}.pkl"
            result_path = results_dir / result_filename
            
            with open(result_path, 'wb') as f:
                pickle.dump(results, f)
            
            completed_datasets.append(dataset_name)
            
            if verbose:
                print(f"Results saved to: {result_path}")
                print(f"Accuracy: {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
                print(f"Time: {results['time']:.2f}s")
                print(f"Features selected: {sum(1 for k, v in results.items() if k in TOPOLOGICAL_FEATURES and v)}")
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Create comprehensive summary after all datasets are processed
    if completed_datasets:
        print(f"\n{'='*60}")
        print("Creating comprehensive summary...")
        print(f"{'='*60}")
        
        save_iterative_selection_summary(
            datasets=completed_datasets,
            model_type=model_type
        )
        
        print(f"Processed {len(completed_datasets)} datasets successfully.")
    else:
        print("No datasets were processed successfully.")

def compare_selection_methods(dataset_name: str, verbose: bool = True) -> Dict:
    """
    Compare iterative feature selection with the importance-based method.
    
    Args:
        dataset_name: Name of the dataset
        verbose: Whether to print comparison
        
    Returns:
        Dictionary comparing both methods
    """
    # Load iterative selection results
    iterative_path = Path("results") / "iterative_feature_selection" / f"{dataset_name}_iterative_randomforest.pkl"
    importance_path = Path("results") / f"{dataset_name}_sum_rf.pkl"
    
    comparison = {'dataset': dataset_name}
    
    try:
        if iterative_path.exists():
            with open(iterative_path, 'rb') as f:
                iterative_results = pickle.load(f)
            comparison['iterative'] = iterative_results
        else:
            comparison['iterative'] = None
            
        if importance_path.exists():
            with open(importance_path, 'rb') as f:
                importance_results = pickle.load(f)
            comparison['importance_based'] = importance_results
        else:
            comparison['importance_based'] = None
            
        if verbose and comparison['iterative'] and comparison['importance_based']:
            print(f"\nComparison for {dataset_name}:")
            print(f"Iterative method:")
            print(f"  Accuracy: {comparison['iterative']['acc_mean']:.4f} ± {comparison['iterative']['acc_std']:.4f}")
            print(f"  Features: {comparison['iterative']['num_features']}")
            print(f"  Time: {comparison['iterative']['time']:.2f}s")
            
            print(f"Importance-based method:")
            print(f"  Accuracy: {comparison['importance_based']['acc_mean']:.4f} ± {comparison['importance_based']['acc_std']:.4f}")
            num_importance_features = sum(1 for k, v in comparison['importance_based'].items() 
                                        if k in TOPOLOGICAL_FEATURES and v)
            print(f"  Features: {num_importance_features}")
            print(f"  Time: {comparison['importance_based']['time']:.2f}s")
            
    except Exception as e:
        if verbose:
            print(f"Error comparing methods for {dataset_name}: {str(e)}")
        comparison['error'] = str(e)
    
    return comparison 