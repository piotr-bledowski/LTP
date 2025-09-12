#!/usr/bin/env python3
"""
Main script for running iterative feature selection experiments.
This implements forward feature selection without using precalculated feature importances.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List

from iterative_feature_selection import (
    run_iterative_feature_selection_experiment,
    compare_selection_methods,
    save_iterative_selection_summary,
    TOPOLOGICAL_FEATURES
)

# Suppress warnings for cleaner output
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative Feature Selection for Local Topological Profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
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
            "AIDS",
            "BZR",
            "COIL-DEL",
            "COIL-RAG",
            "COX2",  # poprawiona literówka zamiast COS2
            "DHFR",
            "FRANKENSTEIN",
            "Letter-high",
            "Letter-med",
            "Letter-low",
            "MCF-7",
            "Mutagenicity",
        ],
        default=["DD"],
        help="Dataset names to process, use 'all' to run on all datasets."
    )

    parser.add_argument(
        "--model_type",
        choices=["LinearSVM", "KernelSVM", "RandomForest"],
        default="RandomForest",
        help="Classification algorithm to use."
    )

    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.0001,
        help="Minimum accuracy improvement required to add a feature."
    )

    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Maximum number of topological features to select (None for no limit)."
    )

    parser.add_argument(
        "--mode",
        choices=["select", "compare", "both"],
        default="select",
        help="Mode: 'select' runs iterative selection, 'compare' compares methods, 'both' does both."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress information."
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity."
    )

    return parser.parse_args()


def get_dataset_list(datasets_arg: List[str]) -> List[str]:
    """Convert datasets argument to actual list of dataset names."""
    if "all" in datasets_arg:
        return [
            "DD", "NCI1", "PROTEINS_full", "ENZYMES",
            "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY",
            "REDDIT-MULTI-5K", "COLLAB", "AIDS",
            "BZR",
            "COIL-DEL",
            "COIL-RAG",
            "COX2",  # poprawiona literówka zamiast COS2
            "DHFR",
            "FRANKENSTEIN",
            "Letter-high",
            "Letter-med",
            "Letter-low",
            "MCF-7",
            "Mutagenicity",
        ]
    return datasets_arg


def main():
    args = parse_args()

    # Handle verbosity
    verbose = args.verbose and not args.quiet

    # Get actual dataset list
    datasets = get_dataset_list(args.datasets)

    if verbose:
        print("Iterative Feature Selection for Local Topological Profile")
        print("=" * 60)
        print(f"Datasets: {datasets}")
        print(f"Model: {args.model_type}")
        print(f"Minimum improvement: {args.min_improvement}")
        print(f"Maximum features: {args.max_features if args.max_features else 'No limit'}")
        print(f"Mode: {args.mode}")
        print(f"Available topological features ({len(TOPOLOGICAL_FEATURES)}):")
        for i, feature in enumerate(TOPOLOGICAL_FEATURES, 1):
            print(f"  {i:2d}. {feature}")
        print("=" * 60)

    # Run iterative feature selection
    if args.mode in ["select", "both"]:
        if verbose:
            print("\nRunning iterative feature selection...")

        run_iterative_feature_selection_experiment(
            datasets=datasets,
            model_type=args.model_type,
            min_improvement=args.min_improvement,
            max_features=args.max_features,
            verbose=verbose
        )

        if verbose:
            print("\nIterative feature selection completed!")

    # Compare methods
    if args.mode in ["compare", "both"]:
        if verbose:
            print("\nComparing selection methods...")

        for dataset_name in datasets:
            try:
                comparison = compare_selection_methods(dataset_name, verbose=verbose)

                # Save comparison results
                comparison_dir = Path("results") / "comparisons"
                comparison_dir.mkdir(parents=True, exist_ok=True)

                import pickle
                comparison_path = comparison_dir / f"{dataset_name}_comparison.pkl"
                with open(comparison_path, 'wb') as f:
                    pickle.dump(comparison, f)

                if verbose:
                    print(f"Comparison saved to: {comparison_path}")

            except Exception as e:
                print(f"Error comparing methods for {dataset_name}: {str(e)}")

    # Generate additional summary if we ran iterative selection
    if args.mode in ["select", "both"] and verbose:
        print("\nGenerating additional summary files...")
        try:
            save_iterative_selection_summary(
                datasets=datasets,
                model_type=args.model_type
            )
        except Exception as e:
            print(f"Warning: Could not generate summary: {e}")

    if verbose:
        print("\nAll experiments completed!")
        print("\nResults locations:")
        print("  Individual results: results/iterative_feature_selection/")
        print("  Summary files: results/iterative_feature_selection/iterative_selection_summary_*.txt/.csv")
        print("  Comparisons: results/comparisons/")
        print("  Plots: plots/iterative_feature_selection/")


if __name__ == "__main__":
    main()
