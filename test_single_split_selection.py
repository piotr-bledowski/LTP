#!/usr/bin/env python3
"""
Test script for the modified single-split feature selection method.
This script demonstrates the new functionality that runs on all splits sequentially.
"""

import os
import sys
import warnings
import subprocess

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

def test_single_split_feature_selection():
    """Test the new single-split feature selection functionality."""
    
    print("Testing single-split feature selection functionality...")
    print("=" * 60)
    
    # Test with a small dataset
    dataset = "ENZYMES"  # Usually has fewer samples, faster to test
    model_type = "RandomForest"
    
    print(f"Dataset: {dataset}")
    print(f"Model: {model_type}")
    print("Mode: Single-split sequential (runs on ALL splits)")
    print()
    
    # Run single-split mode (now runs on all splits automatically)
    print("Running single-split feature selection on all splits...")
    cmd = [
        sys.executable, "main.py",
        "--dataset_name", dataset,
        "--single_split_selection", "true",
        "--model_type", model_type,
        "--verbose", "false"  # Reduce verbosity for cleaner output
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # Increased timeout
        
        if result.returncode == 0:
            print("✓ Single-split feature selection completed successfully!")
            print("\nKey output lines:")
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in [
                    'dataset', 'splits', 'summary', 'accuracy', 'improvement', 
                    'results saved', 'processed', 'baseline'
                ]):
                    print(f"  {line}")
                    
            # Look for generated files
            expected_csv = f"results/single_split_results_{model_type.lower()}.csv"
            if os.path.exists(expected_csv):
                print(f"\n✓ CSV results file created: {expected_csv}")
                
                # Read and display a few lines from the CSV
                try:
                    import pandas as pd
                    df = pd.read_csv(expected_csv)
                    print("\nCSV contents preview:")
                    print(df[['Dataset', 'Accuracy_Mean', 'Accuracy_Std', 'Total_Improvement', 'Num_Splits_Processed']].head())
                except Exception as e:
                    print(f"  Could not read CSV: {e}")
            else:
                print(f"\n⚠ CSV results file not found: {expected_csv}")
                
        else:
            print("✗ Single-split feature selection failed!")
            print("Error output:")
            print(result.stderr)
            print("Standard output:")
            print(result.stdout[-1000:])  # Last 1000 chars
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out after 10 minutes")
    except Exception as e:
        print(f"✗ Error running test: {e}")
    
    print("\n" + "=" * 60)

def compare_methods():
    """Show how to compare the two approaches."""
    
    print("\nHow to use the updated single-split approach:")
    print("=" * 60)
    
    print("1. Single-split sequential mode (NEW):")
    print("   python main.py --dataset_name ENZYMES --single_split_selection true")
    print("   → Runs feature selection on ALL splits sequentially")
    print("   → Outputs aggregated statistics in CSV format")
    print("   → No need to specify split_idx")
    print()
    
    print("2. Multi-split averaging mode (ORIGINAL):")
    print("   python main.py --dataset_name ENZYMES --single_split_selection false")
    print("   → Averages importance across all splits, evaluates on all splits")
    print("   → Requires pre-computed importance files")
    print()
    
    print("3. Iterative feature selection (ALTERNATIVE):")
    print("   python main.py --dataset_name ENZYMES --iterative_feature_selection true")
    print("   → Forward selection without pre-computed importance")
    print()
    
    print("Key benefits of the new approach:")
    print("- Evaluates algorithm robustness across different data splits")
    print("- Provides mean ± std accuracy across all runs")
    print("- Shows feature selection consistency")
    print("- Self-contained (no pre-computed files needed)")
    print("- Generates comprehensive CSV with all metrics")

def show_expected_csv_format():
    """Show what the output CSV contains."""
    
    print("\nExpected CSV output format:")
    print("=" * 60)
    
    columns = [
        "Dataset", "Accuracy_Mean", "Accuracy_Std", "Baseline_Accuracy", 
        "Total_Improvement", "Time_Mean_Seconds", "Time_Std_Seconds",
        "Total_Time_Seconds", "Num_Splits_Processed", "Selected_Features_Frequencies",
        "Model_Type", "Feature_degree_sum", "Feature_shortest_paths", "..."
    ]
    
    print("Main columns:")
    for col in columns[:10]:
        print(f"  - {col}")
    print("  - Individual feature frequency columns (Feature_*)")
    print()
    
    print("The CSV provides:")
    print("  - Mean and std accuracy across all split runs")
    print("  - Comparison with baseline (LDP-only) performance")
    print("  - Execution time statistics")
    print("  - Feature selection frequencies (how often each feature was selected)")
    print("  - Number of splits successfully processed")

if __name__ == "__main__":
    test_single_split_feature_selection()
    compare_methods()
    show_expected_csv_format() 