import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import os
import pickle
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime

def load_data(data_file: str, test_file: str = None, train_pct: float = 0.7, test_pct: float = 0.3, used_splits: set = None):
    """Load and split data according to plan"""
    if test_file:
        # Pre-split data
        train_df = pd.read_csv(data_file)
        test_df = pd.read_csv(test_file)
        class_col = [col for col in train_df.columns if col.lower() == 'class'][0]
        
        X_train = train_df.drop(columns=[class_col])
        y_train = train_df[class_col].astype(str)
        X_test = test_df.drop(columns=[class_col])
        y_test = test_df[class_col].astype(str)
    else:
        # Single dataset - split it
        data_df = pd.read_csv(data_file)
        class_col = [col for col in data_df.columns if col.lower() == 'class'][0]
        
        X = data_df.drop(columns=[class_col])
        y = data_df[class_col].astype(str)
        
        # Round to ensure train + test = total
        n_samples = len(X)
        train_size = int(round(train_pct * n_samples))
        test_size = n_samples - train_size
        
        # Keep trying different random seeds until we get a unique split
        max_attempts = 1000
        for attempt in range(max_attempts):
            random_seed = np.random.randint(0, 2**31)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, test_size=test_size, random_state=random_seed, stratify=y
            )
            
            # Create a hash of the split to check for uniqueness
            if used_splits is not None:
                # Create a simple hash based on the indices of train/test samples
                train_indices = set(X_train.index)
                test_indices = set(X_test.index)
                split_hash = hash(frozenset(train_indices))
                
                if split_hash not in used_splits:
                    used_splits.add(split_hash)
                    break
            else:
                break
        else:
            print(f"Warning: Could not find unique split after {max_attempts} attempts")
    
    # Normalize
    min_vals = X_train.min()
    max_vals = X_train.max()
    X_train = (X_train - min_vals) / (max_vals - min_vals)
    X_test = (X_test - min_vals) / (max_vals - min_vals)
    
    return X_train, y_train, X_test, y_test

def get_classifier(classifier_type: str, k: int = 3, metric: str = 'euclidean'):
    """Get classifier instance"""
    if classifier_type == 'dt':
        return DecisionTreeClassifier(random_state=np.random.randint(0, 2**31))
    elif classifier_type == 'knn':
        return KNeighborsClassifier(n_neighbors=k, metric=metric)
    elif classifier_type == 'svm':
        return SVC(kernel='rbf', random_state=np.random.randint(0, 2**31))
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

def get_class_distribution(y_series):
    """Get class distribution as a dictionary"""
    return y_series.value_counts().to_dict()

def format_class_distribution(class_dist):
    """Format class distribution for display with class names"""
    if not class_dist:
        return "N/A"
    return ", ".join([f"{cls}({count})" for cls, count in sorted(class_dist.items())])

def find_min_max_experiments(experiments, metric_key):
    """Find experiments with min and max values for a given metric"""
    if not experiments:
        return None, None, None, None
    
    min_val = min(exp[metric_key] for exp in experiments)
    max_val = max(exp[metric_key] for exp in experiments)
    
    min_exp = next(exp for exp in experiments if exp[metric_key] == min_val)
    max_exp = next(exp for exp in experiments if exp[metric_key] == max_val)
    
    return min_val, max_val, min_exp, max_exp

def find_representative_experiments(experiments, metric_key='cases_needed'):
    """Find most representative experiments: closest to average, best (min), and worst (max)"""
    if not experiments:
        return None, None, None
    
    # Calculate average
    avg_val = np.mean([exp[metric_key] for exp in experiments])
    
    # Find experiment closest to average
    avg_exp = min(experiments, key=lambda exp: abs(exp[metric_key] - avg_val))
    
    # Find best (minimum) and worst (maximum) experiments
    best_exp = min(experiments, key=lambda exp: exp[metric_key])
    worst_exp = max(experiments, key=lambda exp: exp[metric_key])
    
    return avg_exp, best_exp, worst_exp

def generate_summary_report(all_converged_experiments, split_info_list, args):
    """Generate the complete summary report as a string"""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("OVERALL SUMMARY ACROSS ALL SPLITS")
    report_lines.append("="*80)
    
    if all_converged_experiments:
        report_lines.append(f"Total experiments that converged: {len(all_converged_experiments)}/{args.splits * args.iterations}")
        
        # Calculate overall statistics
        all_iterations = [conv['iteration'] for conv in all_converged_experiments]
        all_cases = [conv['cases_needed'] for conv in all_converged_experiments]
        
        report_lines.append(f"\nOverall Statistics:")
        report_lines.append(f"Average iterations needed: {np.mean(all_iterations):.1f}")
        report_lines.append(f"Average cases needed: {np.mean(all_cases):.1f}")
        
        # Min/Max cases with class distribution and accuracy
        min_cases, max_cases, min_exp, max_exp = find_min_max_experiments(all_converged_experiments, 'cases_needed')
        report_lines.append(f"Min cases needed: {min_cases} (class dist: {format_class_distribution(min_exp['class_distribution'])}, accuracy: {min_exp['accuracy']:.3f})")
        report_lines.append(f"Max cases needed: {max_cases} (class dist: {format_class_distribution(max_exp['class_distribution'])}, accuracy: {max_exp['accuracy']:.3f})")
        report_lines.append(f"Standard deviation of cases needed: {np.std(all_cases):.1f}")
        
        # Representative experiments
        avg_exp, best_exp, worst_exp = find_representative_experiments(all_converged_experiments, 'cases_needed')
        if avg_exp and best_exp and worst_exp:
            report_lines.append(f"\nRepresentative Experiments (based on cases needed):")
            report_lines.append(f"Most representative of average: Split {avg_exp['split']}, Exp {avg_exp['experiment']} "
                              f"({avg_exp['cases_needed']} cases, class dist: {format_class_distribution(avg_exp['class_distribution'])}, accuracy: {avg_exp['accuracy']:.3f})")
            report_lines.append(f"Most representative of best case: Split {best_exp['split']}, Exp {best_exp['experiment']} "
                              f"({best_exp['cases_needed']} cases, class dist: {format_class_distribution(best_exp['class_distribution'])}, accuracy: {best_exp['accuracy']:.3f})")
            report_lines.append(f"Most representative of worst case: Split {worst_exp['split']}, Exp {worst_exp['experiment']} "
                              f"({worst_exp['cases_needed']} cases, class dist: {format_class_distribution(worst_exp['class_distribution'])}, accuracy: {worst_exp['accuracy']:.3f})")
        
        # SVM-specific overall statistics
        if args.classifier == 'svm':
            all_support_vectors = [conv['support_vectors'] for conv in all_converged_experiments]
            report_lines.append(f"\nOverall SVM Support Vector Statistics:")
            report_lines.append(f"Average support vectors needed: {np.mean(all_support_vectors):.1f}")
            
            # Min/Max support vectors with class distribution and accuracy
            min_sv, max_sv, min_sv_exp, max_sv_exp = find_min_max_experiments(all_converged_experiments, 'support_vectors')
            report_lines.append(f"Min support vectors needed: {min_sv} (class dist: {format_class_distribution(min_sv_exp['class_distribution'])}, accuracy: {min_sv_exp['accuracy']:.3f})")
            report_lines.append(f"Max support vectors needed: {max_sv} (class dist: {format_class_distribution(max_sv_exp['class_distribution'])}, accuracy: {max_sv_exp['accuracy']:.3f})")
            report_lines.append(f"Standard deviation of support vectors: {np.std(all_support_vectors):.1f}")
            
            # Representative experiments for support vectors
            avg_sv_exp, best_sv_exp, worst_sv_exp = find_representative_experiments(all_converged_experiments, 'support_vectors')
            if avg_sv_exp and best_sv_exp and worst_sv_exp:
                report_lines.append(f"\nRepresentative Experiments (based on support vectors):")
                report_lines.append(f"Most representative of average: Split {avg_sv_exp['split']}, Exp {avg_sv_exp['experiment']} "
                                  f"({avg_sv_exp['support_vectors']} support vectors, class dist: {format_class_distribution(avg_sv_exp['class_distribution'])}, accuracy: {avg_sv_exp['accuracy']:.3f})")
                report_lines.append(f"Most representative of best case: Split {best_sv_exp['split']}, Exp {best_sv_exp['experiment']} "
                                  f"({best_sv_exp['support_vectors']} support vectors, class dist: {format_class_distribution(best_sv_exp['class_distribution'])}, accuracy: {best_sv_exp['accuracy']:.3f})")
                report_lines.append(f"Most representative of worst case: Split {worst_sv_exp['split']}, Exp {worst_sv_exp['experiment']} "
                                  f"({worst_sv_exp['support_vectors']} support vectors, class dist: {format_class_distribution(worst_sv_exp['class_distribution'])}, accuracy: {worst_sv_exp['accuracy']:.3f})")
        
        # Per-split summary
        report_lines.append(f"\nPer-Split Summary:")
        for split_num in range(1, args.splits + 1):
            split_experiments = [conv for conv in all_converged_experiments if conv['split'] == split_num]
            split_info = next((info for info in split_info_list if info['split_num'] == split_num), None)
            
            if split_experiments:
                split_cases = [conv['cases_needed'] for conv in split_experiments]
                if args.classifier == 'svm':
                    split_support_vectors = [conv['support_vectors'] for conv in split_experiments]
                    min_cases, max_cases, min_exp, max_exp = find_min_max_experiments(split_experiments, 'cases_needed')
                    min_sv, max_sv, min_sv_exp, max_sv_exp = find_min_max_experiments(split_experiments, 'support_vectors')
                    report_lines.append(f"Split {split_num}: {len(split_experiments)}/{args.iterations} converged, "
                              f"avg cases: {np.mean(split_cases):.1f}, "
                              f"min: {min_cases} ({format_class_distribution(min_exp['class_distribution'])}, acc: {min_exp['accuracy']:.3f}), "
                              f"max: {max_cases} ({format_class_distribution(max_exp['class_distribution'])}, acc: {max_exp['accuracy']:.3f}), "
                              f"avg SV: {np.mean(split_support_vectors):.1f}, "
                              f"min SV: {min_sv} ({format_class_distribution(min_sv_exp['class_distribution'])}, acc: {min_sv_exp['accuracy']:.3f}), "
                              f"max SV: {max_sv} ({format_class_distribution(max_sv_exp['class_distribution'])}, acc: {max_sv_exp['accuracy']:.3f})")
                    if split_info:
                        report_lines.append(f"  Train dist: {format_class_distribution(split_info['train_class_dist'])}, "
                                  f"Test dist: {format_class_distribution(split_info['test_class_dist'])}")
                else:
                    min_cases, max_cases, min_exp, max_exp = find_min_max_experiments(split_experiments, 'cases_needed')
                    report_lines.append(f"Split {split_num}: {len(split_experiments)}/{args.iterations} converged, "
                              f"avg cases: {np.mean(split_cases):.1f}, "
                              f"min: {min_cases} ({format_class_distribution(min_exp['class_distribution'])}, acc: {min_exp['accuracy']:.3f}), "
                              f"max: {max_cases} ({format_class_distribution(max_exp['class_distribution'])}, acc: {max_exp['accuracy']:.3f})")
                    if split_info:
                        report_lines.append(f"  Train dist: {format_class_distribution(split_info['train_class_dist'])}, "
                                  f"Test dist: {format_class_distribution(split_info['test_class_dist'])}")
            else:
                report_lines.append(f"Split {split_num}: 0/{args.iterations} converged")
                if split_info:
                    report_lines.append(f"  Train dist: {format_class_distribution(split_info['train_class_dist'])}, "
                              f"Test dist: {format_class_distribution(split_info['test_class_dist'])}")
    else:
        report_lines.append("No experiments reached the threshold across all splits!")
    
    report_lines.append("="*80)
    
    return "\n".join(report_lines)

def run_single_experiment(args_tuple):
    """Run a single experiment - designed for parallel execution"""
    (exp_num, X_train, y_train, X_test, y_test, classifier_type, threshold, m, action, classifier_kwargs, split_dir) = args_tuple
    
    # Set different random seed for each experiment
    np.random.seed(np.random.randint(0, 2**31))
    
    # Initialize used training subset
    used_indices = set()
    used_X = pd.DataFrame(columns=X_train.columns)
    used_y = pd.Series(dtype=str)
    
    exp_results = []
    iteration = 0
    
    while len(used_indices) < len(X_train):
        # Save current state
        current_accuracy = None
        current_metrics = {}
        
        # Select m random cases from available training data
        available = set(range(len(X_train))) - used_indices
        if len(available) < m:
            selected = list(available)
        else:
            available_indices = list(available)
            # Random sampling from available cases (preserves natural class distribution)
            selected = np.random.choice(available_indices, m, replace=False).tolist()
        
        # Apply action
        if action == 'additive':
            used_indices.update(selected)
        else:  # subtractive
            used_indices.difference_update(selected)
        
        # Update used data
        used_X = X_train.iloc[list(used_indices)]
        used_y = y_train.iloc[list(used_indices)]
        
        # Skip if not enough classes
        if len(used_y.unique()) < 2:
            iteration += 1
            continue
        
        # Train and evaluate
        clf = get_classifier(classifier_type, **classifier_kwargs)
        clf.fit(used_X, used_y)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Extract metrics
        if classifier_type == 'dt':
            current_metrics = {'depth': clf.get_depth(), 'leaves': clf.get_n_leaves()}
        elif classifier_type == 'knn':
            current_metrics = {'k': clf.n_neighbors, 'metric': clf.metric}
        elif classifier_type == 'svm':
            current_metrics = {'support_vectors': len(clf.support_vectors_)}
        
        exp_results.append({
            'iteration': iteration,
            'used_size': len(used_indices),
            'accuracy': accuracy,
            'metrics': current_metrics,
            'class_distribution': get_class_distribution(used_y)
        })
        
        # Check threshold
        if accuracy >= threshold:
            # Save converged data
            if classifier_type == 'dt':
                with open(f'{split_dir}/dt_tree_exp_{exp_num}.pkl', 'wb') as f:
                    pickle.dump(clf, f)
            elif classifier_type == 'svm':
                # Save support vectors
                sv_indices = clf.support_
                sv_data = used_X.iloc[sv_indices].copy()
                sv_data['class'] = used_y.iloc[sv_indices].values
                sv_data.to_csv(f'{split_dir}/sv_exp_{exp_num}.csv', index=False)
                
                # Save full converged training set
                conv_data = used_X.copy()
                conv_data['class'] = used_y.values
                conv_data.to_csv(f'{split_dir}/converged_exp_{exp_num}.csv', index=False)
                
                # Store support vector indices in metrics for plotting
                current_metrics['support_vector_indices'] = sv_indices.tolist()
            
            break
        
        iteration += 1
    
    return exp_results

def run_iterative_testing(X_train, y_train, X_test, y_test, classifier_type, threshold, m, iterations, action, split_dir, **classifier_kwargs):
    """Main testing loop using parallel execution"""
    print(f"Running {iterations} experiments in parallel...")
    
    # Prepare arguments for parallel execution
    args_list = []
    for exp in range(iterations):
        args_tuple = (exp + 1, X_train, y_train, X_test, y_test, classifier_type, threshold, m, action, classifier_kwargs, split_dir)
        args_list.append(args_tuple)
    
    # Run experiments in parallel
    max_workers = min(multiprocessing.cpu_count(), iterations)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_experiment, args_list))
    
    print(f"Completed {iterations} experiments")
    return results

def plot_svm_cases_vs_support_vectors(all_results, exp_dir):
    """Create parallel coordinates plot showing cases and support vectors per experiment"""
    if not all_results:
        return
    
    # Extract data for plotting
    experiments = []
    cases = []
    support_vectors = []
    
    for exp_idx, exp_results in enumerate(all_results):
        for iter_result in exp_results:
            if iter_result['accuracy'] >= 0.95:  # threshold reached
                experiments.append(exp_idx + 1)
                cases.append(iter_result['used_size'])
                support_vectors.append(iter_result['metrics'].get('support_vectors', 0))
                break
    
    if not experiments:
        return
    
    # Create the parallel coordinates plot - width scales with number of experiments
    width = max(20, len(experiments) * 0.3)  # At least 20 inches, or 0.3 inches per experiment
    fig, ax = plt.subplots(figsize=(width, 8))
    
    # Create the parallel coordinates
    x_pos = np.arange(len(experiments))
    
    # Plot support vectors polyline first (behind)
    ax.plot(x_pos, support_vectors, 's-', linewidth=2, markersize=8, label='Number of Support Vectors', color='red')
    
    # Plot cases polyline second (in front)
    ax.plot(x_pos, cases, 'o-', linewidth=2, markersize=8, label='Number of Cases', color='blue')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{exp}' for exp in experiments])
    
    # Labels and title
    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Count')
    ax.set_title('SVM: Cases vs Support Vectors per Experiment (Parallel Coordinates)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = f'{exp_dir}/svm_cases_vs_support_vectors.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Parallel coordinates plot saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Iterative Sureness Testing')
    parser.add_argument('--data', required=True, help='Training data file')
    parser.add_argument('--test-data', help='Test data file (optional)')
    parser.add_argument('--train-pct', type=float, default=0.7, help='Training percentage')
    parser.add_argument('--test-pct', type=float, default=0.3, help='Test percentage')
    parser.add_argument('--classifier', required=True, choices=['dt', 'knn', 'svm'], help='Classifier type')
    parser.add_argument('--k', type=int, default=3, help='k for KNN')
    parser.add_argument('--metric', default='euclidean', help='Distance metric for KNN')
    parser.add_argument('--threshold', type=float, default=0.95, help='Accuracy threshold')
    parser.add_argument('--m', type=int, default=5, help='Cases per iteration')
    parser.add_argument('--iterations', type=int, default=10, help='Number of experiments per split')
    parser.add_argument('--splits', type=int, default=1, help='Number of splits to test')
    parser.add_argument('--action', choices=['additive', 'subtractive'], default='additive', help='Action type')
    parser.add_argument('--plot', action='store_true', help='Create plots (default: False)')
    
    args = parser.parse_args()
    
    # Validate percentages
    if abs(args.train_pct + args.test_pct - 1.0) > 1e-6:
        parser.error("train_pct + test_pct must equal 1.0")
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f'results/exp_{timestamp}'
    os.makedirs(exp_dir, exist_ok=True)
    
    # Prepare classifier kwargs
    classifier_kwargs = {}
    if args.classifier == 'knn':
        classifier_kwargs = {'k': args.k, 'metric': args.metric}
    
    # Store all results across splits
    all_splits_results = []
    all_converged_experiments = []
    split_info_list = []  # Store split class distribution info
    used_splits = set()  # Track used splits to avoid duplicates
    
    print(f"Running {args.splits} split(s) with {args.iterations} experiments each...")
    print(f"Results will be saved to: {exp_dir}")
    print("="*80)
    
    for split_num in range(1, args.splits + 1):
        print(f"\nSPLIT {split_num}/{args.splits}")
        print("-" * 40)
        
        # Load data for this split (ensure unique split)
        print(f"Loading data for split {split_num} (ensuring unique split)...")
        X_train, y_train, X_test, y_test = load_data(
            args.data, args.test_data, args.train_pct, args.test_pct, used_splits
        )
        
        # Create subfolder for this split
        split_dir = f'{exp_dir}/split_{split_num}'
        os.makedirs(split_dir, exist_ok=True)
        
        # Save train and test subsets to CSV for this split
        train_df = X_train.copy()
        train_df['class'] = y_train
        train_df.to_csv(f'{split_dir}/train_subset.csv', index=False)
        
        test_df = X_test.copy()
        test_df['class'] = y_test
        test_df.to_csv(f'{split_dir}/test_subset.csv', index=False)
        
        # Calculate and display class distributions
        train_class_dist = get_class_distribution(y_train)
        test_class_dist = get_class_distribution(y_test)
        
        print(f"Training subset saved to: {split_dir}/train_subset.csv ({len(X_train)} samples)")
        print(f"  Training class distribution: {format_class_distribution(train_class_dist)}")
        print(f"Test subset saved to: {split_dir}/test_subset.csv ({len(X_test)} samples)")
        print(f"  Test class distribution: {format_class_distribution(test_class_dist)}")
        
        # Store split class distributions for final summary
        split_info = {
            'split_num': split_num,
            'train_class_dist': train_class_dist,
            'test_class_dist': test_class_dist
        }
        split_info_list.append(split_info)
        
        # Run testing for this split
        split_results = run_iterative_testing(
            X_train, y_train, X_test, y_test,
            args.classifier, args.threshold, args.m, args.iterations, args.action, split_dir,
            **classifier_kwargs
        )
        
        all_splits_results.append(split_results)
        
        # Process results for this split
        split_converged_experiments = []
        for exp_idx, exp_results in enumerate(split_results):
            for iter_result in exp_results:
                if iter_result['accuracy'] >= args.threshold:
                    # Get additional metrics for SVM
                    sv_count = None
                    if args.classifier == 'svm':
                        sv_count = iter_result['metrics'].get('support_vectors', 'N/A')
                    
                    # Get class distribution for this converged experiment
                    class_dist = iter_result.get('class_distribution', {})
                    experiment_data = {
                        'split': split_num,
                        'experiment': exp_idx + 1,
                        'iteration': iter_result['iteration'],
                        'cases_needed': iter_result['used_size'],
                        'accuracy': iter_result['accuracy'],
                        'support_vectors': sv_count,
                        'class_distribution': class_dist
                    }
                    split_converged_experiments.append(experiment_data)
                    all_converged_experiments.append(experiment_data)
                    break
        
        # Print split-specific results
        if split_converged_experiments:
            print(f"Split {split_num} - Experiments that converged: {len(split_converged_experiments)}/{args.iterations}")
            
            # Calculate split statistics
            split_iterations = [conv['iteration'] for conv in split_converged_experiments]
            split_cases = [conv['cases_needed'] for conv in split_converged_experiments]
            
            print(f"Split {split_num} Statistics:")
            print(f"  Average iterations needed: {np.mean(split_iterations):.1f}")
            print(f"  Average cases needed: {np.mean(split_cases):.1f}")
            
            # Min/Max cases with class distribution and accuracy
            min_cases, max_cases, min_exp, max_exp = find_min_max_experiments(split_converged_experiments, 'cases_needed')
            print(f"  Min cases needed: {min_cases} (class dist: {format_class_distribution(min_exp['class_distribution'])}, accuracy: {min_exp['accuracy']:.3f})")
            print(f"  Max cases needed: {max_cases} (class dist: {format_class_distribution(max_exp['class_distribution'])}, accuracy: {max_exp['accuracy']:.3f})")
            print(f"  Standard deviation of cases needed: {np.std(split_cases):.1f}")
            
            # SVM-specific statistics for this split
            if args.classifier == 'svm':
                split_support_vectors = [conv['support_vectors'] for conv in split_converged_experiments]
                print(f"  Average support vectors needed: {np.mean(split_support_vectors):.1f}")
                
                # Min/Max support vectors with class distribution and accuracy
                min_sv, max_sv, min_sv_exp, max_sv_exp = find_min_max_experiments(split_converged_experiments, 'support_vectors')
                print(f"  Min support vectors needed: {min_sv} (class dist: {format_class_distribution(min_sv_exp['class_distribution'])}, accuracy: {min_sv_exp['accuracy']:.3f})")
                print(f"  Max support vectors needed: {max_sv} (class dist: {format_class_distribution(max_sv_exp['class_distribution'])}, accuracy: {max_sv_exp['accuracy']:.3f})")
                print(f"  Standard deviation of support vectors: {np.std(split_support_vectors):.1f}")
        else:
            print(f"Split {split_num} - No experiments reached the threshold!")
    
    # Create cases vs support vectors plot for SVM (using all splits data)
    if args.plot and args.classifier == 'svm':
        print("\nCreating cases vs support vectors plot for SVM...")
        # Flatten all results for plotting
        all_results_flat = []
        for split_results in all_splits_results:
            all_results_flat.extend(split_results)
        plot_svm_cases_vs_support_vectors(all_results_flat, exp_dir)
    
    # Generate and save summary report to file
    summary_report = generate_summary_report(all_converged_experiments, split_info_list, args)
    
    # Save summary to text file
    summary_file = f'{exp_dir}/summary_report.txt'
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    print(f"\nSummary report saved to: {summary_file}")
    print("="*80)

if __name__ == "__main__":
    main()
