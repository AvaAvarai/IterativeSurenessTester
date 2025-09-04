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

def load_data(data_file: str, test_file: str = None, train_pct: float = 0.7, test_pct: float = 0.3):
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
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=42, stratify=y
        )
    
    # Normalize
    min_vals = X_train.min()
    max_vals = X_train.max()
    X_train = (X_train - min_vals) / (max_vals - min_vals)
    X_test = (X_test - min_vals) / (max_vals - min_vals)
    
    return X_train, y_train, X_test, y_test

def get_classifier(classifier_type: str, k: int = 3, metric: str = 'euclidean'):
    """Get classifier instance"""
    if classifier_type == 'dt':
        return DecisionTreeClassifier(random_state=42)
    elif classifier_type == 'knn':
        return KNeighborsClassifier(n_neighbors=k, metric=metric)
    elif classifier_type == 'svm':
        return SVC(kernel='rbf', random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

def run_iterative_testing(X_train, y_train, X_test, y_test, classifier_type, threshold, m, iterations, action, **classifier_kwargs):
    """Main testing loop following the plan exactly"""
    results = []
    
    for exp in range(iterations):
        print(f"Experiment {exp + 1}/{iterations}")
        
        # Set different random seed for each experiment to ensure different case selection
        np.random.seed(42 + exp)
        
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
            
            # Select m random cases from train using stratified sampling
            available = set(range(len(X_train))) - used_indices
            if len(available) < m:
                selected = list(available)
            else:
                # Get available indices and their corresponding labels
                available_indices = list(available)
                available_labels = y_train.iloc[available_indices]
                
                # Stratified sampling: ensure we maintain class balance
                selected = []
                unique_classes = available_labels.unique()
                
                # Calculate how many samples to take from each class
                samples_per_class = m // len(unique_classes)
                remaining_samples = m % len(unique_classes)
                
                for i, cls in enumerate(unique_classes):
                    class_indices = [idx for idx, label in zip(available_indices, available_labels) if label == cls]
                    if len(class_indices) > 0:
                        # Take samples_per_class from this class, plus 1 extra if we have remaining samples
                        n_samples = min(samples_per_class + (1 if i < remaining_samples else 0), len(class_indices))
                        if n_samples > 0:
                            class_selected = np.random.choice(class_indices, n_samples, replace=False)
                            selected.extend(class_selected)
                
                # If we didn't get enough samples due to class imbalance, fill with random remaining
                if len(selected) < m:
                    remaining_available = [idx for idx in available_indices if idx not in selected]
                    if len(remaining_available) > 0:
                        additional_needed = m - len(selected)
                        additional_selected = np.random.choice(remaining_available, 
                                                           min(additional_needed, len(remaining_available)), 
                                                           replace=False)
                        selected.extend(additional_selected)
            
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
                'metrics': current_metrics
            })
            
            print(f"  Iter {iteration}: {len(used_indices)} cases, Acc: {accuracy:.4f}")
            
            # Check threshold
            if accuracy >= threshold:
                print(f"  THRESHOLD REACHED at iteration {iteration}")
                
                # Save converged data
                if classifier_type == 'dt':
                    with open(f'results/dt_tree_exp_{exp+1}.pkl', 'wb') as f:
                        pickle.dump(clf, f)
                elif classifier_type == 'svm':
                    # Save support vectors
                    sv_indices = clf.support_
                    sv_data = used_X.iloc[sv_indices].copy()
                    sv_data['class'] = used_y.iloc[sv_indices].values
                    sv_data.to_csv(f'results/sv_exp_{exp+1}.csv', index=False)
                    
                    # Save full converged training set
                    conv_data = used_X.copy()
                    conv_data['class'] = used_y.values
                    conv_data.to_csv(f'results/converged_exp_{exp+1}.csv', index=False)
                
                break
            
            iteration += 1
        
        results.append(exp_results)
    
    return results

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
    parser.add_argument('--iterations', type=int, default=10, help='Number of experiments')
    parser.add_argument('--action', choices=['additive', 'subtractive'], default='additive', help='Action type')
    
    args = parser.parse_args()
    
    # Validate percentages
    if abs(args.train_pct + args.test_pct - 1.0) > 1e-6:
        parser.error("train_pct + test_pct must equal 1.0")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(
        args.data, args.test_data, args.train_pct, args.test_pct
    )
    
    # Save train and test subsets to CSV
    train_df = X_train.copy()
    train_df['class'] = y_train
    train_df.to_csv('results/train_subset.csv', index=False)
    
    test_df = X_test.copy()
    test_df['class'] = y_test
    test_df.to_csv('results/test_subset.csv', index=False)
    
    print(f"Training subset saved to: results/train_subset.csv ({len(X_train)} samples)")
    print(f"Test subset saved to: results/test_subset.csv ({len(X_test)} samples)")
    
    # Prepare classifier kwargs
    classifier_kwargs = {}
    if args.classifier == 'knn':
        classifier_kwargs = {'k': args.k, 'metric': args.metric}
    
    # Run testing
    results = run_iterative_testing(
        X_train, y_train, X_test, y_test,
        args.classifier, args.threshold, args.m, args.iterations, args.action,
        **classifier_kwargs
    )
    
    # Save results
    import json
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to results/")
    
    # Print summary of experiments that reached threshold
    print("\n" + "="*60)
    print("EXPERIMENTS THAT REACHED THRESHOLD")
    print("="*60)
    
    converged_experiments = []
    for exp_idx, exp_results in enumerate(results):
        for iter_result in exp_results:
            if iter_result['accuracy'] >= args.threshold:
                # Get additional metrics for SVM
                sv_count = None
                if args.classifier == 'svm':
                    sv_count = iter_result['metrics'].get('support_vectors', 'N/A')
                
                converged_experiments.append({
                    'experiment': exp_idx + 1,
                    'iteration': iter_result['iteration'],
                    'cases_needed': iter_result['used_size'],
                    'accuracy': iter_result['accuracy'],
                    'support_vectors': sv_count
                })
                break
    
    if converged_experiments:
        print(f"Total experiments that converged: {len(converged_experiments)}/{args.iterations}")
        print("\nIterations and Cases Needed:")
        if args.classifier == 'svm':
            print("-" * 60)
            for conv in converged_experiments:
                print(f"Experiment {conv['experiment']:2d}: Iteration {conv['iteration']:2d}, Cases: {conv['cases_needed']:3d}, Accuracy: {conv['accuracy']:.4f}, Support Vectors: {conv['support_vectors']}")
        else:
            print("-" * 40)
            for conv in converged_experiments:
                print(f"Experiment {conv['experiment']:2d}: Iteration {conv['iteration']:2d}, Cases: {conv['cases_needed']:3d}, Accuracy: {conv['accuracy']:.4f}")
        
        # Calculate statistics
        iterations_needed = [conv['iteration'] for conv in converged_experiments]
        cases_needed = [conv['cases_needed'] for conv in converged_experiments]
        
        print(f"\nStatistics:")
        print(f"Average iterations needed: {np.mean(iterations_needed):.1f}")
        print(f"Average cases needed: {np.mean(cases_needed):.1f}")
        print(f"Min cases needed: {min(cases_needed)}")
        print(f"Max cases needed: {max(cases_needed)}")
    else:
        print("No experiments reached the threshold!")
    
    print("="*60)

if __name__ == "__main__":
    main()
