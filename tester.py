import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
from tqdm import tqdm
import os
import argparse

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Find the class column (case-insensitive)
    class_col = [col for col in df.columns if col.lower() == 'class'][0]
    
    # Convert class column to string type
    df[class_col] = df[class_col].astype(str)
    
    # Separate features and labels
    X = df.drop(columns=[class_col])
    y = df[class_col]
    
    return X, y, df[class_col].unique()

def get_classifier(classifier_type, k_value=None):
    """
    Factory function to create classifiers
    """
    if classifier_type.lower() == 'svm':
        return SVC(kernel='rbf', probability=True)
    elif classifier_type.lower() == 'knn':
        if k_value is None:
            raise ValueError("k_value must be specified for KNN classifier")
        return KNeighborsClassifier(n_neighbors=k_value)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

def run_experiment(X, y, classifier_type='svm', k_value=None, n_experiments=10, increment_size=5, threshold=0.95):
    n_samples = len(X)
    eval_size = int(0.3 * n_samples)
    train_size = n_samples - eval_size
    
    # Store results
    all_accuracies = []
    threshold_reached = []  # Store when threshold is reached for each experiment
    
    for exp in tqdm(range(n_experiments), desc="Running experiments"):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X.iloc[indices]
        y_shuffled = y.iloc[indices]
        
        # Split into train and eval
        X_eval = X_shuffled[:eval_size]
        y_eval = y_shuffled[:eval_size]
        X_train = X_shuffled[eval_size:]
        y_train = y_shuffled[eval_size:]
        
        # Track accuracies for this experiment
        exp_accuracies = []
        threshold_reached_this_exp = None
        
        # Incremental training
        for i in range(increment_size, train_size + 1, increment_size):
            # Train on current subset
            X_train_subset = X_train[:i]
            y_train_subset = y_train[:i]
            
            # Skip if not enough classes in training subset
            if len(np.unique(y_train_subset)) < 2:
                exp_accuracies.append(None)
                continue
            
            # Get and train classifier
            clf = get_classifier(classifier_type, k_value)
            clf.fit(X_train_subset, y_train_subset)
            
            # Evaluate
            y_pred = clf.predict(X_eval)
            accuracy = accuracy_score(y_eval, y_pred)
            exp_accuracies.append(accuracy)
            
            # Check if threshold is reached
            if threshold_reached_this_exp is None and accuracy >= threshold:
                threshold_reached_this_exp = i
        
        all_accuracies.append(exp_accuracies)
        threshold_reached.append(threshold_reached_this_exp)
    
    return all_accuracies, threshold_reached

def plot_results(all_accuracies, threshold_reached, increment_size, threshold, classifier_type):
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot accuracy progression
    plt.figure(figsize=(12, 6))
    
    # Convert None values to NaN and calculate statistics
    valid_accuracies = np.array([[acc if acc is not None else np.nan for acc in exp] for exp in all_accuracies])
    mean_accuracies = np.nanmean(valid_accuracies, axis=0)
    std_accuracies = np.nanstd(valid_accuracies, axis=0)
    
    x = np.arange(increment_size, len(mean_accuracies) * increment_size + increment_size, increment_size)
    plt.plot(x, mean_accuracies, 'b-', label='Mean Accuracy')
    plt.fill_between(x, mean_accuracies - std_accuracies, mean_accuracies + std_accuracies, alpha=0.2)
    
    # Add threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    # Plot threshold reached points
    valid_thresholds = [t for t in threshold_reached if t is not None]
    if valid_thresholds:
        plt.scatter(valid_thresholds, [threshold] * len(valid_thresholds), 
                   color='g', marker='*', s=100, label='Threshold Reached')
    
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.title(f'{classifier_type.upper()} Accuracy vs Training Set Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/accuracy_progression_{classifier_type.lower()}.png')
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run incremental learning experiments with selected classifier')
    parser.add_argument('--classifier', type=str, choices=['svm', 'knn'], required=True,
                      help='Classifier type to use (svm or knn)')
    parser.add_argument('--k', type=int,
                      help='k value for KNN classifier (required if classifier is knn)')
    parser.add_argument('--data', type=str, default='wbc9.csv',
                      help='Path to the dataset file (default: wbc9.csv)')
    parser.add_argument('--experiments', type=int, default=100,
                      help='Number of experiments to run (default: 100)')
    parser.add_argument('--increment', type=int, default=5,
                      help='Increment size for training (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.95,
                      help='Accuracy threshold to track (default: 0.95)')
    
    args = parser.parse_args()
    
    # Validate k value for KNN
    if args.classifier.lower() == 'knn' and args.k is None:
        parser.error("--k value is required when using KNN classifier")
    
    # Load and preprocess data
    X, y, classes = load_and_preprocess_data(args.data)
    
    # Run experiments with selected classifier
    print(f"\nRunning experiments with {args.classifier.upper()}")
    if args.classifier.lower() == 'knn':
        print(f"Using k={args.k} for KNN")
    
    all_accuracies, threshold_reached = run_experiment(
        X, y, args.classifier, args.k, args.experiments, args.increment, args.threshold
    )
    
    # Plot results
    plot_results(all_accuracies, threshold_reached, args.increment, args.threshold, args.classifier)
    
    # Print summary statistics
    valid_accuracies = np.array([[acc if acc is not None else np.nan for acc in exp] for exp in all_accuracies])
    mean_accuracies = np.nanmean(valid_accuracies, axis=0)
    std_accuracies = np.nanstd(valid_accuracies, axis=0)
    
    print(f"\n{args.classifier.upper()} Results:")
    print(f"Number of experiments: {args.experiments}")
    print(f"Increment size: {args.increment}")
    print(f"Accuracy threshold: {args.threshold}")
    if args.classifier.lower() == 'knn':
        print(f"k value: {args.k}")
    print("\nFinal accuracy statistics:")
    print(f"Mean accuracy: {mean_accuracies[-1]:.3f} ± {std_accuracies[-1]:.3f}")
    
    # Print threshold statistics
    valid_thresholds = [t for t in threshold_reached if t is not None]
    if valid_thresholds:
        print(f"\nThreshold ({args.threshold}) reached statistics:")
        print(f"Mean samples needed: {np.mean(valid_thresholds):.1f} ± {np.std(valid_thresholds):.1f}")
        print(f"Min samples needed: {min(valid_thresholds)}")
        print(f"Max samples needed: {max(valid_thresholds)}")
        print(f"Number of experiments reaching threshold: {len(valid_thresholds)}/{args.experiments}")
    else:
        print(f"\nThreshold ({args.threshold}) was not reached in any experiment")

if __name__ == "__main__":
    main()
