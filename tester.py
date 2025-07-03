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
import math

def load_and_preprocess_data(train_file, test_file=None):
    # Load the training dataset
    train_df = pd.read_csv(train_file)
    
    # Find the class column (case-insensitive)
    class_col = [col for col in train_df.columns if col.lower() == 'class'][0]
    
    # Convert class column to string type
    train_df[class_col] = train_df[class_col].astype(str)
    
    # Separate features and labels
    X_train = train_df.drop(columns=[class_col])
    y_train = train_df[class_col]
    
    if test_file:
        # Load test data
        test_df = pd.read_csv(test_file)
        test_df[class_col] = test_df[class_col].astype(str)
        X_test = test_df.drop(columns=[class_col])
        y_test = test_df[class_col]
        
        # Calculate min and max across both datasets
        combined_X = pd.concat([X_train, X_test])
        min_vals = combined_X.min()
        max_vals = combined_X.max()
        
        # Normalize both datasets using the same min and max values
        X_train = (X_train - min_vals) / (max_vals - min_vals)
        X_test = (X_test - min_vals) / (max_vals - min_vals)
        
        return X_train, y_train, X_test, y_test, train_df[class_col].unique()
    else:
        # If no test file, normalize just the training data
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        return X_train, y_train, None, None, train_df[class_col].unique()

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

def run_experiment(X, y, X_test=None, y_test=None, classifier_type='svm', k_value=None, n_experiments=10, increment_size=5, threshold=0.95, save_converged=False):
    n_samples = len(X)
    
    if X_test is None:
        # Use random split if no test file provided
        eval_size = int(0.3 * n_samples)
        train_size = n_samples - eval_size
    else:
        # Use all training data when test file is provided
        train_size = n_samples
    
    # Store results
    all_accuracies = []
    threshold_reached = []  # Store when threshold is reached for each experiment
    training_subsets = []  # Store the training subsets for plotting
    test_sets = []  # Store the test sets for plotting
    converged_data = []  # Store the converged data for each experiment
    
    for exp in tqdm(range(n_experiments), desc="Running experiments"):
        if X_test is None:
            # Shuffle data and split into train and eval
            indices = np.random.permutation(n_samples)
            X_shuffled = X.iloc[indices]
            y_shuffled = y.iloc[indices]
            
            X_eval = X_shuffled[:eval_size]
            y_eval = y_shuffled[:eval_size]
            X_train = X_shuffled[eval_size:]
            y_train = y_shuffled[eval_size:]
        else:
            # Use provided test data
            X_train = X
            y_train = y
            X_eval = X_test
            y_eval = y_test
        
        # Store test set for plotting
        test_sets.append((X_eval, y_eval))
        
        # Track accuracies for this experiment
        exp_accuracies = []
        threshold_reached_this_exp = None
        exp_subsets = []  # Store subsets for this experiment
        converged_subset = None  # Store the converged subset for this experiment
        
        # Incremental training
        for i in range(increment_size, train_size + 1, increment_size):
            # Train on current subset
            X_train_subset = X_train[:i]
            y_train_subset = y_train[:i]
            
            # Store subset for plotting
            exp_subsets.append((X_train_subset, y_train_subset))
            
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
                converged_subset = (X_train_subset, y_train_subset)
        
        all_accuracies.append(exp_accuracies)
        threshold_reached.append(threshold_reached_this_exp)
        training_subsets.append(exp_subsets)
        converged_data.append(converged_subset)
    
    # Save converged data if requested
    if save_converged and any(converged_data):
        os.makedirs('results', exist_ok=True)
        for i, (X_conv, y_conv) in enumerate(converged_data):
            if X_conv is not None:
                # Combine features and labels
                conv_df = X_conv.copy()
                conv_df['class'] = y_conv
                # Save to CSV
                conv_df.to_csv(f'results/converged_data_exp_{i+1}.csv', index=False)
    
    return all_accuracies, threshold_reached, training_subsets, test_sets

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

def plot_parallel_coordinates_grid(training_subsets, test_sets, increment_size, threshold_reached=None, threshold=0.95):
    """
    Create a grid of parallel coordinate plots showing data distribution at different subset sizes
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Use the first experiment's subsets for plotting
    exp_subsets = training_subsets[0]
    X_test, y_test = test_sets[0]
    
    # Check if threshold was reached in the first experiment
    threshold_subplot = None
    if threshold_reached and len(threshold_reached) > 0 and threshold_reached[0] is not None:
        # Calculate which subplot corresponds to the threshold
        threshold_subplot = (threshold_reached[0] // increment_size) - 1  # -1 because we start from increment_size
        print(f"\n🎯 THRESHOLD REACHED: Subplot {threshold_subplot + 1} (training subset size: {threshold_reached[0]}) reached accuracy threshold {threshold}")
    else:
        print(f"\n❌ THRESHOLD NOT REACHED: No subplot reached the accuracy threshold of {threshold}")
    
    # Get all unique classes and create consistent color mapping
    all_classes = set()
    for _, y_subset in exp_subsets:
        all_classes.update(y_subset.unique())
    all_classes.update(y_test.unique())
    
    # Create color mapping using Set2 colormap
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_classes)))
    class_colors = dict(zip(sorted(all_classes), colors))
    
    # Calculate grid dimensions (including test set plot)
    n_plots = len(exp_subsets) + 1  # +1 for test set
    n_cols = min(4, n_plots)  # Maximum 4 columns
    n_rows = math.ceil(n_plots / n_cols)
    
    # Create figure with subplots and constrained layout
    fig = plt.figure(figsize=(24, 6 * n_rows), constrained_layout=True)
    
    # Create a subplot for each subset size
    for i, (X_subset, y_subset) in enumerate(exp_subsets):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # Create DataFrame for parallel coordinates
        plot_data = X_subset.copy()
        plot_data['class'] = y_subset
        
        # Plot parallel coordinates with consistent colors
        for cls in sorted(all_classes):
            if cls in y_subset.unique():
                mask = plot_data['class'] == cls
                parallel_coordinates(plot_data[mask], 'class', color=[class_colors[cls]], ax=ax)
                ax.get_legend().remove()  # Remove individual legends
        
        # Customize plot
        subset_size = len(X_subset)
        if threshold_subplot is not None and i == threshold_subplot:
            ax.set_title(f'Training Subset Size: {subset_size} 🎯 THRESHOLD REACHED!', 
                        color='red', fontweight='bold', fontsize=12)
            # Add a red border around the threshold subplot
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        else:
            ax.set_title(f'Training Subset Size: {subset_size}')
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
    
    # Add test set plot as the last subplot
    ax = plt.subplot(n_rows, n_cols, n_plots)
    plot_data = X_test.copy()
    plot_data['class'] = y_test
    
    # Plot test set with consistent colors
    for cls in sorted(all_classes):
        if cls in y_test.unique():
            mask = plot_data['class'] == cls
            parallel_coordinates(plot_data[mask], 'class', color=[class_colors[cls]], ax=ax)
            ax.get_legend().remove()  # Remove individual legends
    
    ax.set_title(f'Test Set (Size: {len(X_test)})')
    ax.grid(True)
    plt.xticks(rotation=45)
    
    # Add single legend in bottom right corner
    handles = [plt.Line2D([0], [0], color=color, label=cls) 
              for cls, color in class_colors.items()]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.98, 0.02))
    
    # Save the figure with high DPI
    plt.savefig('results/parallel_coordinates_grid.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run incremental learning experiments with selected classifier')
    parser.add_argument('--classifier', type=str, choices=['svm', 'knn'], required=True,
                      help='Classifier type to use (svm or knn)')
    parser.add_argument('--k', type=int,
                      help='k value for KNN classifier (required if classifier is knn)')
    parser.add_argument('--train-data', type=str, required=True,
                      help='Path to the training dataset file')
    parser.add_argument('--test-data', type=str,
                      help='Path to the test dataset file (optional)')
    parser.add_argument('--experiments', type=int, default=100,
                      help='Number of experiments to run (default: 100)')
    parser.add_argument('--increment', type=int, default=5,
                      help='Increment size for training (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.95,
                      help='Accuracy threshold to track (default: 0.95)')
    parser.add_argument('--plot', action='store_true',
                      help='Show parallel coordinates plots (default: False)')
    parser.add_argument('--save-converged', action='store_true',
                      help='Save converged data to CSV files (default: False)')
    
    args = parser.parse_args()
    
    # Validate k value for KNN
    if args.classifier.lower() == 'knn' and args.k is None:
        parser.error("--k value is required when using KNN classifier")
    
    # Load and preprocess data
    X, y, X_test, y_test, classes = load_and_preprocess_data(args.train_data, args.test_data)
    
    # Run experiments with selected classifier
    print(f"\nRunning experiments with {args.classifier.upper()}")
    if args.classifier.lower() == 'knn':
        print(f"Using k={args.k} for KNN")
    
    all_accuracies, threshold_reached, training_subsets, test_sets = run_experiment(
        X, y, X_test, y_test, args.classifier, args.k, args.experiments, 
        args.increment, args.threshold, args.save_converged
    )
    
    # Create parallel coordinates grid plot if requested
    if args.plot:
        print("\nCreating parallel coordinates grid plot...")
        plot_parallel_coordinates_grid(training_subsets, test_sets, args.increment, threshold_reached, args.threshold)
    
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
