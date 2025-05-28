# IterativeSurenessTester

First sureness measure test tool with Supervised Iterative Learning alternative to Active Learning to find ML model stability point in supervised classifiers.

![Screenshot](parallel_coordinates_grid.png)

## Usage

```bash
python tester.py [options]
```

### Required Arguments

- `--classifier {svm,knn}`: Type of classifier to use
  - `svm`: Support Vector Machine classifier
  - `knn`: K-Nearest Neighbors classifier

- `--train-data PATH`: Path to the training dataset file (CSV format)

### Optional Arguments

- `--k INT`: k value for KNN classifier (required if using KNN)
- `--test-data PATH`: Path to the test dataset file (CSV format). If not provided, the training data will be split into train/test sets.
- `--experiments INT`: Number of experiments to run (default: 100)
- `--increment INT`: Increment size for training (default: 5)
- `--threshold FLOAT`: Accuracy threshold to track (default: 0.95)
- `--plot`: Enable parallel coordinates plots visualization
- `--save-converged`: Save converged data to CSV files (saved in results directory)

### Example Commands

1. Run KNN with k=3 on MNIST dataset:
```bash
python tester.py --classifier knn --k 3 --train-data mnist_train_dr.csv --test-data mnist_test_dr.csv --experiments 1 --increment 100
```

2. Run SVM with default settings and save converged data:
```bash
python tester.py --classifier svm --train-data data.csv --save-converged
```

3. Run KNN with visualization and custom threshold:
```bash
python tester.py --classifier knn --k 5 --train-data data.csv --threshold 0.90 --plot
```

## Output

The tool generates several outputs in the `results` directory:

1. `accuracy_progression_{classifier}.png`: Plot showing accuracy progression over training set size
2. `parallel_coordinates_grid.png`: Parallel coordinates plot showing data distribution (if --plot is enabled)
3. `converged_data_exp_{n}.csv`: CSV files containing the converged training subsets (if --save-converged is enabled)

## Data Format

Input CSV files should contain:
- Features as columns
- A column named 'class' (case-insensitive) containing the class labels
- All features should be numeric
