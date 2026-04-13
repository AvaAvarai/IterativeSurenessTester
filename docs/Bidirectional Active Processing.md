# Bidirectional Active Processing (BAP)

Our Electronics publication that defines this algorithm and methodology is available both [online](https://www.mdpi.com/2079-9292/15/3/580) and [locally](../bidirectional%20active%20processing/Quantifying%20AI%20Model%20Trust%20as%20a%20Model%20Sureness%20Measure%20by%20Bidirectional%20Active%20Processing%20and%20Visual%20Knowledge%20Discovery.pdf).

Our original implementation referenced in our Electronics publication is the IterativeSureness Tester Git repository script `rebuild.py`:
- Local: [IterativeSurenessTester](../../software/IterativeSurenessTester/)  
- Cloud: [Iterative Sureness Tester](https://github.com/AvaAvarai/IterativeSurenessTester)  

This definition is a respecification of the algorithm published in our Electronics publication. This version is for personal use in our research. This version will eventually replace the version found in our Github repository Iterative Sureness Tester. This is a baseline version which has one major assumption for simplicity: The ML classifier used is singular and not an ensemble or stacked model. Using an ML classifier that is multiple would change sampling, voting, and tie resolution.

Algorithm input of Iterative Sureness Tester Annotated with relation to this new definition:
```python
    # Now called train:
        parser.add_argument('--data', required=True, help='Training data file (or single dataset to split)')
    # Now called test:
        parser.add_argument('--test-data', help='Test data file (optional - if provided, data will not be split)')
    # Now called split:
        parser.add_argument('--train-pct', type=float, default=0.7, help='Training percentage (ignored if --test-data provided)')
        parser.add_argument('--test-pct', type=float, default=0.3, help='Test percentage (ignored if --test-data provided)')
    # Still called classifier:
        parser.add_argument('--classifier', required=True, choices=['dt', 'knn', 'svm'], help='Classifier type')
    # Now called parameters 
        parser.add_argument('--k', type=int, default=3, help='k for KNN')
    # Now called distance:
        parser.add_argument('--metric', default='euclidean', help='Distance metric for KNN')
    # Now called t:
        parser.add_argument('--threshold', type=float, default=0.95, help='Accuracy threshold')
    # Still called m:
        parser.add_argument('--m', type=int, default=5, help='Cases per iteration (or single m value)')
    # No longer an option, can run multiple times or via a script to do this:
        parser.add_argument('--m-values', help='Comma-separated list of m values to test (e.g., "1,5,10,20,50,100")')
    # No longer an option, can run multiple times or via a script to do this:
        parser.add_argument('--m-range', help='Range of m values: start,stop,step (e.g., "1,100,10")')
    # Now called n:
        parser.add_argument('--iterations', type=int, default=10, help='Number of experiments per split')
    # Still called splits:
        parser.add_argument('--splits', type=int, default=1, help='Number of splits to test')
    # Now called direction:
        parser.add_argument('--action', choices=['additive', 'subtractive'], default='additive', help='Action type')
    # not essential will use in future work:
        parser.add_argument('--plot', action='store_true', help='Create plots (default: False)')
```

Input should be entered either with flags passing them or a configuration file in TOML.

Algorithm Input (parameter name with colon then definition):
    - train: CSV file of labeled ML train dataset  
    - testing: Enum flag denoting test data source:  
        - testing.fixed: Test labeled ML dataset (e.g. MNIST dataset has a test dataset)  
            - test: CSV file of labeled ML test dataset  
        - testing.split: Static split  
            - split: Split ratio of train to test data (e.g. 80:20)  
        - testing.cv: Cross-validation  
            - folds: Number of folds to use (e.g. 5-fold CV or 10-fold CV)  
    - classifier: ML classifier algorithm  
        - parameters: Classifier (hyper)parameter set  
    - distance: Distance metric to use  
    - goal: Stopping criterion of set construction iteration success  
        - t: Classifier accuracy (i.e. the criterion threshold that we will use.)  
    - direction: Enum flag denoting direction of set construction  
        - direction.forward: Forward (i.e. starts with empty set and adds to it.)  
        - direction.backward: Backward (i.e. start with full set and removes from it.)  
    - splits: number of splits to run n iterations on, set to 1 for fixed test set.  
    - n: Number of set construction iterations to be ran independently in parallel on same split  
    - m: Number of cases added/removed each iteration  
    - sampling: Enum flag denoting sampling method  
        - sampling.random: Random  
        - sampling.stratified: Stratified  
        - sampling.cluster: Cluster  
        - sampling.systematic: Systematic  
    - voting: Enum flag denoting classifier voting method  
        - voting.majority: Majority  
        - voting.probability: Probability  
        - voting.weighted: Weighted with the specification of case weights  
        - voting.plurality: Plurality  
        - voting.copeland: Copeland  
        - voting.bucklin: Bucklin  
    - tie: Enum flag denoting classifier tie resolution method  
        - tie.random: Random  
        - tie.stratified: Stratified  
        - tie.predefined: Predefined (e.g. order in dataset or smallest class first.)  
        - tie.confidence: Highest confidence  
        - tie.boundary: Decision boundary distance  
        - tie.majority: Majority class  
        - tie.cost: Most costly to miss  
    - seed: Pseudorandom number generator seed for reproducibility.  

Algorithm Output:
    - Configuration settings txt file  
    - CSV file for training case set reached in each iteration that meets goals with random seed noted in filenames  
    - table of statistics listed for each magnitude of iterations tested (i.e. if we test 10,000 times we want 10, 100, 1,000, and 10,000 each listed individually) with standard deviation for mean terms.  
        - Mean Cases needed to reach goal  
        - Mean Cases needed to reach goal  
        - Min Cases needed to reach goal  
        - Min Cases needed to reach goal percentage of total training data  
        - Max Cases needed to reach goal  
        - Max Cases needed to reach goal percentage of total training data  
        - Mean Model Accuracy of all models built  
        - Mean Model Accuracy of models that reach goal  
        - Convergence Rate as ratio of iterations that met goal to all iterations ran  
        - Mean Model Sureness measure ratio as 1 - (mean cases needed %).  
        - Case count needed to reach goal that required classifier voting  
        - Case count needed to reach goal perecentage of total training data that required classifier voting  

Algorithm:

set prng seed = seed
Repeat splits times in parallel:
    if not testing.external then split data to get train and test
    Repeat n times in parallel:
        if direction.forward initialize empty caseset else as all of train
        while classifier trained on caseset tested on test accuracy < t and cases remain to add or remove:
            if direction.forward add m cases from train to caseset else remove m cases from caseset using sampling
        if cases do not remain to add or remove:
            then run failed, note this for stats output
        else:
            run succeeded note this and save the caseset to file
        increment the seed

## Improvement Idea

- Visual Selection Seeding to reduce search space. We need to validate how much of a reduction this is. This is a way we can demonstrate the visual approach advantage.  
