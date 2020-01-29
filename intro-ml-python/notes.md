# Introduction to Machine Learning with Python: A Guide for Data Scientists

Authors: Andreas C. Müller and Sarah Guido

<img src="https://images-na.ssl-images-amazon.com/images/I/51wF9ONArKL.jpg" title="book" width="150" />

# Table of Contents

- [1. Introduction](#1-introduction)
- [2. Supervised Learning](#2-supervised-learning)
- [3. Unsupervised Learning and Preprocessing](#3-unsupervised-learning-and-preprocessing)
- [4. Representing Data and Engineering Features](#4-representing-data-and-engineering-features)
- [5. Model Evaluation and Improvement](#5-model-evaluation-and-improvement)
- [6. Algorithm Chains and Pipelines](#6-algorithm-chains-and-pipelines)
- [7. Working with Text Data](#7-working-with-text-data)
- [8. Wrapping Up](#8-wrapping-up)


# 1. Introduction
## Why ML?
Using handcoded rules to make decisions has two disadvantages:
- logic is specific to a domain and task. Change the task slightly -> rewrite the whole system
- designing rules requires a deep understanding of how a decision should be made by a human expert

## Knowing your task and knowing your data
When building a ML solution:
- What question(s) am I trying to answer? Do I think the data collected can answer that question?
- What is the best way to phrase my question(s) as a machine learning problem?
- Have I collected enough data to represent the problem I want to solve?
- What features of the data did I extract, and will these enable the right predictions?
- How will I measure success in my application?
- How will the machine learning solution interact with other parts of my research or business product?

> Many spend a lot of time building complex ML solutions, only to find out they don't solve the right problem. When going deep into the technical aspects of ML, it is easy to lose sight of the ultimate goals

### Jupyter notebook
Interactive environment for running code in the browser

### NumPy
- `ndarray`: multidimensional (*n*) array with elements of the same type
- high-level mathematical functions, such as linear algebra, Fourier transform, pseudorandom number generators

### SciPy
- `scipy.sparse`: provides sparse matrices
- advanced linear algebra routines, mathematical function optimization, signal processing, statistical distributions

### Matplotlib
- On jupyter: `%matplotlib inline`
- Primary scientific plotting library in Python

### Pandas
Library for data wrangling and analysis
- `DataFrame`: allows each column to have a separate type

### mglearn
Library of utility functions wrote for this specific book. Avoid boilerplate with plotting and loading data

## First things first: look at your data
Before building a ML model, inspect the data:
- task easily solvable without ML
- desired information may not be contained in the data
- find abnormalities and peculiarities
- real world: inconsistencies in the data and unexpected measurements are very common
- *scatter plot*: `pd.scatter_matrix`

# 2. Supervised Learning

# 3. Unsupervised Learning and Preprocessing

# 4. Representing Data and Engineering Features
**Feature engineering**: how to represent your data best for a particular application -> can have a bigger influence on the performance of a model than the exact parameters you choose

## Categorical Variables
### One-Hot Encoding (Dummy Variables)
Replace a categorical variable with one or more features that can have the values 0 and 1 -> introduce a new feature per category

> In pandas `pd.get_dummies(data)` automatically transform all columns that have object type or are categorical

### Numbers Can Encode Categoricals
The `get_dummies` function in pandas treats all numbers as continuous and will not create dummy variables for them. To get around this, use scikit-learn's `OneHotEncoder`

### Binning, Discretization, Linear Models and Trees
One way to make linear models more powerful on continuous data is to use *binning* (aka *discretization*) of the feature to split it up into multiple features

Binning features generally has no beneficial effect for tree-based models, as these models can learn to split up the data anywhere

### Interactions and Polynomials
Enrich a feature representation, particularly for linear models, is adding *interaction features* and *polynomial features* of the original data

### Univariate Nonlinear Transformations
Applying mathematical functions like:
- `log`, `exp`: help adjusting the relative scales in the data
- `sin`, `cos`: dealing with data that encodes periodic patterns

Most models work best when each feature (and in regression also the target) is loosely Gaussian distributed -> histogram should have something resembling the familiar "bell curve" shape. Using `log` or `exp` is a hacky but simple and efficient way to achieve this -> helpful when dealing with integer count data

> These kind of transformations are irrelevant for tree-based models, but might be essential for linear models. Sometimes it's also a good idea to transform the target variable in regression

## Automatic Feature Selection
Adding more features makes all models more complex, and so increases the chance of overfitting

It can be good idea to reduce the number of features to only the most useful ones, and discard the rest

### Univariate Statistics
Compute whether there is a statistically significant relationship between each feature and the target -> the features with highest confidence are selected. Also know as *analysis of variance* (ANOVA) for classification

Only consider each feature individually. *f_classify* or *f_regression* tests in scikit-learn and then *SelectKBest* or *SelectPercentile*

### Model-Based Feature Selection
Uses a supervised ML model to judge the importance of each feature, and keeps only the most important ones
- Decision tree-based models: *feature_importances_* attribute
- Linear models: coefficients can capture feature importances

> Model-based considers all features at once, so can capture interactions. *SelectFromModel*

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="median")
```

### Iterative Feature Selection
A series of models are built, with varying numbers of features. Two methods:
- starting with no features and adding one by one
- starting with all features and removing one by one

More computationally expensive

*recursive feature elimination* (RFE): starts with all features, builds a model, and discards the least important feature according to the model -> repeat

> **Feature Selection**: Can speed up prediction, allow for more interpretable model. In most real-world cases, is unlikely to provide large gains in performance

## Utilizing Expert Knowledge
Prior knowledge about the nature of the task can be encoded in the features to aid a ML algorithm

> Adding a feature does not force a machine learning algorithm to use it, and even if the holiday information turns out to be noninformative for flight prices, augmenting the data with this information doesn’t hurt.

- COOL example with bike rental on the book -> check to see the coefficients learned by the linear model

# 5. Model Evaluation and Improvement
## Cross-Validation
Data is split repeatedly and multiple models are trained. Most common: *k-fold cross-validation*

> Scikit-learn: `cross_val_score` from the *model_seleciton* module

High variance in the metric (e.g., accuracy) between folds -> model is very dependent on the particular folds for train, or it could also be consequence of the small size of the dataset

### Benefits of Cross-Validation
- avoid "lucky"/"unlucky" random train_test_split
- in cross-validation each example will be in the training set exactly once: each example is in one of the folds, and each fold is the test set once -> the model needs to generalize well to all of the samples in the dataset for all of the cross-validation scores to be high
- multiple splits provides information about how sensitive the model is to the selection of the training set -> idea of the best/worst case scenarios
- use data more effectively: more data usually leads to more accurate models 

Disadvantage: increased computational cost -> train *k* models instead of one

> Cross-validation does not return a model, its purpose is only to evaluate how well a given algorithm will generalize when trained on a specific dataset

### Stratified k-Fold Cross-Validation
*Stratified k-fold cross-validation*: split the data such that the proportions between classes are the same in each fold as they are in the whole dataset

Results in more reliable estimates of generalization performance

For regression scikit-learn uses the standard *k-fold cross-validation* by default

### Leave-one-out cross-validation
*k-fold cross-validation* where each fold is a single sample. `LeaveOneOut` on sklearn.model_selection

Can be very time consuming for large datasets, but sometimes provides better estimates on small datasets

### Shuffle-split cross-validation
Each split samples train_size many points for the training set and test_size many (disjoint) points for the test set. This splitting is repeated n_iter times

- allows for control over the number of iterations independently of the training and test sizes
- allows for using only part of the data in each iteration by providing train_size and test_size that don't add up to one -> subsampling like that can be useful for experimenting with large datasets

> `ShuffleSplit` and `StratifiedShuffleSplit` on sklearn

### Cross-validation with groups
When there are groups in the data that are highly related

`GroupKFold`: takes an array of groups as arguments -> indicates groups in the data that should not be split when creating the training and test sets, and should not be confused with the class label

> GroupKFold: important for medical applications (multiple samples for same patient), also speech recognition

## Grid Search
Trying all possible combinations of the parameters of interest

### The danger of overfitting the parameters and the validation set
To avoid this split the data in three sets: the training set to build the model, the validation (or development) set to select the parameters of the model, and the test set to evaluate the performance of the selected parameters

> After selecting the best parameters using the validation set, rebuild the model using the parameter settings found, but now training on both the training data and the validation data

**Important to keep the distinction of training, validation and test sets clear!** Evaluating more than one model on the test set and choosing the better of the two will result in an overly optimistic estimate of how accurate the model is -> "Leak" information from the test set into the model

### Grid Search with Cross-Validation
Beautiful plot from `mglearn`:

```python
mglearn.plots.plot_cross_val_selection()
```

`GridSearchCV` on sklearn: implemented in the form of an estimator -> not only searchs for the best parameters, but also automatically fits a new model on the whole training dataset with the parameters that yielded the best CV performance

> `best_score_` != `score`: first stores the mean CV accuracy performed in the training set, second evaluate the output of the predict method of the model trained on the whole training set!

## Analyzing the result of cross-validation
```python
# cool example of a heatmap using SVM
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
                      ylabel='C', yticklabels=param_grid['C'], cmap="viridis")
```

Optimum values for each parameter on the edges of the plot: parameters not large enough!

## Nested cross-validation
An outer loop over splits of the data into training and test sets. For each of them, a grid search is run (which might result in different best parameters for each split in the outer loop). Then, for each outer split, the test set score using the best settings is reported

- Rarely used in practice
- Useful for evaluating how well a given model works on a particular dataset
- Computationally expensive procedure

```python
# example of nested CV
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5),
                         iris.data, iris.target, cv=5)
```

## Evaluation Metrics and Scoring
### Keep the end goal in mind
- *Business metric*: We are interested in using the predictions as part of a larger decision-making process, you should think about the high-level goal of the application
- *Business impact*: consequences of choosing a particular algorithm for a ML application

When choosing a model or adjusting parameters, you should pick the model or parameter values that have the most positive influence on the business metric

> Sometimes infeasible to put models in production just for testing purposes (high business risk): find some surrogate evaluation procedure (as close as possible to the business goal), using an evaluation metric that is easier to compute

## Metrics for binary classification
### Kinds of errors
- **false positive**: incorrect positive prediction, **type I error**
- **false negative**: incorrect negative prediction, **type II error**

### Imbalanced datasets
Accuracy is an inadequate measure for quantifying predictive performance in most imbalanced settings
- `pred_most_frequent`: model that make predictions to the most frequent class
- `pred_dummy`: random predictions

### Confusion matrices
`confusion_matrix`
- rows: true classes
- columns: predicted classes

| -              | -                  | -                  |
| -------------- | ------------------ | ------------------ |
| negative class | TN                 | FP                 |
| positive class | FN                 | TP                 |
| -              | predicted negative | predicted positive |

- Accuracy = (TP+TN)/(TP+TN+FP+FN)

- Precision = (TP)/(TP+FP) 

Precision is used when the goal is to limit the number of false positives. AKA *positive predictive value (PPV)*

- Recall = (TP)/(TP+FN)

Recall is used when we need to identify all positive samples; that is, when it is important to avoid false negatives. AKA *sensitivity*, *hit rate*, or *true positive rate (TPR)*

- F-score or f-measure or f1-score -> F = 2*(precision*recall)/(precision+recall)

Harmonic mean of precision and recall

```python
from sklearn.metrics import classification_report
```

### Taking uncertainty into account
- You can change the decision threshold depending on the problem
- *calibration*: a calibrated model is a model that provides an accurate measure of its uncertainty

### Precision-recall curves and ROC curves
Setting a requirement on a classifier like 90% recall is often called setting the *operation point*

Precision-recall curve: look at all possible thresholds, or all possible trade-offs of precision and recalls at once. `precision_recall_curve`

> *Average precision*: Area under the precision-recall curve. `average_precision_score`

Receiver operating characteristics (ROC) curve: consider all possible thresholds for a given classifier, shows the *false positive rate (FPR)* against the *true positive rate (TPR)*. `roc_curve`

- TPR = recall = (TP)/(TP+FN)
- FPR = (FP)/(FP+TN)

Average precision always returns a value between 0 (worst) and 1 (best). Predicting randomly always produces an AUC of 0.5 -> AUC is much better than accuracy as a metric for imbalanced datasets

AUC: evaluating the ranking of positive samples. Probability that a randomly picked point of the positive class will have a higher score according to the classifier than a randomly picked point from the negative class

> AUC does not make use of the default threshold, so adjusting the decision threshold might be necessary to obtain useful classification results from a model with a high AUC

## Metrics for Multiclass Classification
Metrics for multiclass classification are derived from binary, but averaged over all classes

For imbalanced datasets: multiclass f-score -> one binary f-score per class (that being the positive) and others being the negative -> then average these per-class f-scores

## Regression metrics
- Rˆ2 is enough for most applications
- Mean squared error (MSE)
- Mean absolute error (MAE)

## Using evaluation metrics in model selection
`scoring` parameters for classification:
- `accuracy`
- `roc_auc`
- `average_precision`
- `f1`, `f1_macro`, `f1_micro`, `f1_weighted`

for regression:
- `r2`
- `mean_squared_error`
- `mean_absolute_error`

for more, see: `sklearn.metrics.scores`

# 6. Algorithm Chains and Pipelines
ML algorithms requires chaining together many different processing steps and ML models. `Pipeline` class simplify the process of building chains of transformations and models

## Parameter selection with preprocessing
```python
mglearn.plots.plot_improper_processing()
```

> Splitting the dataset during cross-validation should be done *before doing any preprocessing*. Any process that extracts knowledge from the dataset should only ever be applied to the training portion of the dataset, so any CV should be the "outermost loop" in your processing

## Building Pipelines
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

- reduce the code needed for "preprocessing + classification"
- main benefit: now you can use this single estimator in `cross_val_score` or `GridSearchCV`

```python
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_params_)
```

> For each split in the CV, the Scaler is refit with only the training splits and no information is leaked from the test split in to the parameter search

## Information Leakage
The impact varies depending on the preprocessing step:
- estimating the scale of the data using the test fold usually doesn't have a terrible impact
- using the test fold in feature extraction and feature selection can lead to **substantial differences** in outcomes

## The General Pipeline Interface
- not restricted to preprocessing and classification
- only requirement: all estimators but the last step need to have a transform method
- during `Pipeline.fit`, the pipeline calls fit and then transform on each step; for the last step, just fit is called
- when predicting, we similarly transform the data using all but the last step, and then call predict on the last step
- there is no requirement to have predict in the last step; the last step is only required to have a fit method

### Convenient pipeline creation with make_pipeline
```python
from sklearn.pipeline import make_pipeline
pipe_short = make_pipeline(StandardScaler(), LogisticRegression())
```

### Accessing step attributes
`named_steps` attribute -> dictionary from the step names to estimators
```python
components = pipe.named_steps["pca"].components_
```

```python
grid.best_estimator_.named_steps["logisticregression"].coef_
```

### Grid-Searching Which Model to Use
```python
pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])

from sklearn.ensemble import RandomForestClassifier

param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)],
     'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

# 7. Working with Text Data

# 8. Wrapping Up

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>