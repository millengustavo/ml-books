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

# 6. Algorithm Chains and Pipelines

# 7. Working with Text Data

# 8. Wrapping Up

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>