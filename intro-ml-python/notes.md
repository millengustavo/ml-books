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

# 5. Model Evaluation and Improvement

# 6. Algorithm Chains and Pipelines

# 7. Working with Text Data

# 8. Wrapping Up

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>