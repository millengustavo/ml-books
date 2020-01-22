# The Hundred-Page Machine Learning Book 
Author: Andriy Burkov

<img src="http://themlbook.com/images/og-image3.png" title="book" width="800" />

# Table of Contents

- [1 Introduction](#1-introduction)
  * [What is Machine Learning](#what-is-machine-learning)
  * [Supervised Learning](#supervised-learning)
  * [Unsupervised Learning](#unsupervised-learning)
  * [Semi-supervised Learning](#semi-supervised-learning)
  * [Reinforcement Learning](#reinforcement-learning)
  * [Why the Model Works on New Data](#why-the-model-works-on-new-data)
- [2 Notation and Definitions](#2-notation-and-definitions)
- [3 Fundamental Algorithms](#3-fundamental-algorithms)
- [4 Anatomy of a Learning Algorithm](#4-anatomy-of-a-learning-algorithm)
  * [Building Blocks of a Learning Algorithm](#building-blocks-of-a-learning-algorithm)
  * [Gradient Descent](#gradient-descent)
  * [How Machine Learning Engineers Work](#how-machine-learning-engineers-work)
  * [Learning Algorithms' Particularities](#learning-algorithms--particularities)
- [5 Basic Practice](#5-basic-practice)
  * [Feature Engineering](#feature-engineering)
  * [One-Hot Encoding](#one-hot-encoding)
  * [Binning](#binning)
  * [Normalization](#normalization)
  * [Standardization](#standardization)
  * [Dealing with Missing Features](#dealing-with-missing-features)
  * [Data Imputation Techniques](#data-imputation-techniques)
  * [Learning Algorithm Selection](#learning-algorithm-selection)
  * [Three Sets](#three-sets)
  * [Underfitting and Overfitting](#underfitting-and-overfitting)
  * [Regularization](#regularization)
  * [Model Performance Assessment](#model-performance-assessment)
    + [Confusion Matrix](#confusion-matrix)
    + [Precision/Recall](#precision-recall)
    + [Accuracy](#accuracy)
    + [Cost-Sensitive Accuracy](#cost-sensitive-accuracy)
    + [Area under the ROC Curve (AUC)](#area-under-the-roc-curve--auc-)
    + [Hyperparameter Tuning](#hyperparameter-tuning)
    + [Cross-Validation](#cross-validation)
- [6 Neural Networks and Deep Learning](#6-neural-networks-and-deep-learning)
- [7 Problems and Solutions](#7-problems-and-solutions)
- [8 Advanced Practice](#8-advanced-practice)
  * [Handling Imbalanced Datasets](#handling-imbalanced-datasets)
  * [Combining Models](#combining-models)
  * [Training Neural Networks](#training-neural-networks)
  * [Advanced Regularization](#advanced-regularization)
  * [Handling Multiple Inputs](#handling-multiple-inputs)
  * [Handling Multiple Outputs](#handling-multiple-outputs)
  * [Transfer Learning](#transfer-learning)
  * [Algorithmic Efficiency](#algorithmic-efficiency)
- [9 Unsupervised Learning](#9-unsupervised-learning)
- [10 Other Forms of Learning](#10-other-forms-of-learning)
  * [Metric Learning](#metric-learning)
  * [Learning to Rank](#learning-to-rank)
  * [Learning to Recommend](#learning-to-recommend)
    + [Factorization Machines (FM)](#factorization-machines--fm-)
    + [Denoising Autoencoders (DAE)](#denoising-autoencoders--dae-)
  * [Self-Supervised Learning: Word Embeddings](#self-supervised-learning--word-embeddings)
- [11 Conclusion](#11-conclusion)
  * [Topic Modeling](#topic-modeling)
  * [Gaussian Process (GP)](#gaussian-process--gp-)
  * [Generalized Linear Models (GLM)](#generalized-linear-models--glm-)
  * [Probabilistic Graphical Models (PGM)](#probabilistic-graphical-models--pgm-)
  * [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo--mcmc-)
  * [Generative Adversarial Networks (GAN)](#generative-adversarial-networks--gan-)
  * [Genetic Algorithms (GA)](#genetic-algorithms--ga-)
  * [Reinforcement Learning (RL)](#reinforcement-learning--rl-)

# 1 Introduction
## What is Machine Learning
Process of solving a practical problem:
1. Gathering a dataset
2. Algorithmically building a statistical model based on that dataset to be used somehow to solve the practical problem

## Supervised Learning
Dataset is a collection of **labeled examples**

Goal is to use the dataset to produce a model that takes a feature vector as input and outputs information that allows deducing the label for this feature vector

## Unsupervised Learning
Dataset is a collection of **unlabeled examples**

Goal is to create a model that takes a feature vector as input and either transforms it into another vector or into a value that can be used to solve a practical problem

## Semi-supervised Learning
Dataset contains both labeled and unlabeled examples. Usually unlabeled quantity >> labeled quantity

Goal is the same as supervised learning. When you add unlabeled examples, you add more information about your problem, a larger sample reflects better the probability distribution the data we labeled came from

## Reinforcement Learning
Machine "lives" in an environment and is capable of perceiving the **state** as a vector of features. Machine can execute **actions** in every state. Different actions bring different **rewards** and could also move the machine to another state.

The goal of RL algorithm is to learn a **policy**. A policy is a function that takes the feature vector of a state as input and outputs an optimal action to execute. The action is optimal if it maximizes the expected average reward

## Why the Model Works on New Data
*PAC ("probably approximately correct") learning*: theory that helps to analyze whether and under what conditions a learning algorithm will probably output an approximately correct classifier

# 2 Notation and Definitions

# 3 Fundamental Algorithms

# 4 Anatomy of a Learning Algorithm

## Building Blocks of a Learning Algorithm
1. a loss function
2. an optimization criterion based on the loss function
3. an optimization routine leveraging training data to find a solution to the optimization criterion

## Gradient Descent
Iterative optimization algorithm for finding the minimum of a function

Find a *local minimum*: Starts at some random point and takes steps proportional to the negative of the gradient of the function at the current point

Gradient descent proceeds in **epochs**. Epoch: using the training set entirely to update each parameter

The learning rate controls the size of an update

Regular gradient descent is sensitive to the choice of the learning rate and slow for large datasets

> **Minibatch stochastic gradient descent (SGD)**: speed up the computation by approximating the gradient descent using smaller batches (subsets) of the training data.

Upgrades to SGD:
- Adagrad
- Momentum
- RMSprop
- Adam

## How Machine Learning Engineers Work
You don't implement algorithms yourself, you use libraries, most of which are open source -> *scikit-learn*

## Learning Algorithms' Particularities
- different hyperparameters
- some can accept categorical features
- some allow the data analyst to provide weightings for each class -> influence the decision boundary
- some given a feature vector only output the class -> others the probability
- some allow for online learning
- some can be used for both classification and regression

# 5 Basic Practice

## Feature Engineering
Transforming raw data into a dataset. Labor-intensive process, demands creativity and domain knowledge

Highly informative features = high predictive power

**Low bias**: predicts the training data well

## One-Hot Encoding
Transform categorical feature into several binary ones -> increase the dimensionality of the feature vectors

## Binning
Also called bucketing. Convert a continuous feature into multiple binary features (bins or buckets), based on value range

Can help the learning algorithm to learn using fewer examples

## Normalization
Converting the actual range of values into a standard range of values, typically in the interval [-1, 1] or [0, 1]

Can increase speed of learning. Avoid numerical overflow

## Standardization
Also called **z-score normalization**. Values are rescaled so that they have the properties of a standard normal distribution with `mean=0` and `stdev=1`

If feature has outliers -> prefer standardization than normalization

Feature rescaling -> usually benefical to most learning algorithms

## Dealing with Missing Features
- Remove the examples (if data big enough)
- Using algorithm that can deal with missing features
- Data imputation

## Data Imputation Techniques
- Average
- Median
- Value outside the normal range (i.e., -1 for data in [0, 1])
- Use the missing value as target for a regression problem
- If data large enough: add binary indicator (another column) for each feature with missing value

> Use the same data imputation technique to fill the missing values on the test set you used to complete the training data

## Learning Algorithm Selection
- Explainability
- In-memory vs. out-of-memory
- Number of features and examples
- Categorical vs. numerical features
- Nonlinearity of the data
- Training speed
- Prediction speed
- Test on the validation set

## Three Sets
1. Training set
2. Validation set
3. Test set

Shuffle the examples and split the dataset into three subsets. Training set is usually the biggest one, use it to build the model. Validation and test sets are roughly the same sizes, much smaller than the training set. The learning algorithm cannot use these two subsets to build the model -> those two are also often called *holdout sets*

Why two holdout sets? We use the validation set to choose the learning algorithm and find the best hyperparameters. We use the test set to assess the model before putting it in production

## Underfitting and Overfitting
**High bias**: model makes many mistakes on the training data -> **underfitting**. Reasons:
- model is too simple for the data
- the features are not informative enough

Solutions:
- try a more complex model
- engineer features with higher predictive power

**Overfitting**: model predicts very well the training data but poorly the data from at least one of the two holdout sets. Reasons:
- model is too complex for the data
- too many features but a small number of training examples

**High variance**: error of the model due to its sensitivity to small fluctuations in the training set

The model learn the idiosyncrasies of the training set: the noise in the values of features, the sampling imperfection (due to small dataset size) and other artifacts extrinsic to the decision problem at hand but present in the training set

Solutions:
- try a simpler model
- reduce the dimensionality of the dataset
- add more training data
- regularize the model

## Regularization
Methods that force the learning algorithm to build a less complex model. Often leads to slightly higher bias but significantly reduces the variance -> **bias-variance tradeoff**

- **L1 regularization**: produces a sparse model, most of its parameters equal to zero. Makes feature selection by deciding which features are essential for prediction. *Lasso regularization*
- **L2 regularization**: penalizes larger weights, if your only goal is to decrease variance, L2 usually gives better results. L2 also has the advantage of being differentiable, so gradient descent can be used for optimizing the objective function. *Ridge Regularization*
- **Elastic net regularization**: combine L1 and L2

Neural networks also benefit from two other regularization techniques:
- Dropout
- Batch Normalization

Also non-mathematical methods have a regularization effect: data augmentation and early stopping

## Model Performance Assessment
Model *generalizes well*: model performs well on predicting the test set

Overfitting: error on the test data is *substantially higher* then the error obtained in the training data

### Confusion Matrix
Table that summarizes how successful the classification model is at predicting examples belonging to various classes

Used to calculate two other metrics: precision and recall

### Precision/Recall
- **Precision**: ratio of correct positive predictions to the overall number of positive predictions: TP/(TP+FP)
- **Recall**: ratio of correct positive predictions to the overall number of positive examples in the dataset: TP/(TP+FN)

In practice, almost always have to choose between high precision or high recall -> usually impossible to have both
- assign a higher weighting to the examples of a specific class
- tune hyperparameters to maximize precision or recall on the validation set
- vary the decision threshold for algorithms that return probabilities of classes

### Accuracy
Number of correctly classified examples divided by the total number of classified examples: (TP+TN)/(TP+TN+FP+FN)

Useful metric when errors in predicting all classes are equally important

### Cost-Sensitive Accuracy
When different classes have different importances

Assign a cost (positive number) to both types of mistakes: FP and FN. Then compute the counts TP, TN, FP, FN as usual and multiply the counst for FP and FN by the corresponding cost before calculating the accuracy normally

### Area under the ROC Curve (AUC)
ROC curve ("receiver operating characteristic", comes from radar engineering): use a combination of the **true positive rate** (define exactly as recall) and **false positive rate** (proportion of negative examples predicted incorrectly) to build up a summary picture of the classification performance

TPR = TP/(TP+FN)

FPR = FP/(FP+TN)

ROC curvers can only be used to assess classifiers that return some confidence score (or a probability) of prediction

The higher the **area under the ROC curve (AUC)** the better the classifier. AUC > 0.5 -> better than a random classifier. AUC = 1 -> perfect classifier -> TPR closer to 1 while keeping FPR near 0

### Hyperparameter Tuning
- **Grid Search**: when you have enough data for a validation set and the number of hyperparameters and their range is not too large
- **Random Search**: instead of providing discrete set of values to explore, you provide a statistical distribution for each hyperparameter from which values are randomly samples and set the total number of combinations you want to try
- **Bayesian hyperparameter optimization**: use past evaluation results to choose the next values to evaluate
- **Gradient-based techniques**
- **Evolutionary optimization techniques**

### Cross-Validation
When you have few training examples, it could be prohibitive to have both validation and test set. You would prefer to use more data to train the model. In such case, you only split your data into training and test. Then you use **cross-validation** on the training set to simulate a validation set

# 6 Neural Networks and Deep Learning

# 7 Problems and Solutions

# 8 Advanced Practice

## Handling Imbalanced Datasets
- Set the cost of misclassification of examples of the minority class higher
- oversampling -> make multiple copies of the example of some class
- undersampling -> randomly remove from training set some examples of the majority class
- synthetic minority oversampling technique (SMOTE)
- adaptive synthetic sampling method (ADASYN)
- algorithms less sensitive to imbalanced datasets: Decision trees, Random Forest, Gradient Boosting

## Combining Models
Ensemble models, typically combine models of the same nature. Boost performance by combining hundreds of weak models. We can sometimes get an additional performance gain by combining strong models made with different learning algorithms (two or three models):
- averaging
- majority vote
- stacking

> **Stacking**: building a meta-model that takes the output of base models as input. Make sure your stacked model performs better on the validation set than each of the base models you stacked. When several **uncorrelated** strong models agree they are more likely to agree on the correct outcome


## Training Neural Networks
- Challenge to convert your data into the input the network can work with (i.e., resize images, word embeddings)
- The choice of specific NN architecture is a difficult one
- Decide the number of layers, their type and size
- Regularization

## Advanced Regularization
For NNs, besides L1 and L2 regularization:
- Dropout: for each pass, temporarily exclude at random some units frrom the computation
- Early Stopping: stop training once observe a decreased performance on the validation set
- Batch Normalization: standardize the outputs of each layer
- Data augmentation: create a synthetic example from an original by applying various transformations

## Handling Multiple Inputs
Multimodal data -> e.g., input is an image and text and binary output indicates whether the text describes this image

It's hard to adapt shallow learning algorithms to work with multimodal data -> train one shallow model on the image and another one in the text 

## Handling Multiple Outputs
Some problems you would like to predict multiple outputs for one input -> sometimes can convert into a multi-label classification problem -> Subnetworks

## Transfer Learning
Pick an existing model trained on some dataset, and adapt this model to predict examples from another dataset, different from the one the model was built on

1. Build a deep model on the original big dataset
2. Compile a much smaller labeled dataset for your second model
3. Remove the last one or several layers from the first model
4. Replace the removed layers with new layers adapted for the new problem
5. "Freeze" the parameters of the layers remaining from the first model
6. Use your smaller labeled dataset and gradient descent to train the parameters of only the new layers

## Algorithmic Efficiency
**Big O notation**: classify algorithms according to how their running time or space requirements grow as the input size grows. Complexity measured in the worst case

- avoid using loops whenever possible
- use appropriate data structures (e.g., if order is not important, use a set instead of a list)
- use dict (hashmap) -> allows you to define a collection of key-value pairs with very fast lookups for keys
- use Scientific Python packages -> many methods implemented in C
- if you need to iterate over a vast collection of elements, use generators that create a function that return one element at a time rather than all the elements at once
- *cProfile* package to find inefficiencies
- *multiprocessing* package
- *PyPy*, *Numba*

# 9 Unsupervised Learning

# 10 Other Forms of Learning

## Metric Learning
You can create a metric that would work better for your dataset

> One-shot learning with siamese networks and triplet loss can be seen as metric learning problem

## Learning to Rank
Supervised learning problem (e.g., optimization of search results returned by a search engine for a query)

Three approaches:
- pointwise
- parwise
- listwise

> State of the art rank learning algorithm: **LambdaMART**. Listwise approach -> one popular metric that combines both precision and recall is called *mean average precision (MAP)*

In typical supervised learning algorithm, we optimize the cost instead of the metric (usually metrics are not differentiable). In LambdaMART the metric is optmized directly

## Learning to Recommend
- **Content-based filtering**: learning what users like based on the description of the content they consume -> user can be trapped in a "filter bubble"
- **Collaborative filtering**: recommendations to one user are computed based on what other users consume or rate -> content of the item consumed is ignored -> huge and extremely sparse matrix

Real-world recommender systems -> hybrid approach

### Factorization Machines (FM)
Explicity designed for sparse datasets. Users and items are encoded as one-hot vectors

### Denoising Autoencoders (DAE)
NN that reconstructs its input from the bottleneck layer. Ideal tool to build a recommender system: input is corrupted by noise while the output shouldn't be

Idea: new items a user could like are seen as if they were removed from the complete set by some corruption process -> goal of the denoising autoencoder is to reconstruct those removed items

> Another effective collaborative-filtering model is an FFNN with two inputs and one output

## Self-Supervised Learning: Word Embeddings
Word embeddings: feature vectors that represent words -> similar words have similar feature vectors

**word2vec**: pretrained embeddings for many languages are available to download online. **skip-gram**

> Self-supervised: the labeled examples get extracted from the unlabeled data such as text

# 11 Conclusion

## Topic Modeling
Prevalent unsupervised learning problem. **Latent Dirichlet Allocation (LDA)** -> You decide how many topics are in your collection, the algorithm assigns a topic to each word in this collection. To extract the topics from a document -> count how many words of each topic are present in that document

## Gaussian Process (GP)
Supervised learning method that competes with kernel regression

## Generalized Linear Models (GLM)
Generalization of the linear regression to modeling various forms of dependency between the input feature vector and the target

## Probabilistic Graphical Models (PGM)
One example: Conditional Random Fields (CRF) -> model the input sequence of words and relationships between the features and labels in this sequence as a sequential *dependency graph*

**Graph**: structure consisting of a colletion of nodes and edges that join a pair of nodes

> PGMs are also know under names of Bayesian networks, belief networks and probabilistic independence networks

## Markov Chain Monte Carlo (MCMC)
If you work with graphical models and want to sample examples from a very complex distribution defined by the dependency graph. MCMC is a class of algorithms for sampling from any probability distribution defined mathematically

## Generative Adversarial Networks (GAN)
Class of NN used in unsupervised learning. System of two neural networks contesting with each other in a *zero-sum game* setting

## Genetic Algorithms (GA)
Numerical optimization technique used to optimize undifferentiable optimization objective functions. Use concepts from evolutionary biology to search for a global optimum (minimum or maximum) of an optimization problem, by mimicking evolutionary biological processes

> GA allow finding solutions to any measurable optimization criteria (i.e., optimize hyperparameters of a learning algorithm -> typically much slower than gradient-based optimization techniques)

## Reinforcement Learning (RL)
Solves a very specific kind of problem where the decision making is sequential. There's an agent acting in a unknown environment. Each action brings a reward and moves the agent to another state of the envinronment. The goal of the agent is to optimize its long-term reward

