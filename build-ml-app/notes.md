# Building Machine Learning Powered Applications: Going from Idea to Product

Author: Emmanuel Ameisen

![build-ml-app-cover](cover.jpg)

# Part I. Find the Correct ML Approach

# Ch1. From Product Goal to ML Framing

> ML is particularly useful to build systems for which we are unable to define a heuristic solution

Start from a concrete business problem, determine whether it requires ML, then work on finding the ML approach that will allow you to iterate as rapidly as possible

1. Framing your product goal in an ML paradigm
2. Evaluating the feasibility of that ML task

Estimating the challenge of data acquisition ahead of time is crucial in order to succeed

## Data availability scenarios
- Labeled data exists
- Weakly labeled data exists
- Unlabeled data exists
- We need to acquire data

> "Having an imperfect dataset is entirely fine and shouldn’t stop you. The ML process is iterative in nature, so starting with a dataset and getting some initial results is the best way forward, regardless of the data quality."

## The Simplest Approach: being the algorithm
Start with a human heuristic and then build a simple model: initial baseline = first step toward a solution -> Great way to inform what to build next

## What to focus on in an ML project
Find the *impact bottleneck*: piece of the pipeline that could provide the most value if improved

Imagine that the impact bottleneck is solved: it was worth the effort you estimate it would take?

## Which modeling techniques to use
Spend the manual effort to look at inputs and outputs of your model: see if anything looks weird. Looking at your data helps you think of good heuristics, models and ways to reframe the product

# Ch2. Create a Plan

## Measuring Success
First model: simplest model that could address a product's needs -> generating and analyzing results is the fastest way to make progress in ML

- Baseline: heuristics based on domain knowledge
- Simple model
- Complex model

> You don't always need ML: even features that could benefit from ML can often simply use a heuristic for their first version (you may realize that you don't need ML at all)

## Business Performance
Product metrics: goals of your product or feature. Ultimately the only ones that matter, all other metrics should be used as tools to improve product metrics

### Updating an app to make a modeling task easier
- Change an interface so that a model's results can be omitted if they are below a confidence threshold
- Present a few other predictions or heuristics in addition to model's top prediction
- Communicate to users that model is still in an experimental phase and giving them opportunities to provide feedback

> "A product should be designed with reasonable assumptions of model performance in mind. If a product relies on a model being perfect to be useful, it is very likely to produce innacurate or even dangerous results"

## Freshness and Distribution Shift
Distribution of the data shifts -> model often needs to change in order to maintain the same level of performance

## Leverage Domain Expertise
Best way to devise heuristics -> see what experts are currently doing. Most practical applications are not entirely novel. How do people currently solve the problem you are trying to solve?

Second best way -> look at your data. Based on your dataset, how would you solve this task if you were doing it manually?

### Examining the data
EDA: process of visualizing and exploring a dataset -> to get an intuition to a given business problem. Crucial part of building any data product

## Stand on the Shoulders of giants
1. Reproduce existing results
2. Build on top of them

## To make regular progress: start simple
1. Start with the simplest model that could address your requirements
2. Build an end-to-end prototype including this model
3. Judge its performance: optimization metrics and product goal

Looking at the performance of a simple model on an initial dataset is the best way to decide what task should be tackled next

## Diagnose Problems
Write analysis and exploration functions:
- Visualize examples the model performs the best and worst on
- Explore data
- Explore model results

# Part II. Build a Working Pipeline

# Ch3. Build your first end-to-end pipeline
First iteration: lackluster by design. Goal: allow us to have all the pieces of a pipeline in place:
- prioritize which ones to improve next
- identify the impact bottleneck

> "Frequently, your product is dead even if your model is successful" - Monica Rogati

## Test your workflow
Evaluate:
- usefulness of the current user experience
- results of your handcrafted model

## Finding the impact bottleneck
Next challenge to tackle next:
- iterating on the way we present results to the users or;
- improving model performance by identifying key failure points

# Ch4. Acquire an initial dataset
Understanding your data well leads to the biggest performance improvements

## Iterate on datasets
Data gathering, preparation and labeling should be seen as an iterative process, just like modeling

**ML engineering**: engineering + ML = products

Choosing an initial dataset, regularly updating it, and augmenting it is the majority of the work

**Data**: best source of inspiration to develop new models and the first place to look for answers when things go wrong

> Models only serve as a way to extract trends and patterns from existing data. Don't overestimate the impact of working on the model and underestimate the value of working on the data

Before noticing predictive trends, start by examining **quality**

## Data quality rubric

### Format
Validate that you understand the way in which the data was processed

### Quality
Notice the quality **ahead of time** -> missing labels, weak labels

### Quantity and distribution
Estimate:
- enough data?
- feature values are within reasonable range?

### Summary statistics
> Identifying differences in distributions between classes of data early: will either make our modeling task easier or prevent us from overestimating the performance of a model that may just be leveraging one particularly informative feature.

### Data leakage
Using training and validation data for vectorizing/preprocessing can cause data leakage -> leveraging info from outside the training set to create training features

## Clustering
As with dimensionality reduction: additional way to surface issues and interesting data points

## Let data inform features and models
The more data you have and the less noisy your data is, the less feature engineering work you usually have to do

### Feature crosses
Feature generated by multiplying (crossing) two or more features -> nonlinear combination of features -> allows our model to discriminate more easily

### Giving your model the answer
New binary feature that takes a nonzero value only when relevant combination of values appear

## Robert Munro: how do you find, label and leverage data

### Uncertainty sampling
Identify examples that your model is most uncertain about and find similar examples to add to the training set

### "Error model"
Use the mistakes your model makes as labels: "predicted correctly" or "predicted incorrectly". Use the trained error model on unlabeled data and label the examples that it predicts your model will fail on

### "Labeling model"
To find the best examples to label next. Identify data points that are most different from what you've already labeled and label those

### Validation
While you should use strategies to gather data, you should always randomly sample from your test set to validate your model

# Part III. Iterate on Models

# Ch5. Train and evaluate your model

## The simplest appropriate model
Not the best approach: try every possible model, benchmark and pick the one with the best results on a test set

### Simple model
- Quick to implement: won't be your last
- Understandable: debug easily
- Deployable: fundamental requirement for a ML-powered application

> Model explainability and interpretability: ability for a model to expose reasons that caused it to make predictions

## Test set
"While using a test set is a best practice, practitioners sometimes use the validation set as a test set. This increases the risk of biasing a model toward the validation set but can be appropriate when running only a few experiments"

## Data leakage
- Temporal data leakage
- Sample contamination

Always investigate the results of a model, especially if it shows surprisingly strong performance

## Bias variance trade-off
- Underfitting: weak performance on the training set = high bias
- Overfitting: strong performance on the training set, but weak performance on the validation set = high variance

## Evaluate your model: look beyond accuracy
- **Contrast data and predictions**
- **Confusion matrix**: see whether our model is particularly successful on certain classes and struggles on some others
- **ROC Curve**: plot a threshold on it to have a more concrete goal than simply getting the largest AUC score
- **Calibration Curve**: whether our model's outputed probability represents its confidence well. Shows the fraction of true positive examples as a function of the confidence of our classifier
- **Dimensionality reduction for errors**: identify a region in which a model performs poorly and visualize a few data points in it
- **The top-k method**
  - **k best performing examples**: identify features that are successfully leveraged by a model
  - **k worst performing examples**: on train: identify trends in data the model fails on, identify additional features that would make them easier for a model. On validation: identify examples that significantly differ from the train data
  - **k most uncertain examples**: on train: often a symptom of conflicting labels. On validation: can help find gaps in your training data

> Top-k implementation: [book's Github repository](https://github.com/hundredblocks/ml-powered-applications/blob/master/ml_editor/model_evaluation.py#L250-L295)

## Evaluate Feature Importance
- Eliminate or iterate on features that are currently not helping the model
- Identify features that are suspiciously predictive, which is often a sign of data leakage

## Black-box explainers
Attempt to explain a model's predictions independently of its inner workings, i.e. LIME and SHAP

# Ch6. Debug your ML problems