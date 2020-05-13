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