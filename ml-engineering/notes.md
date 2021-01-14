# Machine Learning Engineering
Author: Andriy Burkov

<img src="https://images-na.ssl-images-amazon.com/images/I/417OSxpSuIL._SX258_BO1,204,203,200_.jpg" title="book"/>

# Ch1. Introduction

When we deploy a model in production, we usually deploy an entire pipeline

**Machine learning engineering (MLE)**:
- encompasses data collection, model training, making the model available for use
- includes any activity that lets ML algorithms be implemented as a part of an effective production system

**ML Engineer**:
- concerned with sourcing the data (from multiple locations), preprocessing it, programming features, training an effective model that will coexist in production with other processes
- stable, maintanable and easily accessible
- ML systems "fail silently" -> must be capable of preventing such failures or to know how to detect and handle them

## When to use ML
Your problem:
- too complex for coding
- constantly changing
- perceptive (image, text, etc)
- unstudied phenomenon
- has a simple objective
- it is cost-effective

## When not to use ML
- explainability is needed
- errors are intolerable
- traditional SWE is a less expensive option
- all inputs and outputs can be enumerated and saved in a DB
- data is hard to get or too expensive

## ML Project Life Cycle
<img src="https://cdn.substack.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0ab34d2-0b33-4814-bc70-07c1bed10d42_2056x1068.png" title="diagram"/>


# Ch2. Before the Project Starts

## Impact of ML
High when:
- ML can replace a complex part of your engineering project
- there's great benefit in getting inexpensive (but probably imperfect) predictions

## Cost of ML
Factors:
- difficulty of the problem
- cost of data
- need for accuracy

## Nonlinear progress
> Progress in ML is nonlinear. Prediction error decreases fast in the beginning, but then gradually slows down

- Make sure the PO (or client) understands the constraints and risks
- Log every activity and track the time it took (helps with reporting and estimations of complexity in the future)

## Why ML projects fail
- lack of experienced talent
- lack of support by the leadership
- missing data infrastructure
- data labeling challenge
- siloed organizations and lack of collaboration
- technically infeasible projects
- lack of alignment between technical and business teams

# Ch3. Data Collections and Preparation

## Train, Validation and Test sets partition
- Data was randomized before the split
- Split was applied to raw data
- Validation and test sets follow the same distribution
- Leakage was avoided

## Data Sampling strategies
- random sampling
- systematic sampling
- stratified sampling
- cluster sampling

> Data versioning is a critical element in supervised learning when the labeling is done by multiple labelers

## Dataset Documentation
- what the data means
- how it was collected
- methods used to creat it 
- details of train-validation-test splits
- details of all pre-processing steps
- explanation of any data that were excluded
- format used to store it
- types of attributes/features
- number of examples
- possible values for labels / allowable range for a numerical target

# Ch4. Feature Engineering

## Good features
- high predictive power
- can be computed fast
- reliable
- uncorrelated

> The distribution of feature values in the training set should be similar to the distribution of values the production model will receive

## Feature selection techniques
- Cutting the long tail
- Boruta
- L1 regularization

## Best practices
- scale features
- store and document in schema files or feature stores
- keep code, model and training data in sync

> "Feature extraction code is one of the most important parts of a machine learning system. It must be extensively and systematically tested"

# Ch5. Supervised Model Training (Part 1)

## Baseline
> **Baseline**: a model or algorithm that provides a reference point for comparison. Establish a baseline performance on your problem before start working on a predictive model. 

- simple learning algorithm or
- rule-based or heuristic algorithm (simple statistic)
  - random prediction
  - zero rule algorithm (e.g., always predict the most common class in the training set / average if regression)
- human baseline: Amazon Mechanical Turk (MT) service -> web-platform where people solve simple tasks for a reward

## In-memory vs. out-of-memory

If the dataset can't be fully loaded in RAM -> **incremental learning algorithms**: can improve the model by reading data gradually (Naive Bayes, neural networks)

## Precision and Recall
- **Precision**: ratio of true positive predictions to the overall number of positive PREDICTIONS
- **Recall**: ratio of true positive predictions to the overall number of positive EXAMPLES

## F-measure
- positive real *beta*
- **beta = 2** -> weighs recall twice as high as precision
- **beta = 0.5** -> weighs recall twice as low as precision

## Precision-recall and bias-variance tradeoffs

> By varying the complexity of the model, we can reach the so-called "zone of solutions", a situation in which both bias and variance of the model are relatively low. The solution that optimizes the performance metric is usually found inside that zone

# Ch6. Supervised Model Training (Part 2)

-

# Ch7. Model Evaluation

## Tasks
- estimate legal **risks** of putting the model in production
- understand the **distribution of the data** used to train the model
- **evaluate** the performance of the model prior to deployment
- **monitor** the performance of the deployed model

## A/B Testing
- A: served the old model
- B: served the new model
- apply a statistical significance test to decide whether the new model is statistically different from the old model

## Multi-armed bandit
- start by randomly exposing all models to the users
- gradually reduce the exposure of the least-performing models until only one (the best performing) gets served most of the time

## Bootstrapping
- technique (statistical procedure) to compute a statistical interval for any metric
- consists of building N samples of a dataset
- then training a model
- and computing some statistic using each of those N samples

# Ch8. Model Deployment

## Deployment patterns
- statically
  - installable binary of the entire software
  - positive: fast execution time for the user; don't have to upload user data to server (user privacy); can be called when the user is offline; keeping the model operation is user's responsibility
  - negative: hard to upgrade model without upgrading whole app; may have messy computational requirements; difficult to monitor the model performance
- dynamically on the user's device
  - similar to static (user runs part of the system on their device), but the model is not a part of the binary code of the app
  - positive: better separation of concerns (easier to update); fast for the user (cheaper for the org's servers)
  - negative: varies depending on strategy; difficult to monitor the model performance
- dynamically on a server:
  - place the model on servers and make it available as REST API or gRPC service
- model streaming

## Deployment strategies
- single: simplest -> serialize new model to file, replace the old one
- silent: new and old version runs in parallel during the switch
- canary: pushes new version to a small fraction of users, while keep the old one running for most
- multi-armed bandit (MAB): way to compare one or more versions of the model in the production env, and select the best performing one

> "The model must be applied to the end-to-end and confidence test data by simulating a regular call from the outside"

## Algorithmic efficiency
- important consideration in model deployment
- you should only write your own code when it's absolutely necessary
- caching speeds up the application when it contains resource-consuming functions frequently called with the same parameter values

# Ch9. Model Serving, Monitoring, and Maintenance
