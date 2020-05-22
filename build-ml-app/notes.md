# Building Machine Learning Powered Applications: Going from Idea to Product

Author: Emmanuel Ameisen

![build-ml-app-cover](cover.jpg)

- [Building Machine Learning Powered Applications: Going from Idea to Product](#building-machine-learning-powered-applications-going-from-idea-to-product)
- [Part I. Find the Correct ML Approach](#part-i-find-the-correct-ml-approach)
- [Ch1. From Product Goal to ML Framing](#ch1-from-product-goal-to-ml-framing)
  - [Data availability scenarios](#data-availability-scenarios)
  - [The Simplest Approach: being the algorithm](#the-simplest-approach-being-the-algorithm)
  - [What to focus on in an ML project](#what-to-focus-on-in-an-ml-project)
  - [Which modeling techniques to use](#which-modeling-techniques-to-use)
- [Ch2. Create a Plan](#ch2-create-a-plan)
  - [Measuring Success](#measuring-success)
  - [Business Performance](#business-performance)
    - [Updating an app to make a modeling task easier](#updating-an-app-to-make-a-modeling-task-easier)
  - [Freshness and Distribution Shift](#freshness-and-distribution-shift)
  - [Leverage Domain Expertise](#leverage-domain-expertise)
    - [Examining the data](#examining-the-data)
  - [Stand on the Shoulders of giants](#stand-on-the-shoulders-of-giants)
  - [To make regular progress: start simple](#to-make-regular-progress-start-simple)
  - [Diagnose Problems](#diagnose-problems)
- [Part II. Build a Working Pipeline](#part-ii-build-a-working-pipeline)
- [Ch3. Build your first end-to-end pipeline](#ch3-build-your-first-end-to-end-pipeline)
  - [Test your workflow](#test-your-workflow)
  - [Finding the impact bottleneck](#finding-the-impact-bottleneck)
- [Ch4. Acquire an initial dataset](#ch4-acquire-an-initial-dataset)
  - [Iterate on datasets](#iterate-on-datasets)
  - [Data quality rubric](#data-quality-rubric)
    - [Format](#format)
    - [Quality](#quality)
    - [Quantity and distribution](#quantity-and-distribution)
    - [Summary statistics](#summary-statistics)
    - [Data leakage](#data-leakage)
  - [Clustering](#clustering)
  - [Let data inform features and models](#let-data-inform-features-and-models)
    - [Feature crosses](#feature-crosses)
    - [Giving your model the answer](#giving-your-model-the-answer)
  - [Robert Munro: how do you find, label and leverage data](#robert-munro-how-do-you-find-label-and-leverage-data)
    - [Uncertainty sampling](#uncertainty-sampling)
    - ["Error model"](#%22error-model%22)
    - ["Labeling model"](#%22labeling-model%22)
    - [Validation](#validation)
- [Part III. Iterate on Models](#part-iii-iterate-on-models)
- [Ch5. Train and evaluate your model](#ch5-train-and-evaluate-your-model)
  - [The simplest appropriate model](#the-simplest-appropriate-model)
    - [Simple model](#simple-model)
  - [Test set](#test-set)
  - [Data leakage](#data-leakage-1)
  - [Bias variance trade-off](#bias-variance-trade-off)
  - [Evaluate your model: look beyond accuracy](#evaluate-your-model-look-beyond-accuracy)
  - [Evaluate Feature Importance](#evaluate-feature-importance)
  - [Black-box explainers](#black-box-explainers)
- [Ch6. Debug your ML problems](#ch6-debug-your-ml-problems)
  - [Software Best Practices](#software-best-practices)
  - [Visualization steps](#visualization-steps)
  - [Separate your concerns](#separate-your-concerns)
  - [Test your ML code](#test-your-ml-code)
  - [Debug training: make your model learn](#debug-training-make-your-model-learn)
    - [Task difficulty](#task-difficulty)
  - [Debug generalization: make your model useful](#debug-generalization-make-your-model-useful)
- [Ch7. Using classifiers for writing recommendations](#ch7-using-classifiers-for-writing-recommendations)
- [Part IV. Deploy and Monitor](#part-iv-deploy-and-monitor)
- [Ch8. Considerations when deploying models](#ch8-considerations-when-deploying-models)
  - [Data Concerns](#data-concerns)
    - [Data ownership](#data-ownership)
    - [Data bias](#data-bias)
      - [Test sets](#test-sets)
  - [Modeling Concerns](#modeling-concerns)
    - [Feedback loops](#feedback-loops)
    - [Inclusive model performance](#inclusive-model-performance)
    - [Adversaries](#adversaries)
  - [Chris Harland: Shipping Experiments](#chris-harland-shipping-experiments)
- [Ch9. Choose Your Deployment Option](#ch9-choose-your-deployment-option)
  - [Server-side deployment](#server-side-deployment)
    - [Streaming API workflow](#streaming-api-workflow)
    - [Batch Predictions](#batch-predictions)
    - [Hybrid Approach](#hybrid-approach)
  - [Client-side deployment](#client-side-deployment)
  - [Browser side](#browser-side)
  - [Federated Learning: a hybrid apporach](#federated-learning-a-hybrid-apporach)
- [Ch10. Build Safeguards for Models](#ch10-build-safeguards-for-models)
  - [Check inputs](#check-inputs)
  - [Model outputs](#model-outputs)
  - [Model failure fallbacks](#model-failure-fallbacks)
    - [Filtering model](#filtering-model)
  - [Engineer for Performance](#engineer-for-performance)
    - [Scale to multiple users](#scale-to-multiple-users)
    - [Caching fo ML](#caching-fo-ml)
      - [Caching inference results](#caching-inference-results)
      - [Caching by indexing](#caching-by-indexing)
  - [Model and data life cycle management](#model-and-data-life-cycle-management)
    - [Reproducibility](#reproducibility)
    - [Resilience](#resilience)
  - [Data Processing and DAGs](#data-processing-and-dags)
  - [Ask for feedback](#ask-for-feedback)
  - [Chris Moody: Empowering Data Scientist to Deploy Models](#chris-moody-empowering-data-scientist-to-deploy-models)
- [Ch11. Monitor and update models](#ch11-monitor-and-update-models)
  - [Monitoring saves lives](#monitoring-saves-lives)
    - [Monitor to inform refresh rate](#monitor-to-inform-refresh-rate)
    - [Monitor to detect abuse](#monitor-to-detect-abuse)
  - [Choose what to monitor](#choose-what-to-monitor)
    - [Performance Metrics](#performance-metrics)
    - [Business metrics](#business-metrics)
  - [CI/CD for ML](#cicd-for-ml)
  - [A/B Testing and Experimentation](#ab-testing-and-experimentation)
    - [Choosing groups and duration](#choosing-groups-and-duration)
    - [Estimating the better variant](#estimating-the-better-variant)
    - [Building the infrastructure](#building-the-infrastructure)
  - [Other approaches](#other-approaches)


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

## Software Best Practices
**KISS principle**: building only what you need

> Most software applications: strong test coverage = high confidence app is functioning well. ML pipelines can pass many tests, but still give entirely incorrect results. Doesn't have just to run, it should produce accurate predictive outputs

Progressive approach, validate:
1. Data flow
2. Learning capacity
3. Generalization and inference

Make sure your pipeline works for a few examples, then write tests to make sure it keeps functioning as you make changes

## Visualization steps
Inspect changes at regular intervals

- **Data loading**: Verify data is formatted correctly
- **Cleaning and feature selection**: remove any unnecessary information
- **Feature generation**: check that the feature values are populated and that the values seem reasonable
- **Data formatting**: shapes, vectors
- **Model output**: first look if the predictions are the right type or shape, then check if the model is actually leveraging the input data

## Separate your concerns
Modular organization: separate each function so that you can check that it individually works before looking at the broader pipeline. Once broken down, you'll be able to write tests

## Test your ML code
[Source code on book's Github repository](https://github.com/hundredblocks/ml-powered-applications/tree/master/tests)

- Test data ingestion
- Test data processing
- Test model outputs

## Debug training: make your model learn
Contextualize model performance: generate an estimate of what an acceptable error for the taks is by labeling a few examples yourself

### Task difficulty
- **The quantity and diversity of data you have**: more diverse/complex the problem = more data for the model to learn from it
- **How predictive the features are**: make the data more expressive to help the model learn better
- **The complexity of your model**: simplest model is good to quickly iterate, but some tasks are entirely out of reach of some models

## Debug generalization: make your model useful
- **Data Leakage**: if you are surprised by validation performance, inspect the features; fixing a leakage issue will lead to lower validation performance, but a better model
- **Overfitting**: model performs drastically better on the training set than on the test set. Add regularization or data augmentation
- **Dataset redesign**: use k-fold cross validation to alleviate concerns that data splits may be of unequal quality

> “If your models aren’t generalizing, your task may be too hard. There may not be enough information in your training examples to learn meaningful features that will be informative for future data points. If that is the case, then the problem you have is not well suited for ML”

# Ch7. Using classifiers for writing recommendations

# Part IV. Deploy and Monitor
Production ML pipelines need to be able to detect data and model failures and handle them with grace -> **proactively**

# Ch8. Considerations when deploying models
- How was the data you are using collected?
- What assumptions is your model making by learning from this dataset?
- Is this dataset representative enough to produce a useful model?
- How could the results of your work be misused?
- What is the intended use and scope of your model?

## Data Concerns

### Data ownership
- Collection
- Usage and permission
- Storage

### Data bias
Datasets: results of specific data collection decisions -> lead to datasets presenting a biased view of the world. ML models learn from datasets -> will reproduce these biases

- Measurement errors or corrupted data
- Representation
- Access

#### Test sets
Build a test set that is inclusive, representative, and realistic -> proxy for performance in production -> improve the chances that every user has an equally positive experience

> Models are trained on historical data -> state of the world in the past. Bias most often affects populations that are already disenfranchised. Working to eliminate bias -> help make systems fairer for the people who need it most

## Modeling Concerns

### Feedback loops
User follow a model's recommendation -> future models make the same recommendation -> models enter a self-reinforcing feedback loop

To limit negative effects of feedback loops -> choose a label that is less prone to creating such a loop

### Inclusive model performance
Look for performance on a segment of the data, instead of only comparing aggregate performance

### Adversaries
Regularly update models

Some types of attacks:
- Fool models into a wrong prediction (most common)
- Use a trained model to learn about the data it was trained on

## Chris Harland: Shipping Experiments
> When giving advice, the cost of being wrong is very high, so precision is the most useful

# Ch9. Choose Your Deployment Option

## Server-side deployment
Setting up a web server that can accept requests from clients, run them through an inference pipeline, and return the results. The servers represents a central failure point for the application and can be costly if the product becomes popular

### Streaming API workflow
**Endpoint approach**

- Quick to implement
- Requires infrastructure to scale linearly with the current number of users (1 user = 1 separate inference call)
- Required when strong latency constraints exist (info the model needs is available only at prediction time and model's prediction is required immediately)

### Batch Predictions
Inference pipeline as a job that can be run on multiple examples at once. Store predictions so they can be used when needed

- Appropriate when you have access to the features need for a model before the model's prediction is required
- Easier to allocate and parallelize resources
- Faster at inference time since results have been precomputed and only need to be retrieved (similar gains to caching)

### Hybrid Approach
- Precompute as many cases as possible
- At inference either retrieve precomputed results or compute them on the spot if they are not available or are outdated
- Have to maintain both a batch and streaming pipeline (more complexity of the system)

## Client-side deployment
Run all computations on the client, eliminating the need for a server to run models. Models are still trained in the same manner and are sent to the device for inference

- Reduces the need to build infra
- Reduces the quantity of data that needs to be transferred between the device and the server
  - Reduces network latency (app may even run without internet)
  - Removes the need for sensitive information to be transferred to a remote server

> If the time it would take to run inference on device is larger than the time it would take to transmit data to the server to be processed, consider running your model in the cloud

On-device deployment is only worthwhile if the latency, infrastructure, and privacy benefits are valuable enough to invest the engineering effort (simplifying a model)

## Browser side
Some libraries use browsers to have the client perform ML tasks

`Tensorflow.js`: train and run inference in JavaScript in the browser for most differentiable models, even trained in different languages such as Python

## Federated Learning: a hybrid apporach
Each client has their own model. Each model learns from their user's data and send aggregated (and potentially anonymized) updates to the server. The server leverages all updates to improve its model and distills this new model back to individual clients. Each user receives a model personalized to their needs, while still benefiting from aggregate information about other users

# Ch10. Build Safeguards for Models
No matter how good a model is, it will fail on some examples -> engineer a system that can gracefully handle such features

## Check inputs
- Very different data from train
- Some features missing
- Unexpected types

Input checks are part of the pipeline -> change the control flow of a program based on the quality of inputs

## Model outputs
Prediction falls outside an acceptable range -> consider not displaying it

> Acceptable outcome: not only if the outcome is plausible -> also depends if the outcome would be **useful for the user**

## Model failure fallbacks
Flag cases that are too hard and encourage user to provide an easier input (e.g. well-lit photo)

Detecting errors:
- Track the confidence of a model
- Build an additional model tasked with detecting examples a main model is likely to fail on

### Filtering model
- ML version of input tests
- Binary classifier
- Estimate how well a model will perform on an example without running the model on it
- Decrease the likelihood of poor results and improve resource usage
- Catch:
  - qualitatively different inputs
  - inputs the model struggled
  - adversarial inputs meant to fool the model
- Minimum criteria:
  - should be fast (reduce the computational burden)
  - should be good at eliminating hard cases

> The faster your filtering model is, the less effective it needs to be

## Engineer for Performance
### Scale to multiple users
ML is horizontally scalable = more servers = keep response time reasonable when the number of requests increases

### Caching fo ML
Storing results to function calls -> future calls with same parameters simply retrieve the stored results

#### Caching inference results
**Least recently used (LRU) cache**: keep track the most recent inputs to a model and their corresponding outputs

- not appropriate if each input is unique
- `functools` Python module proposes a default implementation of an LRU cache that you can use with a simple decorator

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def run_model(data):
  # Insert slow model inference below
  pass
```

#### Caching by indexing
Cache other aspects of the pipeline that can be precomputed. Easy if a model does not only rely on user inputs

> “Caching can improve performance, but it adds a layer of complexity. The size of the cache becomes an additional hyperparameter to tune depending on your application’s workload. In addition, any time a model or the underlying data is updated, the cache needs to be cleared in order to prevent it from serving outdated results"

## Model and data life cycle management
ML application:

- produces reproducible results
- is resilient to model updates
- is flexible enough to handle significant modelling and data processing changes

### Reproducibility
Each model/dataset pair should be assigned an unique identifier -> should be logged each time a model is used in production

### Resilience
- production pipeline should aim to update models without significant downtime
- if a new model performs poorly, we'd like to be able to roll back to the previous one

## Data Processing and DAGs
Directed acyclic graph (DAG): can be used to represent our process of going from raw data to trained model -> each node represent a processing step and each step represent a dependency between two nodes

DAGs helps systematize, debug, and version a pipeline -> can become a crucial time saver

## Ask for feedback
- Explicity asking for feedback (display model's prediction accompanying it with a way for users to judge and correct a prediction)
- Measuring implicit signals

User feedback is a good source of training data and can be the first way to notice a degradation in performance

## Chris Moody: Empowering Data Scientist to Deploy Models
- Make humans and algorithms work together: spend time thinking about the right way to present information
- Canary development -> start deploying the new version to one instance and progressively update instances while monitoring performance

> "Ownership of the entire pipeline leads individuals to optimize for impact and reliability, rather than model complexity"

# Ch11. Monitor and update models
## Monitoring saves lives
Monitoring: track the health of a system. For models: performance and quality of their predictions

### Monitor to inform refresh rate
Detect when a model is not fresh anymore and needs to be retrained. Retraining events happen when accuracy dips below a threshold.

### Monitor to detect abuse
Anomaly detection to detect attacks

## Choose what to monitor
Commonly monitor metrics such as the average time it takes to process a request, the proportion of requests that fail to be processed, and the amount of available resources

### Performance Metrics
- Track changes in the input distribution (*feature drift*)
- Monitor the input distribution (summary statistics)
- Monitor distribution shifts

> **Conterfactual evaluation**: aims to evaluate what would have happened if we hadn't actioned a model -> Not acting on a random subset of examples allow us to observe an unbiased distribution of the positive class. Comparing model predictions to true outcomes for the random data, we can begin to estimate a model's precision and recall

### Business metrics
Product metrics should be closely monitored

## CI/CD for ML
- **CI**: letting multiple developers regularly merge their code back into a central codebase
- **CD**: improving the speed at which new versions of software can be released

CI/CD for ML: make it easier to deploy new models or update existing ones

> "Releasing updates quickly is easy; the challenge comes in guaranteeing their quality (...) There is no substitute for live performance to judge the quality of a model"

**Shadow mode**: deploying a new model in parallel to an existing one. When running inference, both models' predictions are computed and stored, but the application only uses the prediction of the existing model

- estimate a new models' performance in a production environment without changing the user experience
- test the infrastructure required to run inference for a new model (may be more complex)
- but can't observe the user's response to the new model

## A/B Testing and Experimentation
Goal: maximize chances of using the best model, while minimizing the cost of trying out suboptimal models

Expose a sample of users to a new model, and the rest to another. Larger control group (current model) and a smaller treatment group (new version we want to test). Run for a sufficient amount of time -> compare the results for both groups and choose the better model

### Choosing groups and duration
- Users in both groups should be as similar as possible -> any difference in outcome = our model and not difference in cohorts
- Treatment group should be:
  - large enough: statistically meaningful conclusion
  - small as possible: limit exposure to a potentially worse model
- Duration of the test:
  - too short: not enough information
  - too long: risk losing users

### Estimating the better variant
Decide on the size of each group and the length of the experiment before running it

### Building the infrastructure
Branching logic: decides which model to run depending on a given field's value (harder if a model is accessible to logged-out users)

## Other approaches
**Multiarmed bandits**: more flexible approach, can test variants continually and on more than two alternatives. Dynamically update which model to serve based on how well each option in performing

**Contextual multiarmed bandits**: go even further, by learning which model is a better option for each particular user

> "The majority of work involved with building ML products consists of data and engineering work"


<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>