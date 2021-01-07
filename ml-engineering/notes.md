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