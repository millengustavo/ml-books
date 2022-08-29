# Designing Machine Learning Systems
Author: Chip Huyen

# Ch1. Overview of Machine Learning Systems

> This book's goal is to provide you with a framework to develop a solution that best works for your problem, regardless of which algorithm you might end up using

## When to use ML
Before starting an ML project, you might want to ask whether ML is necessary or cost-effective

> Machine learning is an approach to learn complex pattern from existing data and use these patterns to make predictions on unseen data

The patterns your model learns from existing data are only useful if unseen data also share these patterns -> unseen data and training data should come from similar distributions

Most of today's ML algorithms shouldn't be used in any of the following conditions:
- It's unethical
- Simpler solutions do the trick (non-ML solutions)
- It's not cost-effective

## Understanding ML Systems

ML algorithms don't predict the future, but encode the past, thus perpetuating the biases in the data and more. When ML algorithms are deployed at scale, they can discriminate against people at scale. This can especially hurt members of minority groups because misclassification on them could only have a minor effect on models' overall performance metrics

In tradition SWE, you only need to focus on testing and versioning your code. With ML, we have to test and version our data too, and that's the hard part

