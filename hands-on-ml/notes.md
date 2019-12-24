# Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow
Author: Aurélien Geron

# Part I, The Fundamentals of Machine Learning

## CH1. The Machine Learning Landscape
Machine Learning is great for:
- Problems for which existing solutions require a lot of fine-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better than the traditional approach.
- Complex problems for which using a traditional approach yields no good solution: the best Machine Learning techniques can perhaps find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data.  

Broad categories of ML systems:
- Trained with human supervision? (supervised, unsupervised, semisupervised, and Reinforcement Learning)
- Can learn incrementally on the fly? (online versus batch learning)
- Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based versus model-based learning)

### Supervised/Unsupervised Learning
- Supervised: the training set you feed to the algorithm includes the desired solutions (labels). e.g.: k-NN, Linear Regression, Logistic Regression, SVM, Decision Trees, Random Forests, Neural networks
- Unsupervised: the training data is unlabeled. e.g.: K-Means, DBSCAN, HCA, One-class SVM, Isolation Forest, PCA, Kernel PCA, LLE, t-SNE, Apriori, Eclat

```
Tip: It's a good idea to reduce the dimension of your training data before feeding to another ML algorithm (e.g. supervised). Run faster, use less disk/memory, may perform better
```

- Semisupervised: data is partially labeled. e.g.: deep belief networks (DBNs) are based on unsupervised components called restricted Boltzmann machines (RBMs)
- Reinforcement: agent, rewards, penalties and policy

### Batch and Online Learning
- Batch: incapable of learning incrementally, must be trained using all the available data (offline learning). This process can be automated, but the training process can take many hours/resources
- Online: train incrementally by feeding data sequentially (individually or mini-batches). *Out-of-core*: loads part of the data, train, repeat until has run on all of the data (usually done offline, so *online*~*incremental*). Learning rate: high = rapid adaption, quickly forget old data; low = more inertia, learn slowly, less sensitive to noise.

### Instance-Based x Model-Based Learning
- Instance-based: learns the examples by heart, generalize by using similarity measures
- Model-based: build a model of the examples and use that model to make predictions

```
Data versus algorithms (2001 Microsoft researchers paper):

“these results suggest that we may want to reconsider the trade-off between spending time and money on algorithm development versus spending it on corpus development.”
```

### Nonrepresentative Training Data
It is crucial that your training data be representative of the new cases you want to generalize to
- Small sample: high chance of *sampling noise*
- Large sample: if sampling method is flawed = *sampling bias*

### Poor-Quality Data
Training data full of errors, outliers and noise (e.g. poor-quality measurements) -> often worth the effort of cleaning the training data.
- Outliers: discard or fix the errors manually may help
- Missing: ignore the attribute, the instances, fill the missing values, train one model with the feature and one without it

### Irrelevant Features
Feature engineering:
- Feature selection: most useful features
- Feature extraction: combining existing features to produce a more useful one (e.g. dimensionality reduction)
- Creating new features by gathering new data

### Overfitting the Training Data
The model performs well on the training data, but it does not generalize well
- noisy training set
- small training set (*sampling noise*)

```
Overfitting due to model being too complex relative to the amount and noise of the training data. Solutions:

- Simplify the model (fewer parameters), reduce number of attributes, constrain the model (*regularization*)
- Gather more training data
- Reduce noise (e.g. fix data errors, remove outliers)
```

### Underfitting the Training Data
Model is too simple to learn the underlying structure of the data. 
```
Solutions:

- Select a more powerful model, with more parameters
- Feed better features to the learning algorithm (*feature engineering*)
- Reduce the constraints of the model (e.g. reduce the regularization hyperparameter)
```

### Testing and Validating
Split your data into two sets: training and test. The error rate is called *generalization error* (*out-of-sample error*), by evaluating your model on the test set, you get an estimate of this error

Training error low, but generalization error high -> model is overfitting the training data

### Hyperparameter Tuning and Model Selection
**Holdout validation**: hold out part of the training set to evaluate several candidate models and select the best one. The held-out set is called *validation set* (or *development set*, or *dev set*). After the validation process, you train the best model on the full training set (including the validation set), and this gives you the final model. Lastly, you evaluate this final model on the test set to get an estimate of the generalization error

If the validation set is too small, model evaluations will be imprecise -> suboptimal model by mistake.

If the validation set is too large, the remaining training set will be much smaller than the full training set.

Solution: perform repeated *cross-validation*, using many validation sets. Each model is evaluated once per validation set after it is trained on the rest of the data. By averaging out all the evaluations of a model, you have a more accurate measure of its performance (but... more training time).

### Data Mismatch
The validation set and test set must be as representative as possible of the data you expect to use in production.

If this happens, hold out some training data in another set -> *train-dev* set. If after the model trained on the training set performs well on the *train-dev*, then the model is not overfitting. If it performs poorly on the validation set, it's probably a data mismatch problem. Conversely, if the model performs poorly on the train-dev set, it must have overfitted the training set.

```
NO FREE LUNCH THEOREM

If you make absolutely no assumption about the data, then there is no reason to prefer one model over any other. There is no model that is a priori guaranteed to work better

A model is a simplified version of the observations. The simplifications are meant to discard the superfluous details that are unlikely to generalize to new instances
```

## CH2. End-to-End Machine Learning Project

# Part II, Neural Networks and Deep Learning