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

> **Tip**: It's a good idea to reduce the dimension of your training data before feeding to another ML algorithm (e.g. supervised). Run faster, use less disk/memory, may perform better

- Semisupervised: data is partially labeled. e.g.: deep belief networks (DBNs) are based on unsupervised components called restricted Boltzmann machines (RBMs)
- Reinforcement: agent, rewards, penalties and policy

### Batch and Online Learning
- Batch: incapable of learning incrementally, must be trained using all the available data (offline learning). This process can be automated, but the training process can take many hours/resources
- Online: train incrementally by feeding data sequentially (individually or mini-batches). *Out-of-core*: loads part of the data, train, repeat until has run on all of the data (usually done offline, so *online*~*incremental*). Learning rate: high = rapid adaption, quickly forget old data; low = more inertia, learn slowly, less sensitive to noise.

### Instance-Based x Model-Based Learning
- Instance-based: learns the examples by heart, generalize by using similarity measures
- Model-based: build a model of the examples and use that model to make predictions

> **Data versus algorithms** (2001 Microsoft researchers paper):
> “these results suggest that we may want to reconsider the trade-off between spending time and money on algorithm development versus spending it on corpus development.”

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

> Overfitting due to model being too complex relative to the amount and noise of the training data. 
> Solutions:
> - Simplify the model (fewer parameters), reduce number of attributes, constrain the model (regularization)
> - Gather more training data
> - Reduce noise (e.g. fix data errors, remove outliers)

### Underfitting the Training Data
Model is too simple to learn the underlying structure of the data. 
> Solutions:
> - Select a more powerful model, with more parameters
> - Feed better features to the learning algorithm (feature engineering)
> - Reduce the constraints of the model (e.g. reduce the regularization hyperparameter)

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

> **NO FREE LUNCH THEOREM**:  If you make absolutely no assumption about the data, then there is no reason to prefer one model over any other. There is no model that is a priori guaranteed to work better A model is a simplified version of the observations.  The simplifications are meant to discard the superfluous details that are unlikely to generalize to new instances

## CH2. End-to-End Machine Learning Project

This chapter presents an example ml project:

1. **Look at the big picture**

#### Pull out your ML project checklist
#### Frame the problem: what exactly the business objective is. How does the company expect to use and benefit from this model?

> **PIPELINES**: Sequence of data processing components = data pipeline. Each component is fairly self-contained

- What the current solution looks like?
- Frame the problem: supervised? classification/regression? batch/online? etc
#### Select a Performance Measure: RMSE, MAE, accuracy, etc
#### Check the Assumptions (by you or others)

2. **Get the data**

#### Create an isolated environment
```bash
python3 -m pip install --user -U virtualenv
virtualenv my_env
source my_env/bin/activate
# deactivate
```
#### Data Structure
```python
# Pandas DataFrame methods
.head()
.info()
.describe()
.value_counts()
%matplotlib inline
.hist()
```
#### Create a Test Set (20% or less if the dataset is very large)
> **WARNING**: before you look at the data any further, you need to create a test set, put it aside, and never look at it -> avoid the *data snooping* bias
```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```
#### Option: Stratified sampling
```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True) 
```

3. Discover and visualize the data to gain insights

#### Put the test set aside and only explore the training set. If the training set is large, you may want to sample an exploration set
```python
# make a copy to avoid harming the training set
housing = strat_train_set.copy()
```
####  We are very good at spotting patterns in pictures
#### Look for Correlations
```python
corr_matrix = housing.corr()
# how each attribute correlates with one specific
corr_matrix["specific_attr"].sort_values(ascending=False)
# alternative
from pandas.plotting import scatter_matrix
scatter_matrix(housing[attributes], figsize=(12, 8))
```
> **WARNING**: Correlation coefficient only measures linear correlation, it may completely miss out on non-linear relationships!
#### Experimenting with Attribute Combinations
> Attributes with tail-heavy distribution? You may want to transform then (e.g. logarithm)

After engineering features, you may want to look at the correlations again to check if the features created are more correlated with the target.

This is an iterative process: get a prototype up and running, analyze its output, come back to this exploration step

4. **Prepare the data for ML algorithms**

Instead of doing this manually, you should **write functions** for this purpose: reproductibility, reuse in your live system, quickly try various transformations to see which combination works best

#### Data Cleaning
Missing values
```python
# pandas methods
.dropna()
.drop()
.fillna()
```
> **WARNING**: if you choose to fill the missing values, save the value you computed to fill the training set. You will need it later to replace missing values in the test set.
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
# the median can only be used on numerical attributes
imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_num_tr = pd.DataFrame(X, 
                    columns=housing_num.columns,
                    index=housing_num.index) 
```

#### Handling Text and Categorical Attributes
- Ordinal Encoder
```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# problem: ML algorithms will assume two nearby values are more similar than two distant values. That may not be the case
```

- One Hot Encoder
```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```
> **TIP**: attributes with large number of possible categories = large number of input features. You may want to replace the categorical input with useful numerical features related to it

#### Feature Scaling
With few exceptions, ML algorithms don't perform well when the input numerical attributes have very different scales

- Min-max scaling (*normalization*) -> values are shifted and rescaled so that they end up ranging from 0 to 1. (x_i - min_x) / (max_x - min_x). *MinMaxScaler* on Scikit-Learn
- Standardization (zero mean) -> (x_i - mean_x) / std_x. Doesn't bound values to a specific range, but is much less affected by outliers. **StandardScaler** on Scikit-Learn.

#### Transformation Pipelines
Scikit-Learn provides the *Pipeline* class to help with sequences of transformations
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num) 
```

- We have handled numerical and categorical columns separately, but Scikit-Learn has a single transformer able to handle all columns
```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
```

5. **Select a model and train it**
6. **Fine-tune your model**
7. **Present your solution**
8. **Launch, monitor, and maintain your system**


# Part II, Neural Networks and Deep Learning