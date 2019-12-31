# Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow
Author: Aurélien Geron

<img src="https://images-na.ssl-images-amazon.com/images/I/51aqYc1QyrL._SX379_BO1,204,203,200_.jpg" title="book" width="150" />

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

> **Tip**: by default, the remaining columns will be dropped. To avoid this, you can specify "passthrough"

5. **Select a model and train it**

#### Training and Evaluating on the Training Set
```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
```

#### Better Evaluation Using Cross-Validation
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores) 
# cross_val_score expects a utility fn (greater is better)
# rather than a cost fn (lower is better)
```

Try out many other models from various categories, without spending too much time tweaking the hyperparameters. The goal is to shortlist a few (two to five) promising models.

> **Tip**: save the models you experiment
```python
import joblib

joblib.dump(my_model, "my_model.pkl")
# and later
my_model_loaded = joblib.load("my_model.pkl")
```

6. **Fine-tune your model**
#### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

# and after
grid_search.best_params_
grid_search.best_estimator_
```

#### Randomized Search
When the hyperparameter search space is large, it is often preferable to use *RandomizedSearchCV*

#### Ensemble Methods
Another way to fine-tune your system is to try to combine the models that perform best. The group (*ensemble*) will often perform better than the best individual model

#### Analyze the Best Models and Their Errors

```python
feature_importances = grid_search.best_estimator_.feature_importances_
>>> extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
>>> cat_encoder = full_pipeline.named_transformers_["cat"]
>>> cat_one_hot_attribs = list(cat_encoder.categories_[0])
>>> attributes = num_attribs + extra_attribs + cat_one_hot_attribs
>>> sorted(zip(feature_importances, attributes), reverse=True)
[(0.3661589806181342, 'median_income'),
 (0.1647809935615905, 'INLAND'),
 (0.10879295677551573, 'pop_per_hhold'),
 (0.07334423551601242, 'longitude'),
 (0.0629090704826203, 'latitude'),
 (0.05641917918195401, 'rooms_per_hhold'),
 (0.05335107734767581, 'bedrooms_per_room'),
 (0.041143798478729635, 'housing_median_age'),
 (0.014874280890402767, 'population'),
 (0.014672685420543237, 'total_rooms'),
 (0.014257599323407807, 'households'),
 (0.014106483453584102, 'total_bedrooms'),
 (0.010311488326303787, '<1H OCEAN'),
 (0.002856474637320158, 'NEAR OCEAN'),
 (0.00196041559947807, 'NEAR BAY'),
 (6.028038672736599e-05, 'ISLAND')]
```
You may want to try dropping less useful features or understand errors your system makes

#### Evaluate Your System on the Test Set
Get the predictors and the labels from your test set, run your *full_pipeline* to transform the data (call transform(), not fit_transform()), and evaluate the final model on the test set:

```python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)   # => evaluates to 47,730.2
```

> Computing a 95% confidence interval for the generalization
```python
>>> from scipy import stats
>>> confidence = 0.95
>>> squared_errors = (final_predictions - y_test) ** 2
>>> np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
...                          loc=squared_errors.mean(),
...                          scale=stats.sem(squared_errors)))
...
array([45685.10470776, 49691.25001878])
``` 

If you did a lot of hyperparameter tuning, the performance may be worse than your cv (slightly overfit to the training set). Resist the temptation to tweak the hyperparameters to make the numbers look good on the test set

7. **Present your solution**

#### Project prelaunch phase
- What you have learned, what worked/did not, what assumptions were made, what your system's limitations are
- Document everything
- Create nice presentations with clear visualizations and easy-to-remember statements

8. **Launch, monitor, and maintain your system**
Get your solution ready for production (e.g., polish the code, write documentation and tests, and so on). 

Then you can deploy your model to your production environment. 

One way to do this is to save the trained Scikit-Learn model (e.g., using joblib), including the full preprocessing and prediction pipeline, then load this trained model within your production environment and use it to make predictions by calling its predict() method.

But deployment is not the end of the story. You also need to write monitoring code to check your system’s live performance at regular intervals and trigger alerts when it drops

If the data keeps evolving, you will need to update your datasets and retrain your model regularly. You should probably automate the whole process as much as possible. Here are a few things you can automate:

- Collect fresh data regularly and label it (e.g., using human raters).
- Write a script to train the model and fine-tune the hyperparameters automatically. This script could run automatically, for example every day or every week, depending on your needs.
- Write another script that will evaluate both the new model and the previous model on the updated test set, and deploy the model to production if the performance has not decreased (if it did, make sure you investigate why) 
- Evaluate the model's input data quality (missing features, stdev drifts too far from the training set, new categories)
- Keep backups of every model you create and have the process and tools in place to roll back to a previous model quickly
- Keep backups of every version of the datasets too

## CH3. Classification
> Some learning algorithms are sensitive to the order of the training instances, and they perform poorly if they get many similar instances in a row. **Shuffling** the dataset ensures that this won't happen

#### Stochastic Gradient Descent (SGD) classifier
*SGDClassifier* on sklearn. Has the advantage of being capable of handling very large datasets efficiently. Deals with training instances independently, one at a time (suited for online learning)

### Performance Measures
#### Measuring Accuracy using Cross-Validation
The snippet below does roughly the same thing as *cross_val_score()* from sklearn, but with stratified sampling
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))  # prints 0.9502, 0.96565, and 0.96495”
```
> High accuracy can be deceiving if you are dealing with *skewed datasets* (i.e., when some classes are much more frequent than others)

#### Confusion Matrix
Count the number of times instances of class A are classified as class B

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
```

Each row represents an *actual class*, while each column represents a *predicted class*

#### Precision
Accuracy of the positive predictions

> precision = TP/(TP + FP)

TP is the number of true positives, and FP is the number of false positives

#### Recall or Sensitivity, or True Positive Rate (TPR)
Ratio of positive instances that are correctly detected by the classifier

> recall = TP/(TP + FN)

FN is the number of false negatives

```python
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
```

#### F1 Score
Harmonic mean of the precision and recall. The harmonic mean gives much more weight to low values, so the classifier will only get a high F1 score if both recall and precision are high

> F1 = 2*(precision*recall)/(precision+recall)

```python
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
```

The F1 score favors classifiers that have similar precision and recall

#### Precision/Recall trade-off
> *Precision/recall trade-off*: increasing precision reduces recall, and vice versa. e.g., videos safe for kids: prefer reject many good videos (low recall), but keeps only safe ones (high precision)

Scikit-Learn gives you access to the decision scores that it uses to make predicitions, *.decision_function()* method, which returns a score for each instance and then use any threshold you want to make predictions based on those scores

For RandomForestClassifier for example, the method to use is *.predict_proba()*, which returns an array conatining a row per instance and a column per class, each containing the probability that the given instance belongs to the given class.

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

You can plot precision_recall_vs_threshold and choose a good threshold for your project, or plot precision directly against recall (generally you select a precision/recall just before the drop in the plot)

```python
# define threshold
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
# make predictions
y_train_pred_90 = (y_scores >= threshold_90_precision)
# check results
>>> precision_score(y_train_5, y_train_pred_90)
0.9000380083618396
>>> recall_score(y_train_5, y_train_pred_90)
0.4368197749492714
```

#### The ROC Curve
*Receiver operating characteristic* (ROC) curve. Plots the *true positive rate* (recall) against the *false positive rate* (FPR). The FPR is the ratio of negative instances that are incorrectly classified as positive. It is equal to 1 - *true negative rate* (TNR, or *specificity*) which is the ratio of negative instances that are correctly classified as negative

ROC curve plots sensitivity (recall) versus 1 - specificity (*.roc_curve()*)

> The higher the recall (TPR), the more false positives (FPR) the classifier produces. The purely random classifier is the diagonal line in the plot, a good classifier stays as far away from that line as possible (toward the top-left corner)

#### Area under the curve (AUC)
A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5 (*.roc_auc_score()*)

> You should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives. Otherwise, use the ROC curve. 

#### Binary classifiers
1. Choose the appropriate metrics
2. Evaluate your classifiers using cross-validation
3. Select the precision/recall trade-off that fits your needs
4. Use ROC curves and ROC AUC scores to compare various models

### Multiclass Classification
Some algorithms are not capable of handling multiple classes natively (e.g., Logistic Regression, SVM). For 10 classes you would train 10 binary classifiers and select the class whose classifier outputs the highest score. This is the *one-versus-the-rest* (OvR) strategy (also called *one-versus-all*)

*One-versus-one* (OvO) strategy: trains a binary classifier for every pair of digits. (N*(N-1))/2)classifiers! Good strategy for SVM that scales poorly with the size of the training set

> Scikit-Learn detects when you try to use a binary classification algorithm for a multiclass classification task, and it automatically runs OvR or OvO, depending on the algorithm

#### Error Analysis
Analyzing the confusion matrix often gives you insights into ways to improve your classifier.

### Multilabel Classification
Outputs multiple binary tags e.g., face recognition with Alice, Bob and Charlie; only Alice and Charlie in a picture -> output [1, 0, 1]

> Evaluate a multilabel classifier: One approach is to measure the F1 score for each individual label, then simply compute the average score

### Multioutput Classification (multioutput-multiclass classification)
Generalization of multilabel classification where each label can be multiclass (i.e., it can have more than two possible values)

## CH4. Training Models

### Linear Regression
A linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the bias term (also called the intercept term)

A *closed-form solution*, a mathematical equation that gives the result directly

### Gradient Descent
Generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.

It measures the local gradient of the error function with regard to the parameter vector θ, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum!

The size of the steps, is determined by the **learning rate** hyperparameter. If the learning rate is too small, then the algorithm will have to go through many iterations to converge, which will take a long time. If the learning rate is too high, you might jump across the valley and end up on the other side, possibly even higher up than you were before. This might make the algorithm diverge, with larger and larger values, failing to find a good solution

> The MSE cost function for a Linear Regression is a *convex function*: if you pick any two points on the curve, the line segment joining them never crosses the curve. This implies that there are no local minima, just one global minimum. It is also a continuous function with a slope that never changes abruptly. Consequence: Gradient Descent is guaranteed to approach arbitrarily close the global minimum (if you wait long enough and if the learning rate is not too high).

> When using Gradient Descent, you should ensure that all features have a similar scale, or else it will take much longer to converge.

#### Batch vs Stochastic Gradient Descent
The main problem with Batch Gradient Descent is that it uses the whole training set to compute the gradients at every step -> very slow when training set is large 

Stochastic Gradient Descent picks a random instance in the training set at every step and computes the gradients based only on that single instance -> algorithm much faster because it has very little data to manipulate at every iteration. Possible to train on huge training sets, since only one instance in memory at each iteration

> Randomness is good to escape from local optima, but bad because it means that the algorithm can never settle at the minimum -> solution to this dilemma is to gradually reduce the learning rate

### Learning Curves
Learning curves typical of a model that’s underfitting: Both curves have reached a plateau; they are close and fairly high.

> If your model is underfitting the training data, adding more training examples will not help. You need to use a more complex model or come up with better features

If there is a gap between the curves. This means that the model performs significantly better on the training data than on the validation data, which is the hallmark of an overfitting model. If you used a much larger training set, however, the two curves would continue to get closer -> feed more training data until the validation error reaches the training error

### Bias/Variance Trade-off
- **Bias**: Error due to wrong assumptions. A high-bias model is most likely to underfit the training data
- **Variance**: Error due to model's excessive sensitivity to small variations in the training data. Model with many degrees of freedom is likely to have high variance and thus overfit the training data
- **Irreducible error**: due to the noiseness of the data itself. The only way to reduce this part of the error is to clean up the data

> Increasing a model’s complexity will typically increase its variance and reduce its bias. Conversely, reducing a model’s complexity increases its bias and reduces its variance. This is why it is called a trade-off.

### Ridge Regression (Tikhonov regularization) - L2
Keep the models weights as small as possible. It is important to scale the data before performing Ridge Regression, as it is sensitive to the scale of the input features. This is true of most regularized models.

Note that the regularization term should only be added to the cost function during training. Once the model is trained, you want to use the unregularized performance measure to evaluate the model’s performance.

> It is quite common for the cost function used during training to be different from the performance measure used for testing. Apart from regularization, another reason they might be different is that a good training cost function should have optimization-friendly derivatives, while the performance measure used for testing should be as close as possible to the final objective.

### Lasso Regression - L1
*Least Absolute Shrinkage and Selection Operator Regression*. Tends to eliminate the weights of the least important features

### Elastic Net
Regularization term is a simple mix of both Ridge and Lasso's regularization terms

> **When should you use plain Linear Regression (i.e., without any regularization), Ridge, Lasso, or Elastic Net?**
> It is almost always preferable to have at least a little bit of regularization, so generally you should avoid plain Linear Regression. Ridge is a good default, but if you suspect that only a few features are useful, you should prefer Lasso or Elastic Net because they tend to reduce the useless features’ weights down to zero. In general, Elastic Net is preferred over Lasso because Lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated

### Early Stopping
Another way to regularize iterative learning algorithms. Stop training as soon as the validation error reaches a minimum. "Beautiful free lunch", Geoffrey Hinton

### Logistic Regression (Logit Regression)
Estimate the **probability** that an instance belongs to a particular class. Greater than 50% -> positive class, else negative class (binary classifier)

Logistic Regression cost function = log loss

> logit(p) = ln(p/(1-p)) -> also called log-odds

### Softmax Regression (Multinomial Logistic Regression)
Computes a score for each class, then estimates the probability of each class by applying the *softmax function* (*normalized exponential*) to the scores

> Cross entropy -> frequently used to measure how well a set of estimated class probabilities matches the target classes (when k=2 -> equivalent to log loss)

## CH5. Support Vector Machines

Soon...

## CH6. Decision Trees

```python
from sklearn.tree import export_graphviz 

 export_graphviz ( 
         tree_clf , 
         out_file = image_path ( "iris_tree.dot" ), 
         feature_names = iris . feature_names [ 2 :], 
         class_names = iris . target_names , 
         rounded = True , 
         filled = True 
     )
```

One of the many qualities of Decision Trees is that they require very little data preparation. In fact, they don’t require feature scaling or centering at all.

Scikit-Learn uses the  CART algorithm, which produces only  binary trees : nonleaf nodes always have two children (i.e., questions only have yes/no answers). However, other algorithms such as ID3 can produce Decision Trees with nodes that have more than two  children.

### White/Black box models

Decision Trees are intuitive, and their decisions are easy to interpret. Such models are often  called  white box models. In contrast, as we will see, Random Forests or neural networks are generally considered  black box models

> The CART algorithm is a  **greedy algorithm**: it greedily searches for an optimum split at the top level, then repeats the process at each subsequent level. It does not check whether or not the split will lead to the lowest possible impurity several levels down. A greedy algorithm often produces a solution that’s reasonably good but not guaranteed to be optimal.

Making predictions requires traversing the Decision Tree from the root to a leaf. Decision Trees generally are approximately balanced, so traversing the Decision Tree requires going through roughly  O (log 2 ( m )) nodes. 3  Since each node only requires checking the value of one feature, the overall prediction complexity is  O (log 2 ( m )), independent of the number of features. So **predictions are very fast, even when dealing with large training sets**.

Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees

**Nonparametric model**, not because it does not have any parameters but because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data. In contrast, a  parametric model, such as a linear model, has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting (but increasing the risk of underfitting).

> Increasing  min_*  hyperparameters or reducing  max_*  hyperparameters will regularize the model.

### Pruning

Standard statistical tests, such as the  χ 2   test  (chi-squared test), are used  to estimate the probability that the improvement is purely the result of chance (which is called the  null hypothesis ). If this probability, called the  p-value , is higher than a given threshold (typically 5%, controlled by a hyperparameter), then the node is considered unnecessary and its children are deleted

Decision Trees love orthogonal decision boundaries (all splits are perpendicular to an axis), which  makes them sensitive to training set rotation.

One way to limit this problem is to use Principal Component Analysis, which often results in a better orientation of the training data.

### Problems

The main issue with Decision Trees is that they are very sensitive to small variations in the training data.

Random Forests can limit this instability by averaging predictions over many trees


## CH7. Ensemble Learning and Random Forests 

*Wisdom of the crowd*: aggregated answer is better than an expert’s answer.

> A  group of predictors is called an  ensemble ; thus, this technique is called  Ensemble Learning , and an Ensemble Learning algorithm  is called an  Ensemble method

### Hard voting

Train a group of Decision Tree classifiers, each on a different random subset of the training set. To make predictions, you obtain the predictions of all the individual trees, then predict the class that gets the **most votes**

Very simple way to create an even better classifier is to aggregate the predictions of each classifier and predict the class that gets the most votes. This  **majority-vote classifier is called a  hard voting  classifier**

> Even if each classifier is a  weak learner  (meaning it does only slightly better than random guessing), the ensemble can still be a  strong learner  (achieving high accuracy), provided there are a sufficient number of weak learners and they are sufficiently diverse.

### Independent classifiers

Ensemble methods  work best when the predictors are as independent from one another as possible. One way to get diverse classifiers is to train them using very different algorithms. This increases the chance that they will make very different types of errors, improving the ensemble’s accuracy.

```python
from sklearn.ensemble import VotingClassifier 
```

### Soft voting

If  all classifiers are able to estimate class probabilities (i.e., they all have a  predict_proba()  method), then you can tell Scikit-Learn to predict the class with the highest class probability, averaged over all the individual classifiers. This is called  **soft voting**. It often achieves higher performance than hard voting because it gives more weight to highly confident votes. All you need to do is replace  voting="hard"  with  voting="soft"  and ensure that all classifiers can estimate class probabilities

Generally, the net result is that the ensemble has a similar bias but a lower variance than a single predictor trained on the original training set.

### Bagging and Pasting

Scikit-Learn  offers a simple API for both bagging and pasting with the  BaggingClassifier  class (or  BaggingRegressor  for regression).

### Bootstrapping

Bootstrapping introduces a bit more diversity in the subsets that each predictor is trained on, so bagging ends up with a slightly higher bias than pasting; but the extra diversity also means that the predictors end up being less correlated, so the ensemble’s variance is reduced. Overall, bagging often results in better models, which explains why it is generally preferred

### Out-of-bag evaluation

In Scikit-Learn, you can set  oob_score=True  when creating a  BaggingClassifier  to request an automatic oob evaluation after training

### Random Patches and Random Subspaces

Sampling both training instances and features is called the **Random Patches** method. Keeping all training instances (by setting bootstrap=False and max_samples=1.0) but sampling features (by setting bootstrap_features to True and/or max_features to a value smaller than 1.0) is called the **Random Subspaces** method. Sampling features results in even more predictor diversity, trading a bit more bias for a lower variance.

> Instead of building a  BaggingClassifier  and passing it a  DecisionTreeClassifier , you can instead use the  RandomForestClassifier  class, which is more convenient and optimized for Decision Trees

Forest of such extremely random trees is called an  *Extremely Randomized Trees*  ensemble  (*Extra-Trees*). This technique trades more bias for a lower variance. Much faster to train.

> It is hard to tell in advance whether a  RandomForestClassifier  will perform better or worse than an  ExtraTreesClassifier . Generally, the only way to know is to try both and compare them using cross-validation (tuning the hyperparameters using grid search).

Looking at how much the tree nodes that use that feature reduce impurity on average (across all trees in the forest)

### Boosting

**Boosting**  (originally called  hypothesis boosting ) refers  to any Ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor


### AdaBoost

One  way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. This results in new predictors focusing more and more on the hard cases. This is the technique used by  **AdaBoost** .

> This sequential learning technique has some similarities with Gradient Descent, except that instead of tweaking a single predictor’s parameters to minimize a cost function, AdaBoost adds predictors to the ensemble, gradually making it better.

There is one important drawback to this sequential learning technique: it cannot be parallelized (or only partially), since each predictor can only be trained after the previous predictor has been trained and evaluated. As a result, it *does not scale as well as bagging or pasting*.

### Gradient Boosting

Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor  to the  residual errors  made by the previous predictor.

The  learning_rate  hyperparameter scales the contribution of each tree. If you set it to a low value, such as  0.1 , you will need more trees in the ensemble to fit the training set, but the predictions will usually generalize better. This  is a regularization technique called  shrinkage .

If  subsample=0.25 , then each tree is trained on 25% of the training instances, selected randomly. As you can probably guess by now, this technique trades a higher bias for a lower variance. It also speeds up training considerably. This  is called  Stochastic Gradient Boosting .

### XGBoost

Extreme Gradient Boosting. This package was initially developed by Tianqi Chen as part of the Distributed (Deep) Machine Learning Community (DMLC), and it aims to be extremely fast, scalable, and portable. In fact, XGBoost is often an important component of the winning entries in ML competitions.

### Stacking

Stacking  (short for  stacked generalization ). 18  It is based on a simple idea: instead of using trivial functions (such as hard voting) to aggregate the predictions of all predictors in an ensemble, why don’t we train a model to perform this aggregation?

The final predictor  (called a  blender , or a  meta learner )

> To train the blender, a common approach is to use a hold-out set

It is actually possible to train several different blenders this way (e.g., one using Linear Regression, another using Random Forest Regression), to get a whole layer of blenders. The trick is to split the training set into three subsets: the first one is used to train the first layer, the second one is used to create the training set used to train the second layer (using predictions made by the predictors of the first layer), and the third one is used to create the training set to train the third layer (using predictions made by the predictors of the second layer). Once this is done, we can make a prediction for a new instance by going through each layer sequentially

### Brew

Scikit-Learn does not support stacking directly, but it is not too hard to roll out your own implementation (see the following exercises). Alternatively, you can use an open source implementation such as  brew .

## CH8. Dimensionality Reduction

> In some cases, reducing the dimensionality of the training data may filter out some noise and unnecessary details and thus result in higher performance, but in general it won't; it will just speed up training

It is also extremely useful for **data visualization**

### The Curse of Dimensionality
High-dimensional datasets are at risk of being very sparse: most training instances are likely to be far away from each other. This also means that a new instance will likely be far away from any training instance, making predictions much less reliable than in lower dimensions, since they will be based on much larger extrapolations -> **the more dimensions the training set has, the greater the risk of overfitting it**

> The number of training instances required to reach a given density grows exponentially with the number of dimensions

### Projection
Project every training instance perpendicularly onto this subspace

### Manifold
Many dimensionality reduction algorithms work by modeling the manifold on which the training instances lie; this is called Manifold Learning. It relies on the manifold assumption, also called the manifold hypothesis, which holds that most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. This assumption is very often empirically observed

### PCA
Identifies the hyperplane that lies closest to the data, and then it projects the data onto it

### Principal Components
> For each principal component, PCA finds a zero-centered unit vector pointing in the direction of the PC. Since two opposing unit vectors lie on the same axis, the direction of the unit vectors returned by PCA is not stable: if you perturb the training set slightly and run PCA again, the unit vectors may point in the opposite direction as the original vectors 

PCA assumes that the dataset is centered around the origin (Scikit-Learn take care of this)

### Explained Variance Ratio
Instead of arbitrarily choosing the number of dimensions to reduce down to, it is simpler to choose the number of dimensions that add up to a sufficiently large portion of the variance (e.g., 95%). Unless, of course, you are reducing dimensionality for data visualization—in that case you will want to reduce the dimensionality down to 2 or 3

It is possible to compress and decompress a dataset (with loss). The mean squared distance between the original data and the reconstructed data is called the **reconstruction error**

## Kernel PCA
> **Kernel trick** -> math technique that implicity maps instances into a very high-dimensional space (*feature space*). A linear decision boundary in the high-dimensional feature space corresponds to a complex nonlinear decision boundary in the original space

## Locally Linear Embedding (LLE)
Powerful *nonlinear dimensionality reduction* (NLDR) technique. Does not rely on projections, like the previous algorithms do.

> LLE works by first measuring how each training instance linearly relates to its closest neighbors, and then looking for a low-dimensional representation of the training set where these local relationships are best preserved

Scale poorly to very large datasets

## Other Dimensionality Reductions Techniques
- Random Projections: project the data to lower-dimensional space using a random linear projection
- Multidimensional Scaling (MDS): try to preserve the distances between the instances
- Isomap: creates a graph by connecting each instance to its nearest neighbors (try to preserve the geodesic distances between the instances)
- t-Distributed Stochastic Neighbor Embedding (t-SNE): try to keep similar instances close and dissimilar instances apart. Mostly used for visualization -> clusters of instances in high-dimensional space
- Linear Discriminant Analysis (LDA): classification algorithm -> learns the most discriminative axes between the classes -> can be used to define a hyperplane to project the data

## CH9. Unsupervised Learning Techniques
> "If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake" - Yann LeCun

### Clustering
Identifying similar instances and assigning them to *clusters*, or groups of similar instances

- **Customer Segmentation**: i.e., *recommender systems* to suggest X that other users in the same cluster enjoyed
- **Data Analysis**
- **Dimensionality Reduction**: Once a dataset has been clustered, it is usually possible to measure how well an instance fits into a cluster (*affinity*). Each instance’s feature vector x can then be replaced with the vector of its cluster affinities.
- **Anomaly Detection / Outlier Detection**: Any instance that has a low affinity to all the clusters is likely to be an anomaly. Useful in detecting defects in manufacturing, or for *fraud detection*
- **Semi-supervised Learning**: If you only have a few labels, you could perform clustering and propagate the labels to all the instances in the same cluster. This technique can greatly increase the number of labels available for a subsequent supervised learning algorithm, and thus improve its performance
- **Search Engines**
- **Segment an image**: By clustering pixels according to their color, then replacing each pixel's color with the mean color of its cluster, it is possible to considerably reduce the number of different colors in the image -> used in many object detection and tracking systems -> makes it easier to detect the contour of each object

**Algorithms**:
- Instances centered around a particular point -> *centroid*
- Continuous regions of densely packed instances
- Hierarchical, clusters of clusters

### K-Means
Sometimes referred to as *LLoyd-Forgy*

> *Voronoi tessellation/diagram/decomposition/partition*: is a partition of a plane into regions close to each of a given set of objects

**Hard clustering**: assigning each instance to a single cluster. **Soft clustering**: give each instance a score per cluster (can be the distance between the instance and the centroid, or the affinity score such as the Guassian Radial Basis Function)

1. Place the centroids randomly (pick *k* instances at random and use their locations as centroids)
2. Label the instances
3. Update the centroids
4. Label the instances
5. Update the centroids
6. Repeat until the centroids stop moving

> The algorithm is guaranteed to converge in a finite a number of steps (usually quite small). K-Means is generally one of the fastest clustering algorithms

#### K-Means++
Introduced a **smarter initialization step** that tends to select centroids that are distant from one another -> makes the algorithm much less likely to converge to a suboptimal solution

#### Accelerated K-Means and mini-batch K-Means
Accelerated -> exploits the triangle inequality

Mini-batches -> speeds up the algorithm by a factor of 3 or 4 -> makes it possible to cluster huge datasets that do not fit in memory (*MiniBatchKMeans* in Scikit-Learn)

> If the dataset does not fit in memory, the simplest option is to use the *memmap* class. Alternatively, you can pass one mini-batch at a time to the *partial_fit()* method, but this will require much more work, since you will need to perform multiple initializations and select the best one yourself.

#### Finding the optimal number of clusters
Plotting the inertia as a function of the number of clusters k, the curve often contains an inflexion point called the **“elbow”**

A more precise approach (but also more computationally expensive) is to use the silhouette score, which is the mean **silhouette coefficient** over all the instances. An instance’s silhouette coefficient is equal to (b – a) / max(a, b), where a is the mean distance to the other instances in the same cluster (i.e., the mean intra-cluster distance) and b is the mean nearest-cluster distance (i.e., the mean distance to the instances of the next closest cluster, defined as the one that minimizes b, excluding the instance’s own cluster). The silhouette coefficient can vary between –1 and +1. A coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters, while a coefficient close to 0 means that it is close to a cluster boundary, and finally a coefficient close to –1 means that the instance may have been assigned to the wrong cluster.

> **Silhouette diagram**: more informative visualization -> plot every instance's silhouette coefficient, sorted by the cluster they are assigned to and by the value of the coefficient

#### Limits of K-Means
- Necessary to run several times to avoid suboptimal solutions
- You need to specify the number of clusters
- Does not behave well when the clusters have varying sizes, different densities or nonspherical shapes

> It is important to scale the input features before you run K-Means, or the clusters may be very stretched and K-Means will perform poorly. Scaling the features does not guarantee that all the clusters will be nice and spherical, but it generally improves things

> **NOTE**: remember to check the book again, there are some useful practical examples on clustering for preprocessing, semi-supervised learning (*label propagation*)

##### Active learning (*uncertainty sampling*)
1. The model is trained on the labeled instances gathered so far, and this model is used to make predictions on all the unlabeled instances.
2. The instances for which the model is most uncertain (i.e., when its estimated probability is lowest) are given to the expert to be labeled.
3. You iterate this process until the performance improvement stops being worth the labeling effort.

### DBSCAN

Defines clusters as continuous regions of high density. Works well if all the clusters are dense enough and if they are well separated by low-density regions

It is robust to outliers, and it has just two hyperparameters (*eps* and *min_samples*)

### Other Clustering Algorithms
- **Agglomerative clustering**
- **BIRCH**: Balanced Iterative Reducing and Clustering using Hierarchies -> designed specifically for very large datasets
- **Mean-Shift**: computational complexity is O(m^2), not suited for large datasets
- **Affinity propagation**: same problem as mean-shift
- **Spectral clustering**: does not scale very well to large numbers of instances and it does not behave well when the clusters have very different sizes

### Gaussian Mixtures
A *Gaussian mixture model* (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. 

GMM is a *generative model* -> you can sample new instances from it

#### Anomaly Detection using Gaussian Mixtures
Using a GMM for anomaly detection is quite simple: any instance located in a low-density region can be considered an anomaly. You must define what density threshold you want to use.

A closely related task is *novelty detection*: it differs from anomaly detection in that the algorithm is assumed to be trained on a "clean" dataset, uncontaminated by outliers, whereas anomaly detection oes not make this assumption. Outlier detection is often used to clean up a dataset.

#### Selecting the Number of Clusters
Find the model that minimizes a theoretical information criterion -> *Bayesian information criterion* (BIC) or the *Akaike information criterion* (AIC)

#### Bayesian Gaussian Mixture Models
Rather than manually searching for the optimal number of clusters, you can use the BayesianGaussianMixture class, which is capable of giving weights equal (or close) to zero to unnecessary clusters. Set the number of clusters n_components to a value that you have good reason to believe is greater than the optimal number of clusters (this assumes some minimal knowledge about the problem at hand), and the algorithm will eliminate the unnecessary clusters automatically.

> GMM work great on clusters with ellipsoidal shapes, but if you try to fit a dataset with different shapes, you may have bad surprises

### Other Algorithms for Anomaly and Novelty Detection
- **PCA** (*inverse_transform()* method): If you compare the reconstruction error of a normal instance with the reconstruction error of an anomaly, the latter will usually be much larger. This is a simple and often quite efficient anomaly detection approach
- **Fast-MCD** (minimum covariance determinant): Implemented by the EllipticEnvelope class, this algorithm is useful for outlier detection, in particular to clean up a dataset
- **Isolation Forest**: efficient in high-dimensional datasets. Anomalies are usually far from other instances, so on average they tend to get isolated in fewer steps than normal instances
- **Local Outlier Factor (LOF)**: compares the density of instances around a given instance to the density around its neighbors
- **One-class SVM**: better suited for novelty detection. Works great, especially with high-dimensional datasets, but like all SVMs it does not scale to large datasets

# Part II, Neural Networks and Deep Learning

## CH10. Introduction to Artificial Neural Networks with Keras

ANNs are the very core of Deep Learning -> versatile, powerful and scalable

Renewed interest in ANNs:
- Huge quantity of data -> ANNs frequently outperform other ML techniques (large and complex problems)
- Increase in computing power -> GPU cards and cloud computing
- Tweaks to the training algorithms
- Reach fairly close to global optimum
- Virtuous circle of funding and progress

### Perceptron
Simple ANN architecture. Based on *threshold logic unit* (TLU), or *linear threshold unit* (LTU).

A Perceptron is simply composed of a single layer of TLUs, with each TLU connected to all the inputs. *Fully conected layer / dense layer*: when all the neurons in a layer are connected to every neuron in the previous layer.

> **Hebb's rule / Hebbian learning**: "Cells that fire together, wire together", the connection weight between two neurons tends to increase when they fire simultaneously 

The Perceptron learning algorithm strongly resembles Stochastic Gradient Descent. Contrary to Logistic Regression classifiers, Perceptrons do not output a class probability; rather, they make predictions based on a hard threshold

### The Multilayer Perceptron and Backpropagation

Perceptrons are incapable of solving some trivial problems (e.g., XOR) -> true of any linear classification model. This can be solved by stacking multiple Perceptrons -> *Multilayer Perceptron* (MLP)

An MLP is composed of one (passthrough) *input layer*, one or more layers of TLUs, called *hidden layers*, and one final layer of TLUs called the *output layer*. The layers close to the input layer are usually called the *lower layers*, and the ones close to the outputs are usually called the *upper layers*. Every layer except the output layer includes a bias neuron and is fully connected to the next layer

> *Feedforward neural network* (FNN): signal flows only in one direction (from the inputs to the outputs)

When an ANN contains a deep stack of hidden layers -> *Deep neural network* (DNN)

#### Backpropagation

Training algorithm. It is Gradient Descent using and efficient technique for computing the gradients automatically: in just two passes through the network (one forward, one backward), the backpropagation algorithm is able to compute the gradient of the network's error with regard to every single model parameter. It can find out how each connection weight and each bias term should be tweaked in order to reduce the error. Once it has these gradients, it just performs a regular Gradient Descent step, and the whole process is repeated until the network converges to the solution.

> Automatically computing gradients is called automatic differentiation, or *autodiff*. There are various autodiff techniques, with different pros and cons. The one used by backpropagation is called *reverse-mode autodiff*. It is fast and precise, and is well suited when the function to differentiate has many variables (e.g., connection weights) and few outputs (e.g., one loss).

For each training instance, the backpropagation algorithm first makes a prediction (forward pass) and measures the error, then goes through each layer in reverse to measure the error contribution from each connection (reverse pass), and finally tweaks the connection weights to reduce the error (Gradient Descent step)

> It is important to initialize all the hidden layers’ connection weights randomly, or else training will fail. For example, if you initialize all weights and biases to zero, then all neurons in a given layer will be perfectly identical, and thus backpropagation will affect them in exactly the same way, so they will remain identical

#### Activation functions
You need to have some nonlinearity between layers to solve very complex problems

Examples:

- Logistic (sigmoid) function
- Hyperbolic tangent (tanh)
- Rectified Linear Unit (ReLU) -> fast to compute, has become the default

#### Regression MLP architecture
**Hyperparameter - Typical value**

input neurons - One per input feature (e.g., 28 x 28 = 784 for MNIST)

hidden layers - Depends on the problem, but typically 1 to 5

neurons per hidden layer - Depends on the problem, but typically 10 to 100

output neurons - 1 per prediction dimension

Hidden activation - ReLU (or SELU, see Chapter 11)

Output activation - None, or ReLU/softplus (if positive outputs) or logistic/tanh (if bounded outputs)

Loss function - MSE or MAE/Huber (if outliers)

#### Classification MLP architecture

**Hyperparameter - Binary classification - Multilabel binary classification - Multiclass classification**

Input and hidden layers - Same as regression - Same as regression - Same as regression

output neurons - 1 - 1 per label - 1 per class

Output layer activation - Logistic - Logistic - Softmax

Loss function - Cross entropy - Cross entropy - Cross entropy 

### Implementing MLPs with Keras

Tensorflow 2 is arguably just as simple as PyTorch, as it has adopted Keras as its official high-level API and its developers have greatly simplified and cleaned up the rest of the API

> Since we are going to train the neural network using Gradient Descent, we must scale the input features

#### Creating a Sequential model

You can pass a list of layers when creating the Sequential model:

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
```

The model’s summary() method displays all the model’s layers, including each layer’s name (which is automatically generated unless you set it when creating the layer), its output shape (None means the batch size can be anything), and its number of parameters. The summary ends with the total number of parameters, including trainable and non-trainable parameters

> Dense layers often have a lot of parameters. This gives the model quite a lot of flexibility to fit the training data, but it also means that the model runs the risk of overfitting, especially when you do not have a lot of training data

#### Compiling the model

After a model is created, you must call its compile() method to specify the loss function and the optimizer to use. Optionally, you can specify a list of extra metrics to compute during training and evaluation:

```python
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```

#### Training and evaluating the model

Now the model is ready to be trained. For this we simply need to call its fit() method:

```python
>>> history = model.fit(X_train, y_train, epochs=30,
...                     validation_data=(X_valid, y_valid))
```

If the training set was very skewed, with some classes being overrepresented and others underrepresented, it would be useful to set the **class_weight** argument when calling the fit() method, which would give a larger weight to underrepresented classes and a lower weight to overrepresented classes. These weights would be used by Keras when computing the loss

> The fit() method returns a History object containing the training parameters (history.params), the list of epochs it went through (history.epoch), and most importantly a dictionary (history.history) containing the loss and extra metrics it measured at the end of each epoch on the training set and on the validation set (if any) -> use this dictionary to create a pandas DataFrame and call its plot() method to get the learning curves

> When plotting the training curve, it should be shifted by half an epoch to the left

If you are not satisfied with the performance of your model, you should go back and **tune the hyperparameters**. The first one to check is the learning rate. If that doesn’t help, try another optimizer (and always retune the learning rate after changing any hyperparameter). If the performance is still not great, then try tuning model hyperparameters such as the number of layers, the number of neurons per layer, and the types of activation functions to use for each hidden layer. You can also try tuning other hyperparameters, such as the batch size (it can be set in the fit() method using the batch_size argument, which defaults to 32).

### Building Complex Models Using the Functional API
*Wide & Deep* neural network: nonsequential, connects all or part of the inputs directly to the output layer. Makes it possible for the NN to learn both deep patterns (using the deep path) and simple rules (through the short path). Regular MLP forces all the data to flow through the full stack of layers -> simple patterns may end up being distorted by the sequence of transformations

```python
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])
```

### Using the Subclassing API to Build Dynamic Models
Sequential and Functional API are declarative:
- model can easily be saved, clone, shared
- structure can be displayed and analyzed
- framework can infer shapes and check types (caught errors early)
- easy to debug -> static graph of layers
- **problem**: it's static, so dynamic behaviors like loops, varying shapes and conditional branching are not easy

Subclassing API -> imperative programming style

Simply subclass the Model class, crete the layers you need in the constructor, and use them to perform the computations you want in the *call()* method.

> Great API for researches experimenting with new ideas

Cons:
- model's architecture is hidden within the *call()* method -> cannot inspect, save, clone
- Keras cannot check types and shapes ahead of time

> Unles you really need that extra flexibility, you should probably stick to the Sequential/Functional API

### Saving and Restoring a Model

```python
model = keras.layers.Sequential([...])
model.compile([...])
model.fit([...])
model.save("my_keras_model.h5")
# loading the model
model = keras.models.load_model("my_keras_model.h5")
```

### Using Callbacks
The fit() method accepts a callbacks argument that lets you specify a list of objects that Keras will call at the start and end of training, at the start and end of each epoch, and even before and after processing each batch. For example, the ModelCheckpoint callback saves checkpoints of your model at regular intervals during training, by default at the end of each epoch:

```python
[...] # build and compile the model
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])
```

Moreover, if you use a validation set during training, you can set save_best_only=True when creating the ModelCheckpoint. In this case, it will only save your model when its performance on the validation set is the best so far.

```python
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

### Using TensorBoard for Visualization
Interactive visualization tool:
- view learning curves during training
- compare learning curves between multiple runs
- visualize the computation graph
- analyze training statistics
- view images generated by your model
- visualize complex multidimensional data projected down to 3D and automatically clustered

To use it, modify your program so that it outputs the data you want to visualize to special binary log files called *event files*. Each binary data record is called a *summary*

```python
import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'
```
The good news is that Keras provides a nice TensorBoard() callback:
```python
[...] # Build and compile your model
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])
```

You need to start the TensorBoard server:
```bash
$ tensorboard --logdir=./my_logs --port=6006
TensorBoard 2.0.0 at http://mycomputer.local:6006/ (Press CTRL+C to quit)
```

To use directly withing Jupyter

```python
%load_ext tensorboard
%tensorboard --logdir=./my_logs --port=6006
```

### Fine-Tuning Neural Network Hyperparameters