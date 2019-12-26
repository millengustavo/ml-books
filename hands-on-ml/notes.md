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

> **Tip**: by default, the remaining columns will be dropped. To avoid this, you can specify "passthrough"

5. **Select a model and train it**

#### Training and Evaluating on the Training Set
```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

from sklearn.metris import mean_squared_error
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

# Part II, Neural Networks and Deep Learning