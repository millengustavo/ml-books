# Python Machine Learning

Author: Sebastian Rashcka

![python-ml-cover](cover_1.jpg)

## Ch4. Building Good Training Datasets – Data Preprocessing

### Missing Data

Unfortunately, most computational tools are unable to handle such missing values or will produce unpredictable results if we simply ignore them. Therefore, it is crucial that we take care of those missing values before we proceed with further analyses.

Nowadays, most scikit-learn functions support  DataFrame  objects as inputs, but since NumPy array handling is more mature in the scikit-learn API, it is recommended to use NumPy arrays when possible.

```python
# only drop rows where NaN appear in specific columns (here: 'C') 
 >>>   df.dropna(subset=['C'])
```

### Categorical Features

When we are talking about  categorical  data, we have to further distinguish between  ordinal  and  nominal  features. Unfortunately, there is no convenient function that can automatically derive the correct order of the labels of our  size  feature, so we have to define the mapping manually.

> We can simply define a reverse-mapping dictionary,  inv_size_mapping = {v: k for k, v in size_mapping.items()}

Most estimators for classification in scikit-learn convert class labels to integers internally, but it is considered good practice to provide class labels as integer arrays to avoid technical glitches.

Although the color values don't come in any particular order, a learning algorithm will now assume that  green  is larger than  blue , and  red  is larger than  green . Although this assumption is incorrect, the algorithm could still produce useful results. However, those results would not be optimal.

A common workaround for this problem is to use a technique called **one-hot encoding**. The idea behind this approach is to create a new dummy feature for each unique value in the nominal feature column.

When we are using one-hot encoding datasets, we have to keep in mind that this introduces  multicollinearity, which can be an issue for certain methods (for instance, methods that require matrix inversion). If features are  highly correlated, matrices are computationally difficult to invert, which can lead to numerically unstable estimates. To reduce the correlation among variables, we can simply remove one feature column from the one-hot encoded array.

> Providing the class label array  y  as an argument to  stratify  ensures that both training and test datasets have the same class proportions as the original dataset.

Instead of discarding the allocated test data after model training and evaluation, it is a common practice to retrain a classifier on the entire dataset, as it can improve the predictive performance of the model.

### Feature Scaling

**Feature scaling**  is a crucial step  in our preprocessing pipeline that can easily be forgotten.  Decision trees  and  random forests  are two of the very few machine learning  algorithms  where we don't need to worry about feature scaling. Those algorithms are scale invariant. However, the majority of machine learning and optimization algorithms behave much better if features are on the same scale

There are two common approaches to bringing different features onto the same scale:  **normalization**  and  **standardization**

**Standardization** can be more practical for many machine learning algorithms, especially for optimization algorithms such as gradient descent. The reason is that many linear models initialize the weights to 0 or small random values close to 0. Using standardization, we center the feature columns at mean 0 with standard deviation 1 so that the feature columns have the same parameters as a standard normal distribution (zero mean and unit variance), which makes it easier to learn the weights.

Standardization maintains useful information about outliers and makes the algorithm less sensitive to them in contrast to min-max scaling, which scales the data to a limited range of values

> We fit the  StandardScaler  class only once—on the training data—and use those parameters to transform the test dataset or any new data point

The  **RobustScaler**  is especially helpful and recommended if we are working with small datasets that contain  many outliers. If the machine learning algorithm applied to this dataset is prone to  overfitting , the  RobustScaler  can be a good choice

Overfitting means the model fits the parameters too closely with regard to the particular observations in the training dataset, but does not generalize well to new data; we say that the  model has a  high variance . The reason for the overfitting is that our model is too complex for the given training data.

Common ways to reduce overfitting by regularization and dimensionality reduction via **feature selection**, which leads to simpler models by requiring fewer parameters to be fitted to the data.

### Regularization

- **L1 regularization** usually yields sparse feature vectors and most feature weights will be zero. Sparsity can be useful in practice if we have a high-dimensional dataset with many features that are irrelevant, especially in cases where we have more irrelevant dimensions than training examples.

- **L2 regularization** adds a penalty term to the cost function that effectively results in less extreme weight values compared to a model trained with an unregularized cost function.

There are two main categories of dimensionality reduction  techniques:  **feature selection**  and  **feature extraction** . Via feature selection, we select a subset of the original features, whereas in feature extraction, we derive information from the feature set to construct a new feature subspace.

**Greedy algorithms**  make  locally optimal choices at each stage of a combinatorial search problem and generally yield a suboptimal solution to the problem, in contrast to  exhaustive search algorithms , which  evaluate all possible combinations and are guaranteed to find the optimal  solution

By reducing the number of features, we shrank the size of the dataset, which can be useful in real-world applications that may involve expensive data collection steps. Also, by substantially reducing the number of features, we obtain simpler models, which are easier to interpret.

### Random Forest feature importance

Using a random forest, we can measure the feature importance as the averaged impurity decrease computed from all decision trees in the forest, without making any assumptions about whether our data is linearly separable or not.

> As far as interpretability is concerned, the random forest technique comes with an important  gotcha  that is worth mentioning. **If two or more features are highly correlated, one feature may be ranked very highly while the information on the other feature(s) may not be fully captured.**

**SelectFromModel**  object that selects features based on a user-specified threshold after model fitting, which is useful if we want to use the  RandomForestClassifier  as a feature selector and intermediate step in a scikit-learn  Pipeline  object,

## Ch6. Learning Best Practices for Model Evaluation and Hyperparameter Tuning




