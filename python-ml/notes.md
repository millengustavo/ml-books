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

## Ch5. Compressing Data via Dimensionality Reduction

### Feature Extraction
Feature extraction can be understood as an approach to data compression with the goal of maintaining most of the relevant information.

> Feature  extraction is not only used to improve storage space or the computational efficiency of the learning  algorithm, but can also improve the predictive performance by reducing the  curse of dimensionality —especially if we are working with non-regularized models.

### PCA

PCA aims to find the directions of maximum variance in high-dimensional data and projects the data onto a new subspace with equal or fewer dimensions than the original one

Even if the  input features are correlated, the resulting principal components will be mutually orthogonal (uncorrelated). 

> PCA directions are highly sensitive to data scaling, and we need to standardize the features  prior  to PCA if the features were measured on different scales and we want to assign equal importance to all features

### LDA

The general concept behind LDA is very similar to PCA, but whereas PCA attempts to find the orthogonal component axes of maximum variance in a dataset, the goal in LDA is to find the feature subspace that optimizes class separability

### Kernel PCA

If we are dealing with nonlinear problems, which we may encounter rather  frequently in real-world  applications, linear transformation techniques for dimensionality reduction, such as PCA and LDA, may not be the best choice.

> Using the kernel trick, we can compute the similarity between two high-dimension feature vectors in the original feature space


## Ch6. Learning Best Practices for Model Evaluation and Hyperparameter Tuning

> We have to reuse the parameters that were obtained during the fitting of the training data to scale and compress any new data, such as the examples in the separate test dataset

### Combining transformers and estimators in a pipeline

There is no limit to the number of intermediate steps in a pipeline; however, the last pipeline element has to be an estimator.

If we reuse the same test dataset over and over again  during model selection, it will become part of our training data and thus the model will be more likely to overfit. **Despite this issue, many people still use the test dataset for model selection, which is not a good machine learning practice**

### Using k-fold cross-validation to assess model performance

A better way of using the holdout method for model selection is to separate the data into three parts: a training dataset, a validation dataset, and a test dataset. The training dataset is used to fit the different models, and the performance on the validation dataset is then used for the model selection. The advantage of having a test dataset that the model hasn't seen before during the training and model selection steps is that we can obtain a less biased estimate of its ability to generalize to new data.

A disadvantage of the holdout method is that the performance estimate may be very sensitive to how we partition the training dataset into the training and validation subsets; the estimate will vary for different examples of the data

### K-fold cross-validation

In k-fold cross-validation, we randomly split the training dataset into  k  folds without replacement, where  k  – 1 folds  are used for the model training, and one fold is used for performance evaluation. This procedure is repeated  k  times so that we obtain  k  models and performance estimates.

We use k-fold cross-validation for model tuning, that is, finding the optimal hyperparameter values that yield a satisfying generalization performance, which is estimated from evaluating the model performance on the test folds.

> Once we have found satisfactory hyperparameter values, we can retrain the model on the complete training dataset and obtain a final performance estimate using the independent test  dataset. The rationale behind fitting a model to the whole training dataset after k-fold cross-validation is that providing more training examples to a learning algorithm usually results in a more accurate and robust model.

Since k-fold cross-validation is a resampling technique without replacement, the advantage of this approach is that each example will be used for training and validation (as part of a test fold) exactly once, which yields a lower-variance estimate of the model performance than the holdout method.

### Choosing K

A good standard value for  k  in k-fold cross-validation is 10, as empirical evidence shows. For instance, experiments by Ron Kohavi on various real-world datasets suggest that 10-fold cross-validation  offers the best tradeoff between bias and variance ( A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection ,  Kohavi, Ron ,  International Joint Conference on Artificial Intelligence (IJCAI) , 14 (12): 1137-43,  1995 ).

**Small training sets** -> increase the number of folds. If we increase the value of  k , more training data will be used in each iteration, which results in a lower pessimistic bias toward estimating the generalization performance by averaging the individual model estimates.

**Large datasets** -> smaller value for  k , for example,  k  = 5, and still obtain an accurate estimate of the average performance of the model while reducing the computational cost of refitting and evaluating the model on the different folds

### Stratified cross-validation

A slight improvement over the standard k-fold cross-validation approach is stratified k-fold cross-validation, which can yield better bias and variance estimates, especially in cases of unequal class proportions

By plotting the model training and validation accuracies as functions of the training dataset size, we can easily detect whether the model suffers from high variance or high bias, and whether the collection of more data could help to address this problem

### Bias x Variance

**High bias**: This model has both low  training and cross-validation accuracy, which indicates that it underfits the training data. Common ways to address this  issue are to increase the number of parameters of the model, for example, by collecting or constructing additional  features, or by decreasing the degree of regularization

**High variance**: which is indicated by the large gap between the training and cross-validation accuracy. To address this problem of overfitting, we can collect more training data, reduce the complexity of the model, or increase the regularization parameter

For unregularized models, it  can also help to decrease  the number of features via feature selection

> While collecting more training data usually tends to decrease the chance of overfitting, it  may not always help, for example, if the training data is extremely noisy or the model is already very close to optimal.

### Debugging algorithms with learning and validation curves

Validation curves are a useful tool for improving the performance of a model by addressing  issues  such as overfitting or underfitting. Validation curves are related  to learning curves, but instead of plotting the training and test accuracies as functions of the sample size, we vary  the values of the model parameters

### Fine-tuning machine learning models

The grid search approach is quite simple: it's a brute-force exhaustive search paradigm where  we specify a list of values for different  hyperparameters, and the computer evaluates the model performance for each combination to obtain the optimal combination of values from this set

Randomized search usually performs about as well as grid search but is  much more cost- and time-effective. In particular, if we only sample 60 parameter combinations via randomized search, we already have a 95 percent probability of obtaining solutions within 5 percent of the optimal performance ( Random search for hyper-parameter optimization .  Bergstra J ,  Bengio Y .  Journal of Machine Learning Research . pp. 281-305, 2012).

### Algorithm selection with nested cross-validation

If we want to select among different machine learning algorithms, though, another recommended approach is nested cross-validation. In a nice study on the bias in error estimation, Sudhir Varma and Richard Simon concluded that the true error of the estimate is almost unbiased relative to the test dataset when nested cross-validation is used ( Bias in Error Estimation When Using Cross-Validation for Model Selection ,  BMC Bioinformatics ,  S. Varma  and  R. Simon , 7(1): 91,  2006 ).

In nested cross-validation, we have an outer k-fold cross-validation loop to split the data into training and test folds, and an inner loop is used to select the model using k-fold cross-validation on the training fold. After model selection, the test fold is then used to evaluate the model performance

### Looking at different performance evaluation metrics

A confusion matrix is simply a  square matrix that reports the counts of the  true positive  ( TP ),  true negative  ( TN ),  false positive  ( FP ), and  false negative  ( FN ) predictions  of a classifier

The  true positive rate  ( TPR ) and  false positive rate  ( FPR ) are performance metrics that are especially  useful for  imbalanced  class problems

Receiver operating characteristic  ( ROC ) graphs are useful tools to select models for classification based  on their performance with respect to the FPR and TPR, which are computed by shifting the decision threshold of the classifier. The diagonal of a ROC graph can be interpreted as  random guessing , and classification models that fall below the diagonal are considered as worse than random guessing. A perfect classifier would fall into the top-left corner of the graph with a TPR of 1 and an  FPR of 0. Based on the ROC curve, we can then compute the so-called  ROC area under the curve  ( ROC AUC ) to characterize the performance of a classification model.

#### Scoring metrics for multiclass classification

> Micro-averaging is useful if we want to weight each instance or prediction equally, whereas macro-averaging weights all classes equally to evaluate the overall performance of a  classifier with regard to the most frequent class labels. If we are using binary performance metrics to evaluate multiclass classification models in scikit-learn, a normalized or weighted variant of the macro-average is used by default

### Dealing with class imbalance

Class imbalance is a quite common problem when working with real-world data—examples from one class or multiple classes are over-represented in a dataset

The algorithm implicitly learns a model that optimizes the predictions based on the most abundant class in the dataset, in order to minimize the cost or maximize the reward during training.

One way to deal with imbalanced class proportions during model fitting is to assign a larger penalty to wrong predictions on the minority class.

Other popular strategies for dealing with class imbalance include upsampling the minority class, downsampling the majority class, and the generation of synthetic training examples. Unfortunately, there's no universally best solution or technique that works best across different problem domains. Thus, in practice, it is recommended to try out different strategies on  a given problem, evaluate the results, and choose the technique that seems most appropriate

### SMOTE

Synthetic Minority Over-sampling Technique  ( SMOTE ), and you can learn more about this  technique in the original research article by Nitesh Chawla and others:  SMOTE: Synthetic Minority Over-sampling Technique ,  Journal of Artificial Intelligence Research , 16: 321-357,  2002 . It is also highly recommended to check out  imbalanced-learn , a Python library that is entirely focused on imbalanced datasets, including an  implementation of SMOTE

## Ch7. Combining different models for Ensemble Learning

The  goal of  ensemble methods  is to combine different classifiers into a meta-classifier that has better  generalization performance than each individual classifier alone.

### Voting

Majority  voting simply means that we select the class label that has been predicted by the majority of classifiers, that is, received more than 50 percent of the votes.

> To predict a  class label via simple majority or plurality voting, we can combine the predicted class labels of each individual classifier,  , and select the class label,  , that received the most votes

### Stacking

The  stacking algorithm can be understood as a two-level ensemble, where the first level consists of individual classifiers that feed their predictions to the second level, where another classifier (typically logistic regression) is fit to the level-one classifier predictions to make the final predictions. The stacking algorithm has been described in more detail by David H. Wolpert in  Stacked generalization ,  Neural Networks , 5(2):241–259,  1992 .

### Bagging

Instead of using the same training dataset to fit the individual classifiers in the ensemble, we draw bootstrap samples (random samples with replacement) from the initial training dataset, which is why bagging is also known as  bootstrap aggregating .

**Random forests** are a special case of bagging where we also use random feature subsets when fitting the individual decision trees.

> Bagging was first  proposed by Leo Breiman in a  technical report in 1994; he also showed that bagging can improve the accuracy of unstable models and decrease the degree of overfitting.

Bagging algorithm can be an effective approach to reducing the variance of a model. However, bagging is ineffective in reducing model bias, that is, models that are too simple to capture the trend in the data well. This is why we want to perform bagging on an ensemble of classifiers with low bias, for example, unpruned decision trees.

### Boosting

In boosting, the  ensemble consists of very simple base classifiers, also often referred to as  weak learners , which often only have a slight  performance advantage over random guessing—a typical example of a weak learner is a decision tree stump. 

> The key concept behind boosting is to focus on training examples that are hard to classify, that is, to let the weak learners subsequently learn from misclassified training examples to improve the performance of the ensemble.

In contrast  to bagging, the initial formulation of the boosting algorithm uses random subsets of training examples drawn from the training dataset without replacement

Boosting can lead to a decrease in bias as well as variance compared to bagging models

Boosting algorithms such as AdaBoost are also known for their high variance, that is, the tendency to overfit the training data

> It is worth noting that ensemble learning increases the computational complexity compared to individual classifiers. In practice, we need to think carefully about whether we want to pay the price of increased computational costs for an often relatively modest improvement in predictive performance. An often-cited  example of this tradeoff is the famous $1 million  Netflix Prize , which was won using ensemble techniques. The details about  the algorithm were published in  The BigChaos Solution to the Netflix Grand Prize  by  A. Toescher ,  M. Jahrer , and  R. M. Bell ,  Netflix Prize documentation ,  2009

### Gradient Boosting

Another popular variant of boosting is  gradient boosting . AdaBoost and gradient boosting share the  main overall concept: boosting weak learners (such as decision tree stumps) to strong learners. The two approaches, adaptive and gradient boosting, differ mainly with regard to how the weights are updated and how the (weak) classifiers are combined

#### XGBoost

XGBoost, which is essentially a computationally efficient implementation of the original gradient boost algorithm ( XGBoost: A scalable tree boosting system .  Tianqi Chen  and  Carlos Guestrin .  Proceeding of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining .  ACM 2016 , pp. 785-794)

#### HistGradientBoosting

Scikit-learn now also includes a substantially faster version of gradient boosting in version 0.21,  HistGradientBoostingClassifier , which is even faster than XGBoost

> Ensemble methods combine different classification models to cancel out their individual weaknesses, which often results in stable and well-performing models that are very attractive for industrial applications as well as machine learning competitions.

## Ch8. Applying Machine Learning to Sentiment Analysis

Sentiment analysis, sometimes  also  called  opinion mining , is a popular subdiscipline of the broader field of NLP; it is concerned with analyzing the polarity of documents. A popular task in sentiment analysis is the classification of documents based on the expressed opinions or emotions of the authors with regard to a particular topic.

> To visualize the progress and estimated time until completion, use the  Python Progress Indicator  ( **PyPrind** ,  https://pypi.python.org/pypi/PyPrind/ )

### Bag-of-words

The idea behind bag-of-words is quite simple and can be summarized as follows: We create a vocabulary of unique tokens—for example, words—from the entire set of documents. We construct a feature vector from each document that contains the counts of how often each word occurs in the particular document.

> To construct  a bag-of-words model based on the word  counts in the respective documents, we can use the  CountVectorizer  class implemented in scikit-learn

Values in the feature vectors are also  called the  raw term frequencies :  tf ( t ,  d )—the number of times a term,  t , occurs in a document,  d . It should be noted that, in the bag-of-words model, the word or term order in a sentence or document does not matter. The order in which the term frequencies appear in the feature vector is derived from the vocabulary indices, which are usually assigned alphabetically.

#### N-grams

The sequence of items  in the bag-of-words model that we just created is also  called the  1-gram  or  unigram  model—each item or token in the vocabulary represents a single word. More generally, contiguous sequences of items in NLP—words, letters, or  symbols—are also called  n-grams

#### TF-IDF

Frequently occurring words typically don't contain useful or discriminatory information

Term frequency-inverse document frequency  ( tf-idf ), which can be used to downweight  these frequently occurring words in the feature vectors. The tf-idf can be defined as the product of the term frequency and the inverse document frequency

### Cleaning text

The first important step—before we build our bag-of-words model—is to clean the text data by stripping it of all unwanted characters.

One way to  tokenize  documents is to split them into individual words by splitting the cleaned documents at their whitespace characters

#### Stemming

In the context of tokenization, another  useful technique is  word stemming , which is the process of transforming a word into its root form. It allows us to map related words to the same stem

The Porter stemming algorithm  is probably the oldest and simplest stemming algorithm. Other popular stemming algorithms include the  newer  Snowball stemmer  (Porter2 or English stemmer) and the  Lancaster stemmer  (Paice/Husk stemmer)

#### Lemmatization

While stemming can create non-real words, such as  'thu'  (from  'thus' ), as shown in the previous example, a technique called  lemmatization  aims to obtain  the canonical (grammatically correct) forms of individual words—the so-called  lemmas . 

> Lemmatization  is computationally more difficult and expensive compared to stemming and, in practice, it has been observed that stemming and lemmatization have little impact on the performance of text classification

### Stop-word removal 

Stop-words  are simply those words that are extremely common in all sorts of texts and probably bear no (or only a little) useful information that can be used to distinguish between different classes of documents. Examples of stop-words are  is ,  and ,  has , and  like . Removing stop-words can be useful if we are working with raw or normalized term frequencies rather than tf-idfs, which are already downweighting frequently occurring words.

TfidfVectorizer , which combines  CountVectorizer  with the  TfidfTransformer

### Naive Bayes Classifier

A still very popular classifier for text classification is the naïve Bayes classifier, which  gained popularity in applications of email spam filtering. Naïve Bayes classifiers are easy to implement, computationally efficient, and tend to perform particularly well on relatively small datasets

### Out-of-core learning

Out-of-core learning allows us to work with large datasets by fitting the classifier incrementally on smaller batches of a dataset.

Unfortunately, we  can't use  CountVectorizer  for out-of-core learning since it requires holding the complete vocabulary in memory. Also,  TfidfVectorizer  needs to keep all the feature vectors of the training dataset in memory to calculate the inverse document frequencies. However, another useful vectorizer for text processing implemented in scikit-learn is  HashingVectorizer .  HashingVectorizer  is data-independent

Out-of-core learning is very memory efficient

### word2vec

A more  modern alternative to the bag-of-words model is  word2vec , an algorithm that Google released in 2013 ( Efficient Estimation of Word Representations in Vector Space ,  T. Mikolov ,  K. Chen ,  G. Corrado , and  J. Dean , arXiv preprint arXiv:1301.3781,  2013 ). The word2vec algorithm is an unsupervised learning algorithm based on neural networks that attempts to automatically learn the relationship between words. The idea behind word2vec is to put words that have similar meanings into similar clusters, and via clever vector-spacing, the model can reproduce certain words using simple vector math, for example,  king  –  man  +  woman  =  queen .

### Topic Modeling

Topic modeling  describes  the broad task of assigning topics to unlabeled text documents. For example, a typical application would be the categorization of documents in a large text corpus of newspaper articles. In applications of topic modeling, we then aim to  assign category labels to those articles, for example, sports, finance, world news, politics, local news, and so forth

###  Latent Dirichlet Allocation  ( LDA )

LDA is a generative probabilistic model that tries to find groups of words that appear frequently together across different documents

We must define the number of topics beforehand—the number of topics is a hyperparameter of LDA that has to be specified manually.

> The scikit-learn library's implementation of LDA uses the  *expectation-maximization*  ( EM ) algorithm to  update its parameter estimates iteratively

## Ch9. Embedding a Machine Learning Model into a Web Application

### Serializing fitted scikit-learn estimators
- One option for model persistence: Python's in-built `pickle` module
- `protocol=4` to choose the latest and most efficient pickle protocol
- `joblib`: lib, more efficient way to serialize NumPy arrays

> **Pickle can be a security risk**: not secured against malicious code. Pickle was designed to serialize arbitraty objects, the unpickling process will execute code that has been stored in a pickle file

### SQLite
SQLite database can be understood as a single, self-contained database file that allows us to directly access storage files

> free DB browser for SQLite app (https://sqlitebrowser.org/dl/) -> nice GUI for working with SQLite databases

### Flask
Written in Python, provides a convenient interface for embedding existing Python code

### Jinja2
Web templates

### PythonAnywhere
Lets us run a single web application free of charge

## Ch10. Predicting Continuous Target Variables with Regression Analysis

### Linear Regression
**Regression Line**: best-fitting line

**Offsets/Residuals**: vertical lines from the regression line to the training examples -> errors of our prediction

### Visualizing the important characteristics of a dataset
- **Scatterplot matrix**: pair-wise correlations between the different features -> *scatterplotmatrix* on MLxtend (https://github.com/rasbt/mlxtend)

> Training a linear regression model does **not** require that the explanatory or target variables are normally distributed -> only requirement for certain statistics and hypothesis tests

### Looking at relationships using a correlation matrix
- **Correlation matrix**: square matrix that contains the *Pearson product-moment correlation coefficient* (*Pearson's r*) -> linear dependence between pairs of features
- Correlation coefficients are in range -1 to 1. 1 -> perfect positive correlation, 0 -> no correlation and -1: perfect negative correlation

> To fit a linear regression model, we are interested in those features that have a high correlation with our target variable

### Estimating the coefficient of a regression model via scikit-learn
The linear regression implementation in scikit-learn works better with unstandardized variables

### Fitting a robust regression model using RANSAC
- Linear regression models can be heavily impacted by the presence of outliers. In certain situations, a very small subset of our data can have a big effect on the estimated model coefficients

- As an alternative to throwing out outliers, we will look at a robust method of regression using the **RANdom SAmple Consensus (RANSAC)** algorithm, which fits a regression model to a subset of the data, the so-called **inliers**

### Evaluating the performance of linear regression models
- Plot the residuals (the differences or vertical distances between the actual and predicted values) versus the predicted values to diagnose our regression model.
- **Residual plots** are a commonly used graphical tool for diagnosing regression models. They can help to detect nonlinearity and outliers, and check whether the errors are randomly distributed

> Good regression model: errors randomly distributed and the residuals randomly scattered around the centerline

- **MSE**: useful for comparing differente regression models or for tuning their parameters via grid search and cross-validation

- **Rˆ2**: coefficient of determination. Standardized version of the MSE -> better interpretability of the model's performance. Rˆ2 is the fraction of response variance that is captured by the model

### Using regularized methods for regression
- Regularization is one approach to tackling the problem of overfitting by adding additional information, and thereby shrinking the parameter values of the model to induce a penalty against complexity. 

- The most popular approaches to regularized linear regression are the so-called **Ridge Regression, least absolute shrinkage and selection operator (LASSO), and elastic Net**

> Saturation of a model occurs if the number of training examples is equal to the number of features, which is a form of overparameterization. As a consequence, a saturated model can always fit the training data perfectly but is merely a form of interpolation and thus is not expected to generalize well

### Dealing with nonlinear relationships using random forests
In the context of decision tree regression, the MSE is often referred to as **within-node variance**, which is why the splitting criterion is also better known as **variance reduction**

> If the distribution of the residuals does not seem to be completely random around the zero center point -> the model was not able to capture all the exploratory information

- The error of the predictions should not be related to any of the information contained in the explanatory variables; rather, it should reflect the randomness of the real-world distributions or patterns. If we find patterns in the prediction errors, for example, by inspecting the residual plot, it means that the residual plots contain predictive information

- Improve the model by transforming variables, tuning the hyperparameters of the learning algorithm, choosing simpler or more complex models, removing outliers, or including additional variables