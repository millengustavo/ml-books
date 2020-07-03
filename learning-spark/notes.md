# Learning Spark: Lightning-Fast Data Analytics
Authors: Jules S. Damji, Brooke Wenig, Tathagata Das, Denny Lee

![learning_spark_cover](./cover.jpeg)

# Ch1. Introduction to Apache Spark: A Unified Analytics Engine

# Ch2. Downloading Apache Spark and Getting Started

# Ch3. Apache Spark's Structured APIs

# Ch4. Spark SQL and DataFrames: Introduction to Built-in Data Sources

# Ch5. Spark SQL and DataFrames: Interacting with External Data Sources

# Ch6. Spark SQL and Datasets

# Ch7. Optimizing and Tuning Spark Applications

# Ch8. Structured Streaming

# Ch9. Building Reliable Data Lakes with Apache Spark

# Ch10. Machine Learning with MLlib
Spark has two machine learning packages:
- `spark.mllib`: original ml API, based on the RDD API (maintenance mode since Spark 2.0)
- `spark.ml`: newer API, based on DataFrames -> Focuses on O(n) scale-out, so it can scale to massive amounts of data

## Terminology
- **Transformer**: accepts a DataFrame as input, and returns a new DataFrame with one or more columns appended to it. `.transform()` method
- **Estimator**: fits parameters from your DataFrame -> `.fit()` method -> returns a `Model`, which is a transformer
- **Pipeline**: organizes a series of transformers and estimators into a single model. Pipelines are estimators, but the output of `pipeline.fit()` returns a `PipelineModel`, a transformer

## Preparing Features with Transformers
Many algorithms in Spark requires that all the input features are contained within a single vector in your DataFrame

- `VectorAssembler` transformer: takes a list of input columns and creates a new DataFrame with an additional column (`features`) -> combines the values of those input columns into a single vector

## Using Estimators to Build Models
- learn parameters from data
- `estimator_name.fit()` method
- eagerly evaluated (i.e. kick off Spark jobs) whereas transformers are lazily evaluated
- examples of estimators: `Impute`, `DecisionTreeClassifier`, and `RandomForestRegressor`

## Creating a Pipeline
- specify the stages the data to pass through in order
- provide better code reusability and organization
- In Spark, `Pipelines` are estimators
- `PipelineModels` (fitted `Pipelines`) are transformers

### One Hot Encoder
- Common approach: `StringIndexer` and `OneHotEncoder`

> “How does the `StringIndexer` handle new categories that appear in the test data set, but not in the training data set?” There is a `handleInvalid` parameter. Options are `skip` (filter out rows with invalid data), `error` (throw an error), or `keep` (put invalid data in a special additional bucket, at index `numLabels`).

## Saving and Loading Models
- similar to writing DataFrames
- `model.write().save(path)`
- optionally: `model.write().overwrite().save(path)` to overwrite data in the path
- you need to specify the type of model you are loading back in -> always put your transformers/estimators into a `Pipeline`
```python
from pyspark.ml import PipelineModel
savedPipelineModel = PipelineModel.load(pipelinePath)
```

## Optimizing Pipelines
- `.setParallelism(N)`
- put the cross-validator inside the pipeline instead of putting the pipeline inside the cross-validator

# Ch11. Managing, Deploying, and Scaling Machine Learning Pipelines with Apache Spark

# Ch12. Epilogue: Apache Spark 3.0