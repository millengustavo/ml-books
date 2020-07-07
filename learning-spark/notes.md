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

## Model management
Ensure that you can reproduce and track the model's performance

Important aspects:
- **Library versioning**
- **Data evolution**
- **Order of execution**
- **Parallel operations** (nondeterministic outputs)

> Having industry-wide standards for managing models is important so they can be easily shared with partners

### MLflow
Open source platform that helps developers reproduce and share experiments, manage models, and more

#### Tracking
- Logging API agnostic to libraries and envs that actually do the training
- Organized around the concept of *runs* (executions of DS code)
- Runs are aggregated into *experiments*
- Can log:
  - *parameters*
  - *metrics*
  - *artifacts*
  - *metadata*
  - *models*

```python
import mflow
import mlflow.spark

with mlflow.start_run(run_name="your-run-name") as run:
    mlflow.log_param(...)
    (...)
    mlflow.spark.log_model(...)
    (...)
    mlflow.log_metrics(...)
    (...)
    mlflow.log_artifact(...)
```

## Model deployment options with MLlib
- Batch: more efficient per data point -> accumulate less overhead when amortized across all predictions made
- Streaming: nice trade-off between throughput and latency
- Real-time: prioritizes latency over throughput and generates predictions in a few milliseconds

> MLlib is not intended for real-time model inference (computing predictions in under 50 ms)

### Batch
Majority of use cases for deploying ml models. Run a regular job to generate predictions, and save the results to a table, database, data lake, etc, for downstream consumption

MLlib's `model.transform()`: apply the model in parallel to all partitions of your DataFrame

Questions to keep in mind:
- **How frequently will you generate predictions?** latency and throughput trade-off
- **How often will you retrain the model?** MLlib doesn't support online updates or warm restarts
- **How will you version the model?** Use the MLflow model registry to keep track of the models and control how they are transitioning to/from staging, productions, and archived

### Streaming
Structured Streaming can continuously perform inference on incoming data
- more costly than batch
- benefit of generating predictions more frequently
- more complicated to maintain and monitor
- offer lower latency (but if you need really low-latency predictions, you'll need to export your model out of Spark)

> Use `spark.readStream()` rather than `spark.read()` and change the source of the data. We need to define the schema a priori when working with streaming data

### Model export patterns for real-time inference
- ONNX (Open Neural Network Exchange): open standard for machine learning interoperability
- Other 3rd party libs integrate with Spark and are convenient to deploy in real-time scenarios: **XGBoost** and **H20.ai's Sparkling Water**
- **XGBoost** is not technically part of MLlib,
  - **XGBoost4J-Spark** library allows you to integrate distributed XGBoost into your MLlib pipelines. 
  - ease of deployment: train your MLlib pipeline, extract the XGBoost model and save it as a non-Spark model for serving in Python

## Leveraging Spark for Non-MLlib models

### Pandas UDFs
- UDF = user-defined functions
- Build a scikit-learn/TensorFlow model on a single machine (on a subset of your data)
- Perform distributed inference on the entire data set using Spark
- Spark 3.0: `mapInPandas()`: apply a scikit-learn model

### Spark for distributed hyperparameter tuning
#### Joblib
- Set of tools to provide lightweight pipelining in Python
- It has a Spark backend to distribute tasks on a Spark cluster
- `pip install joblibspark`

```python
from joblibspark import register_spark


register_spark() # Register Spark backend
```

#### Hyperopt
- Python lib for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions
- Main ways to scale Hyperopt with Apache Spark:
  - Using single-machine Hyperopt with a distributed training algorithm (e.g. MLlib)
  - Using distributed Hyperopt with single-machine training algorithms with the `SparkTrials` class

> **Koalas**: open source library that implements the Pandas DataFrame API on top of Apache Spark. Replace any `pd` logic in your code with `ks` (Koalas) -> scale up your analyses with Pandas without needing to entirely rewrite your codebase in PySpark. `kdf.to_spark()` = switch to using Spark APIs. `kdf.to_pandas()`= switch to using Pandas API


# Ch12. Epilogue: Apache Spark 3.0