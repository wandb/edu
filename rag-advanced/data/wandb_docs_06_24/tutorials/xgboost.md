# XGBoost

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb)

In this notebook we'll train a XGBoost model to classify whether submitted loan applications will default or not. Using boosting algorithms such as XGBoost increases the performance of a loan assesment, whilst retaining interpretability for internal Risk Management functions as well as external regulators.

This notebook is based on a talk from Nvidia GTC21 by Paul Edwards at ScotiaBank who [presented](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/) how XGBoost can be used to construct more performant credit scorecards that remain interpretable. They also kindly [shared sample code](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard) which we will use throughout this notebook, credit to Stephen Denton (stephen.denton@scotiabank.com) from Scotiabank for sharing this code publicly.

### [Click here](https://wandb.ai/morgan/credit_scorecard) to view and interact with a live W&B Dashboard built with this notebook

# In this notebook

In this colab we'll cover how Weights and Biases enables regulated entities to 
- **Track and version** their data ETL pipelines (locally or in cloud services such as S3 and GCS)
- **Track experiment results** and store trained models 
- **Visually inspect** multiple evaluation metrics 
- **Optimize performance** with hyperparameter sweeps

**Track Experiments and Results**

We will track all of the training hyperparameters and output metrics in order to generate an Experiments Dashboard:

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**Run a Hyperparameter Sweep to Find the Best HyperParameters**

Weights and Biases also enables you to do hyperparameter sweeps, either with our own [Sweeps functionality](https://docs.wandb.ai/guides/sweeps) or with our [Ray Tune integration](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune). See our docs for a full guide of how to use more advanced hyperparameter sweeps options.

![credit_scorecard_2](/images/tutorials/credit_scorecard/credit_scorecard_2.png)

# Setup


```bash
!pip install -qq wandb>=0.13.10 dill
!pip install -qq xgboost>=1.7.4 scikit-learn>=1.2.1
```


```python
import ast
import sys
import json
from pathlib import Path
from dill.source import getsource
from dill import detect

import pandas as pd
import numpy as np
import plotly
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn import metrics
from sklearn import model_selection
import xgboost as xgb

pd.set_option("display.max_columns", None)
```

# Data

## AWS S3, Google Cloud Storage and W&B Artifacts

![credit_scorecard_3](/images/tutorials/credit_scorecard/credit_scorecard_3.png)

Weights and Biases **Artifacts** enable you to log end-to-end training pipelines to ensure your experiments are always reproducible.

Data privacy is critical to Weights & Biases and so we support the creation of Artifacts from reference locations such as your own private cloud such as AWS S3 or Google Cloud Storage. Local, on-premises of W&B are also available upon request. 

By default, W&B stores artifact files in a private Google Cloud Storage bucket located in the United States. All files are encrypted at rest and in transit. For sensitive files, we recommend a private W&B installation or the use of reference artifacts.

## Artifacts Reference Example
**Create an artifact with the S3/GCS metadata**

The artifact only consists of metadata about the S3/GCS object such as its ETag, size, and version ID (if object versioning is enabled on the bucket).

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**Download the artifact locally when needed**

W&B will use the metadata recorded when the artifact was logged to retrieve the files from the underlying bucket.

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

See [Artifact References](https://docs.wandb.ai/guides/artifacts/references) for more on how to use Artifacts by reference, credentials setup etc.

## Log in to W&B
Log in to Weights and Biases 


```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## Vehicle Loan Dataset

We will be using a simplified version of the [Vehicle Loan Default Prediction dataset](https://www.kaggle.com/sneharshinde/ltfs-av-data) from L&T which has been stored in W&B Artifacts. 


```python
# specify a folder to save the data, a new folder will be created if it doesn't exist
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

Create function to pickle functions


```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### Download Data from W&B Artifacts

We will download our dataset from W&B Artifacts. First we need to create a W&B run object, which we will use to download the data. Once the data is downloaded it will be one-hot encoded. This processed data will then be logged to the same W&B as a new Artifact. By logging to the W&B that downloaded the data, we tie this new Artifact to the raw dataset Artifact


```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

Download the subset of the vehicle loan default data from W&B, this contains `train.csv` and `val.csv` files as well as some utils files.


```python
ARTIFACT_PATH = "morgan/credit_scorecard/vehicle_loan_defaults:latest"
dataset_art = run.use_artifact(ARTIFACT_PATH, type="dataset")
dataset_dir = dataset_art.download(data_dir)
```


```python
from data_utils import (
    describe_data_g_targ,
    one_hot_encode_data,
    load_training_data,
)
```

#### One-Hot Encode the Data


```python
# Load data into Dataframe
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# One Hot Encode Data
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# Save Preprocessed data
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### Log Processed Data to W&B Artifacts


```python
# Create a new artifact for the processed data, including the function that created it, to Artifacts
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="One-hot encoded dataset",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# Attach our processed data to the Artifact
processed_ds_art.add_file(processed_data_path)

# Log this Artifact to the current wandb run
run.log_artifact(processed_ds_art)

run.finish()
```

## Get Train/Validation Split

Here we show an alternative pattern for how to create a wandb run object. In the cell below, the code to split the dataset is wrapped with a call to `wandb.init() as run`. 

Here we will:

- Start a wandb run
- Download our one-hot-encoded dataset from Artifacts
- Do the Train/Val split and log the params used in the split 
- Log the new `trndat` and `valdat` datasets to Artifacts
- Finish the wandb run automatically


```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # config is optional here
    # Download the subset of the vehicle loan default data from W&B
    dataset_art = run.use_artifact(
        "vehicle_defaults_processed:latest", type="processed_dataset"
    )
    dataset_dir = dataset_art.download(data_dir)
    dataset = pd.read_csv(processed_data_path)

    # Set Split Params
    test_size = 0.25
    random_state = 42

    # Log the splilt params
    run.config.update({"test_size": test_size, "random_state": random_state})

    # Do the Train/Val Split
    trndat, valdat = model_selection.train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[[targ_var]],
    )

    print(f"Train dataset size: {trndat[targ_var].value_counts()} \n")
    print(f"Validation dataset sizeL {valdat[targ_var].value_counts()}")

    # Save split datasets
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    trndat.to_csv(train_path, index=False)
    valdat.to_csv(val_path, index=False)

    # Create a new artifact for the processed data, including the function that created it, to Artifacts
    split_ds_art = wandb.Artifact(
        name="vehicle_defaults_split",
        type="train-val-dataset",
        description="Processed dataset split into train and valiation",
        metadata={"test_size": test_size, "random_state": random_state},
    )

    # Attach our processed data to the Artifact
    split_ds_art.add_file(train_path)
    split_ds_art.add_file(val_path)

    # Log the Artifact
    run.log_artifact(split_ds_art)
```

#### Inspect Training Dataset
Get an overview of the training dataset


```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### Log Dataset with W&B Tables

With W&B Tables you can log, query, and analyze tabular data that contains rich media such as images, video, audio and more. With it you can understand your datasets, visualize model predictions, and share insights, for more see more in our [W&B Tables Guide](https://docs.wandb.ai/guides/tables)


```python
# Create a wandb run, with an optional "log-dataset" job type to keep things tidy
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)  # config is optional here

# Create a W&B Table and log 1000 random rows of the dataset to explore
table = wandb.Table(dataframe=trndat.sample(1000))

# Log the Table to your W&B workspace
wandb.log({"processed_dataset": table})

# Close the wandb run
wandb.finish()
```

# Modelling

## Fit the XGBoost Model

We will now fit an XGBoost model to classify whether a vehicle loan application will result in a default or not

### Training on GPU
If you'd like to train your XGBoost model on your GPU, simply change set the following in the parameters you pass to XGBoost:

```python
"tree_method": "gpu_hist"
```

#### 1) Initialise a W&B Run


```python
run = wandb.init(project=WANDB_PROJECT, job_type="train-model")
```

#### 2) Setup and Log the Model Parameters


```python
base_rate = round(trndict["base_rate"], 6)
early_stopping_rounds = 40
```


```python
bst_params = {
    "objective": "binary:logistic",
    "base_score": base_rate,
    "gamma": 1,  ## def: 0
    "learning_rate": 0.1,  ## def: 0.1
    "max_depth": 3,
    "min_child_weight": 100,  ## def: 1
    "n_estimators": 25,
    "nthread": 24,
    "random_state": 42,
    "reg_alpha": 0,
    "reg_lambda": 0,  ## def: 1
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",  # use `gpu_hist` to train on GPU
}
```

Log the xgboost training parameters to the W&B run config 


```python
run.config.update(dict(bst_params))
run.config.update({"early_stopping_rounds": early_stopping_rounds})
```

#### 3) Load the Training Data from W&B Artifacts


```python
# Load our training data from Artifacts
trndat, valdat = load_training_data(
    run=run, data_dir=data_dir, artifact_name="vehicle_defaults_split:latest"
)

## Extract target column as a series
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) Fit the model, log results to W&B and save model to W&B Artifacts

To log all our xgboost model parameters we used the `WandbCallback`. This will . See the [W&B docs](https://docs.wandb.ai/guides/integrations), including documentation for other libraries that have integrated W&B including LightGBM and more.


```python
from wandb.integration.xgboost import WandbCallback

# Initialize the XGBoostClassifier with the WandbCallback
xgbmodel = xgb.XGBClassifier(
    **bst_params,
    callbacks=[WandbCallback(log_model=True)],
    early_stopping_rounds=run.config["early_stopping_rounds"]
)

# Train the model
xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])
```

#### 5) Log Additional Train and Evaluation Metrics to W&B


```python
bstr = xgbmodel.get_booster()

# Get train and validation predictions
trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

# Log additional Train metrics
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    y_trn, trnYpreds
)
run.summary["train_ks_stat"] = max(true_positive_rate - false_positive_rate)
run.summary["train_auc"] = metrics.auc(false_positive_rate, true_positive_rate)
run.summary["train_log_loss"] = -(
    y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(1 - trnYpreds)
).sum() / len(y_trn)

# Log additional Validation metrics
ks_stat, ks_pval = ks_2samp(valYpreds[y_val == 1], valYpreds[y_val == 0])
run.summary["val_ks_2samp"] = ks_stat
run.summary["val_ks_pval"] = ks_pval
run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
run.summary["val_acc_0.5"] = metrics.accuracy_score(
    y_val, np.where(valYpreds >= 0.5, 1, 0)
)
run.summary["val_log_loss"] = -(
    y_val * np.log(valYpreds) + (1 - y_val) * np.log(1 - valYpreds)
).sum() / len(y_val)
```

#### 6) Log the ROC Curve To W&B


```python
# Log the ROC curve to W&B
valYpreds_2d = np.array([1 - valYpreds, valYpreds])  # W&B expects a 2d array
y_val_arr = y_val.values
d = 0
while len(valYpreds_2d.T) > 10000:
    d += 1
    valYpreds_2d = valYpreds_2d[::1, ::d]
    y_val_arr = y_val_arr[::d]

run.log(
    {
        "ROC_Curve": wandb.plot.roc_curve(
            y_val_arr,
            valYpreds_2d.T,
            labels=["no_default", "loan_default"],
            classes_to_plot=[1],
        )
    }
)
```

#### Finish the W&B Run


```python
run.finish()
```

Now that we've trained a single model, lets try and optimize its performance by running a Hyperparameter Sweep.

# HyperParameter Sweep

Weights and Biases also enables you to do hyperparameter sweeps, either with our own [Sweeps functionality](https://docs.wandb.ai/guides/sweeps/python-api) or with our [Ray Tune integration](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune). See [our docs](https://docs.wandb.ai/guides/sweeps/python-api) for a full guide of how to use more advanced hyperparameter sweeps options.

**[Click Here](https://wandb.ai/morgan/credit_score_sweeps/sweeps/iuppbs45)** to check out the results of a 1000 run sweep generated using this notebook

#### Define the Sweep Config
First we define the hyperparameters to sweep over as well as the type of sweep to use, we'll do a random search over the learning_rate, gamma, min_child_weights and easrly_stopping_rounds


```python
sweep_config = {
    "method": "random",
    "parameters": {
        "learning_rate": {"min": 0.001, "max": 1.0},
        "gamma": {"min": 0.001, "max": 1.0},
        "min_child_weight": {"min": 1, "max": 150},
        "early_stopping_rounds": {"values": [10, 20, 30, 40]},
    },
}

sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
```

#### Define the Training Function

Then we define the function that will train our model using these hyperparameters. Note that `job_type='sweep'` when initialising the run, so that we can easily filter out these runs from our main workspace if we need to


```python
def train():
    with wandb.init(job_type="sweep") as run:
        bst_params = {
            "objective": "binary:logistic",
            "base_score": base_rate,
            "gamma": run.config["gamma"],
            "learning_rate": run.config["learning_rate"],
            "max_depth": 3,
            "min_child_weight": run.config["min_child_weight"],
            "n_estimators": 25,
            "nthread": 24,
            "random_state": 42,
            "reg_alpha": 0,
            "reg_lambda": 0,  ## def: 1
            "eval_metric": ["auc", "logloss"],
            "tree_method": "hist",
        }

        # Initialize the XGBoostClassifier with the WandbCallback
        xgbmodel = xgb.XGBClassifier(
            **bst_params,
            callbacks=[WandbCallback()],
            early_stopping_rounds=run.config["early_stopping_rounds"]
        )

        # Train the model
        xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])

        bstr = xgbmodel.get_booster()

        # Log booster metrics
        run.summary["best_ntree_limit"] = bstr.best_ntree_limit

        # Get train and validation predictions
        trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
        valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

        # Log additional Train metrics
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
            y_trn, trnYpreds
        )
        run.summary["train_ks_stat"] = max(true_positive_rate - false_positive_rate)
        run.summary["train_auc"] = metrics.auc(false_positive_rate, true_positive_rate)
        run.summary["train_log_loss"] = -(
            y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(1 - trnYpreds)
        ).sum() / len(y_trn)

        # Log additional Validation metrics
        ks_stat, ks_pval = ks_2samp(valYpreds[y_val == 1], valYpreds[y_val == 0])
        run.summary["val_ks_2samp"] = ks_stat
        run.summary["val_ks_pval"] = ks_pval
        run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
        run.summary["val_acc_0.5"] = metrics.accuracy_score(
            y_val, np.where(valYpreds >= 0.5, 1, 0)
        )
        run.summary["val_log_loss"] = -(
            y_val * np.log(valYpreds) + (1 - y_val) * np.log(1 - valYpreds)
        ).sum() / len(y_val)
```

#### Run the Sweeps Agent


```python
count = 10  # number of runs to execute
wandb.agent(sweep_id, function=train, count=count)
```

## W&B already in your favorite ML library

Weights and Biases has integrations in all of your favourite ML and Deep Learning libraries such as:

- Pytorch Lightning
- Keras
- Hugging Face
- JAX
- Fastai
- XGBoost
- Sci-Kit Learn
- LightGBM 

**See [W&B integrations for details](https://docs.wandb.ai/guides/integrations)** 
