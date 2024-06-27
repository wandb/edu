# LightGBM

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb)

Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

Gradient boosting decision trees are the state of the art when it comes to building predictive models for structured data.

[LigthGBM](https://github.com/microsoft/LightGBM), a gradient boosting framework by Microsoft, has dethroned xgboost and become the go to GBDT algorithm (along with catboost). It outperforms xgboost in training speeds, memory usage and the size of datasets it can handle. LightGBM does so by using histogram-based algorithms to bucket continuous features into discrete bins during training.

You can find the **[W&B + LightGBM documentation here](https://docs.wandb.ai/guides/integrations/boosting)** 


## What this notebook covers
* Easy integration of Weights and Biases with LightGBM. 
* `wandb_callback()` callback for metrics logging
* `log_summary()` function to log a feature importance plot and enable model saving to W&B

We want to make it incredible easy for people to look under the hood of their models, so we built a callback that helps you visualize your LightGBM’s performance in just one line of code.

**Note**: Sections starting with _Step_ is all you need to integrate W&B.

# Install, Import, and Log in

## The Usual Suspects


```ipython
!pip install -Uq 'lightgbm>=3.3.1'
```


```python
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
```

## Step 0: Install W&B


```ipython
!pip install -qU wandb
```

## Step 1: Import W&B and Login


```python
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary

wandb.login()
```

# Download and Prepare Dataset



```ipython
!wget https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.train -qq
!wget https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.test -qq
```


```python
# load or create your dataset
df_train = pd.read_csv("regression.train", header=None, sep="\t")
df_test = pd.read_csv("regression.test", header=None, sep="\t")

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
```

# Train

### Step 2: Initialize your wandb run. 

Using `wandb.init()` initialize your W&B run. You can also pass a dictionary of configs. [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/init)

You can't deny the importance of configs in your ML/DL workflow. W&B makes sure that you have access to the right config to reproduce your model. 

[Learn more about configs in this colab notebook $\rightarrow$](http://wandb.me/config-colab)


```python
# specify your configurations as a dict
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["rmse", "l2", "l1", "huber"],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": 0,
}

wandb.init(project="my-lightgbm-project", config=params)
```

> Once you have trained your model come back and click on the **Project page**.

### Step 3: Train with `wandb_callback`


```python
# train
# add lightgbm callback
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=30,
    valid_sets=lgb_eval,
    valid_names=("validation"),
    callbacks=[wandb_callback()],
    early_stopping_rounds=5,
)
```

### Step 4: Log Feature Importance and Upload Model with `log_summary`
`log_summary` will upload calculate and upload the feature importance import and (optionally) upload your trained model to W&B Artifacts so you can use it later


```python
log_summary(gbm, save_model_checkpoint=True)
```

# Evaluate


```python
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval
print("The rmse of prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)
wandb.log({"rmse_prediction": mean_squared_error(y_test, y_pred) ** 0.5})
```

When you are finished logging for a particular W&B run its a good idea to call `wandb.finish()` to tidy up the wandb process (only necessary when using notebooks/colabs)


```python
wandb.finish()
```

# Visualize Results

Click on the **project page** link above to see your results automatically visualized.

<img src="https://imgur.com/S6lwSig.png" alt="Viz" />


# Sweep 101

Use Weights & Biases Sweeps to automate hyperparameter optimization and explore the space of possible models.

## [Check out Hyperparameter Optimization with XGBoost  using W&B Sweep $\rightarrow$](http://wandb.me/xgb-colab)

Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

1. **Define the sweep:** We do this by creating a dictionary or a [YAML file](https://docs.wandb.com/library/sweeps/configuration) that specifies the parameters to search through, the search strategy, the optimization metric et all.

2. **Initialize the sweep:** 
`sweep_id = wandb.sweep(sweep_config)`

3. **Run the sweep agent:** 
`wandb.agent(sweep_id, function=train)`

And voila! That's all there is to running a hyperparameter sweep!

<img src="https://imgur.com/SVtMfa2.png" alt="Sweep Result" />


# Example Gallery

See examples of projects tracked and visualized with W&B in our [Gallery →](https://app.wandb.ai/gallery)

# Basic Setup
1. **Projects**: Log multiple runs to a project to compare them. `wandb.init(project="project-name")`
2. **Groups**: For multiple processes or cross validation folds, log each process as a runs and group them together. `wandb.init(group='experiment-1')`
3. **Tags**: Add tags to track your current baseline or production model.
4. **Notes**: Type notes in the table to track the changes between runs.
5. **Reports**: Take quick notes on progress to share with colleagues and make dashboards and snapshots of your ML projects.

# Advanced Setup
1. [Environment variables](https://docs.wandb.com/library/environment-variables): Set API keys in environment variables so you can run training on a managed cluster.
2. [Offline mode](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): Use `dryrun` mode to train offline and sync results later.
3. [On-prem](https://docs.wandb.com/self-hosted): Install W&B in a private cloud or air-gapped servers in your own infrastructure. We have local installations for everyone from academics to enterprise teams.
4. [Sweeps](https://docs.wandb.com/sweeps): Set up hyperparameter search quickly with our lightweight tool for tuning.
