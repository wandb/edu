---
description: Track your trees with W&B.
displayed_sidebar: default
---

# XGBoost

[**Try in a Colab Notebook here →**](https://wandb.me/xgboost)

The `wandb` library has a `WandbCallback` callback for logging metrics, configs and saved boosters from training with XGBoost. Here you can see a **[live Weights & Biases dashboard](https://wandb.ai/morg/credit_scorecard)** with outputs from the XGBoost `WandbCallback`.

![Weights & Biases dashboard using XGBoost](/images/integrations/xgb_dashboard.png)

## Get Started

Logging XGBoost metrics, configs and booster models to Weights & Biases is as easy as passing the `WandbCallback` to XGBoost:

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# Start a wandb run
run = wandb.init()

# Pass WandbCallback to the model
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# Close your wandb run
run.finish()
```

You can open **[this notebook](https://wandb.me/xgboost)** for a comprehensive look at logging with XGBoost and Weights & Biases

## WandbCallback

### Functionality
Passing `WandbCallback` to a XGBoost model will:
- log the booster model configuration to Weights & Biases
- log evaluation metrics collected by XGBoost, such as rmse, accuracy etc to Weights & Biases
- log training metrics collected by XGBoost (if you provide data to eval_set)
- log the best score and the best iteration
- save and upload your trained model to to Weights & Biases Artifacts (when `log_model = True`)
- log feature importance plot when `log_feature_importance=True` (default).
- Capture the best eval metric in `wandb.summary` when `define_metric=True` (default).

### Arguments
`log_model`: (boolean) if True save and upload the model to Weights & Biases Artifacts

`log_feature_importance`: (boolean) if True log a feature importance bar plot

`importance_type`: (str) one of {weight, gain, cover, total_gain, total_cover} for tree model. weight for linear model.

`define_metric`: (boolean) if True (default) capture model performance at the best step, instead of the last step, of training in your `wandb.summary`.


You can find the source code for WandbCallback [here](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)

:::info
Looking for more working code examples? Check out [our repository of examples on GitHub](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms) or try out a [Colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit\_Scorecards\_with\_XGBoost\_and\_W%26B.ipynb)
:::

## Tuning your hyperparameters with Sweeps

Attaining the maximum performance out of models requires tuning hyperparameters, like tree depth and learning rate. Weights & Biases includes [Sweeps](../sweeps/), a powerful toolkit for configuring, orchestrating, and analyzing large hyperparameter testing experiments.

:::info
To learn more about these tools and see an example of how to use Sweeps with XGBoost, check out [this interactive Colab notebook](http://wandb.me/xgb-sweeps-colab) or try this XGBoost & Sweeps [python script here](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost\_tune.py)
:::

![tl;dr: trees outperform linear learners on this classification dataset.](/images/integrations/xgboost_sweeps_example.png)
