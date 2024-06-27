---
displayed_sidebar: default
---

# Log Summary Metrics

In addition to values that change over time during training, it is often important to track a single value that summarizes a model or a preprocessing step. Log this information in a W&B Run's `summary` dictionary. A Run's summary dictionary can handle numpy arrays, PyTorch tensors or TensorFlow tensors. When a value is one of these types we persist the entire tensor in a binary file and store high level metrics in the summary object such as min, mean, variance, 95th percentile, etc.

 The last value logged with `wandb.log` is automatically set as the summary dictionary in a W&B Run. If a summary metric dictionary is modified, the previous value is lost.

The proceeding code snippet demonstrates how to provide a custom summary metric to W&B:
```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.run.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

You can update the summary attribute of an existing W&B Run after training has completed. Use the [W&B Public API](../../../ref/python/public-api/README.md) to update the summary attribute:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## Customize summary metrics

Custom metric summaries are useful to capture model performance at the best step, instead of the last step, of training in your `wandb.summary`. For example, you might want to capture the maximum accuracy or the minimum loss value, instead of the final value.

Summary metrics can be controlled using the `summary` argument in `define_metric` which accepts the following values: `"min"`, `"max"`, `"mean"` ,`"best"`, `"last"` and `"none"`. The `"best"` parameter can only be used in conjunction with the optional `objective` argument which accepts values `"minimize"` and `"maximize"`. Here's an example of capturing the lowest value of loss and the maximum value of accuracy in the summary, instead of the default summary behavior, which uses the final value from history.

```python
import wandb
import random

random.seed(1)
wandb.init()
# define a metric we are interested in the minimum of
wandb.define_metric("loss", summary="min")
# define a metric we are interested in the maximum of
wandb.define_metric("acc", summary="max")
for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

Here's what the resulting min and max summary values look like, in pinned columns in the sidebar on the Project Page workspace:

![Project Page Sidebar](/images/track/customize_sumary.png)
