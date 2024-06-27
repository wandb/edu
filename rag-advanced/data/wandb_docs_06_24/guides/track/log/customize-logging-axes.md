---
displayed_sidebar: default
---

# Customize Log Axes

Use `define_metric` to set a **custom x axis**.Custom x-axes are useful in contexts where you need to log to different time steps in the past during training, asynchronously. For example, this can be useful in RL where you may track the per-episode reward and a per-step reward.

[Try `define_metric` live in Google Colab →](http://wandb.me/define-metric-colab)

### Customize axes

By default, all metrics are logged against the same x-axis, which is the W&B internal `step`. Sometimes, you might want to log to a previous step, or use a different x-axis.

Here's an example of setting a custom x-axis metric, instead of the default step.

```python
import wandb

wandb.init()
# define our custom x axis metric
wandb.define_metric("custom_step")
# define which metrics will be plotted against it
wandb.define_metric("validation_loss", step_metric="custom_step")

for i in range(10):
    log_dict = {
        "train_loss": 1 / (i + 1),
        "custom_step": i**2,
        "validation_loss": 1 / (i + 1),
    }
    wandb.log(log_dict)
```

The x-axis can be set using globs as well. Currently, only globs that have string prefixes are available. The following example will plot all logged metrics with the prefix `"train/"` to the x-axis `"train/step"`:

```python
import wandb

wandb.init()
# define our custom x axis metric
wandb.define_metric("train/step")
# set all other train/ metrics to use this step
wandb.define_metric("train/*", step_metric="train/step")

for i in range(10):
    log_dict = {
        "train/step": 2**i,  # exponential growth w/ internal W&B step
        "train/loss": 1 / (i + 1),  # x-axis is train/step
        "train/accuracy": 1 - (1 / (1 + i)),  # x-axis is train/step
        "val/loss": 1 / (1 + i),  # x-axis is internal wandb step
    }
    wandb.log(log_dict)
```
