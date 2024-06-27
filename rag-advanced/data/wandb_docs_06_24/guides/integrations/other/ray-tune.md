---
slug: /guides/integrations/ray-tune
description: How to integrate W&B with Ray Tune.
displayed_sidebar: default
---

# Ray Tune

W&B integrates with [Ray](https://github.com/ray-project/ray) by offering two lightweight integrations.

One is the `WandbLoggerCallback`, which automatically logs metrics reported to Tune to the Wandb API. The other one is the `@wandb_mixin` decorator, which can be used with the function API. It automatically initializes the Wandb API with Tune’s training information. You can just use the Wandb API like you would normally do, e.g. using `wandb.log()` to log your training process.

## WandbLoggerCallback

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb configuration is done by passing a wandb key to the config parameter of `tune.run()` (see example below).

The content of the wandb config entry is passed to `wandb.init()` as keyword arguments. The exception are the following settings, which are used to configure the `WandbLoggerCallback` itself:

### Parameters

`api_key_file (str)` – Path to file containing the `Wandb API KEY`.

`api_key (str)` – Wandb API Key. Alternative to setting `api_key_file`.

`excludes (list)` – List of metrics that should be excluded from the `log`.

`log_config (bool)` – Boolean indicating if the config parameter of the results dict should be logged. This makes sense if parameters will change during training, e.g. with `PopulationBasedTraining`. Defaults to False.

### Example

```python
from ray import tune, train
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback

def train_fc(config):
    for i in range(10):
        train.report({"mean_accuracy":(i + config['alpha']) / 10})

search_space = {
    'alpha': tune.grid_search([0.1, 0.2, 0.3]),
    'beta': tune.uniform(0.5, 1.0)
}

analysis = tune.run(
    train_fc,
    config=search_space,
    callbacks=[WandbLoggerCallback(
        project="<your-project>",
        api_key="<your-name>",
        log_config=True
    )]
)

best_trial = analysis.get_best_trial("mean_accuracy", "max", "last")
```

## wandb\_mixin

```python
ray.tune.integration.wandb.wandb_mixin(func)
```

This Ray Tune Trainable `mixin` helps initializing the Wandb API for use with the `Trainable` class or with `@wandb_mixin` for the function API.

For basic usage, just prepend your training function with the `@wandb_mixin` decorator:

```python
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_fn(config):
    wandb.log()
```

Wandb configuration is done by passing a `wandb key` to the `config` parameter of `tune.run()` (see example below).

The content of the wandb config entry is passed to `wandb.init()` as keyword arguments. The exception are the following settings, which are used to configure the `WandbTrainableMixin` itself:

### Parameters

`api_key_file (str)` – Path to file containing the Wandb `API KEY`.

`api_key (str)` – Wandb API Key. Alternative to setting `api_key_file`.

Wandb’s `group`, `run_id` and `run_name` are automatically selected by Tune, but can be overwritten by filling out the respective configuration values.

Please see here for all other valid configuration settings: [https://docs.wandb.com/library/init](https://docs.wandb.com/library/init)

### Example:

```python
from ray import tune
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_fn(config):
    for i in range(10):
        loss = self.config["a"] + self.config["b"]
        wandb.log({"loss": loss})
        tune.report(loss=loss)


tune.run(
    train_fn,
    config={
        # define search space here
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb configuration
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
```

## Example Code

We've created a few examples for you to see how the integration works:

* [Colab](http://wandb.me/raytune-colab): A simple demo to try the integration.
* [Dashboard](https://wandb.ai/anmolmann/ray\_tune): View dashboard generated from the example.
