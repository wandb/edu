---
description: Troubleshoot common W&B Sweep issues.
displayed_sidebar: default
---

# Troubleshoot Sweeps

<head>
  <title>Troubleshoot W&B Sweeps</title>
</head>

Troubleshoot common error messages with the guidance suggested.

### `CommError, Run does not exist` and `ERROR Error uploading`

Your W&B Run ID might be defined if these two error messages are both returned. As an example, you might have a similar code snippet defined somewhere in your Jupyter Notebooks or Python script:

```python
wandb.init(id="some-string")
```

You can not set a Run ID for W&B Sweeps because W&B automatically generates random, unique IDs for Runs created by W&B Sweeps.

W&B Run IDs need to be unique within a project.

We recommend you pass a name to the name parameter when you initialized W&B, if you want to set a custom name that will appear on tables and graphs. For example:

```python
wandb.init(name="a helpful readable run name")
```

### `Cuda out of memory`

Refactor your code to use process-based executions if you see this error message. More specifically, rewrite your code to a Python script. In addition, call the W&B Sweep Agent from the CLI, instead of the W&B Python SDK.

As an example, suppose you rewrite your code to a Python script called  `train.py`. Add the name of the training script (`train.py`) to your YAML Sweep configuration file (`config.yaml` in this example):

```yaml
program: train.py
method: bayes
metric:
  name: validation_loss
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
```

Next, add the following to your `train.py` Python script:

```python
if _name_ == "_main_":
    train()
```

Navigate to your CLI and initialize a W&B Sweep with wandb sweep:

```shell
wandb sweep config.yaml
```

Make a note of the W&B Sweep ID that is returned. Next, start the Sweep job with [`wandb agent`](../../ref/cli/wandb-agent.md) with the CLI instead of the Python SDK ([`wandb.agent`](../../ref/python/agent.md)). Replace `sweep_ID` in the code snippet below with the Sweep ID that was returned in the previous step:

```shell
wandb agent sweep_ID
```

### `anaconda 400 error`

The following error usually occurs when you do not log the metric that you are optimizing:

```shell
wandb: ERROR Error while calling W&B API: anaconda 400 error: 
{"code": 400, "message": "TypeError: bad operand type for unary -: 'NoneType'"}
```

Within your YAML file or nested dictionary you specify a key named "metric" to optimize. Ensure that you log (`wandb.log`) this metric. In addition, ensure you use the _exact_ metric name that you defined the sweep to optimize within your Python script or Jupyter Notebook. For more information about configuration files, see [Define sweep configuration](./define-sweep-configuration.md).
