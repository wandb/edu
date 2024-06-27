---
description: Use a dictionary-like object to save your experiment configuration
displayed_sidebar: default
---

# Configure Experiments

<head>
  <title>Configure a Machine Learning Experiment</title>
</head>

[**Try in a Colab Notebook here**](http://wandb.me/config-colab)

Use the `wandb.config` object to save your training configuration such as: 
- hyperparameters
- input settings such as the dataset name or model type
- any other independent variables for your experiments. 

The `wandb.config` attribute makes it easy to analyze your experiments and reproduce your work in the future. You can group by configuration values in the W&B App, compare the settings of different W&B Runs and view how different training configurations affect the output. A Run's `config` attribute is a dictionary-like object, and it can be built from lots of dictionary-like objects.

:::info
Dependent variables (like loss and accuracy) or output metrics should be saved with `wandb.log`instead.
:::



## Set up an experiment configuration
Configurations are typically defined in the beginning of a training script. Machine learning workflows may vary, however, so you are not required to define a configuration at the beginning of your training script.

:::caution
We recommend that you avoid using dots in your config variable names. Instead, use a dash or underscore instead. Use the dictionary access syntax `["key"]["foo"]` instead of the attribute access syntax `config.key.foo` if your script accesses `wandb.config` keys below the root.
:::


The following sections outline different common scenarios of how to define your experiments configuration.

### Set the configuration at initialization
Pass a dictionary at the beginning of your script when you call the `wandb.init()` API to generate a background process to sync and log data as a W&B Run.

The proceeding code snippet demonstrates how to define a Python dictionary with configuration values and how to pass that dictionary as an argument when you initialize a W&B Run.

```python
import wandb

# Define a config dictionary object
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# Pass the config dictionary when you initialize W&B
run = wandb.init(project="config_example", config=config)
```

:::info
You can pass a nested dictionary to `wandb.config()`. W&B will flatten the names using dots in the W&B backend.
:::

Access the values from the dictionary similarly to how you access other dictionaries in Python:

```python
# Access values with the key as the index value
hidden_layer_sizes = wandb.config["hidden_layer_sizes"]
kernel_sizes = wandb.config["kernel_sizes"]
activation = wandb.config["activation"]

# Python dictionary get() method
hidden_layer_sizes = wandb.config.get("hidden_layer_sizes")
kernel_sizes = wandb.config.get("kernel_sizes")
activation = wandb.config.get("activation")
```

:::note
Throughout the Developer Guide and examples we copy the configuration values into separate variables. This step is optional. It is done for readability.
:::

### Set the configuration with argparse
You can set your configuration with an argparse object. [argparse](https://docs.python.org/3/library/argparse.html), short for argument parser, is a standard library module in Python 3.2 and above that makes it easy to write scripts that take advantage of all the flexibility and power of command line arguments.

This is useful for tracking results from scripts that are launched from the command line.

The proceeding Python script demonstrates how to define a parser object to define and set your experiment config. The functions `train_one_epoch` and `evaluate_one_epoch` are provided to simulate a training loop for the purpose of this demonstration:

```python
# config_experiment.py
import wandb
import argparse
import numpy as np
import random


# Training and evaluation demo code
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # Start a W&B Run
    run = wandb.init(project="config_example", config=args)

    # Access values from config dictionary and store them
    # into variables for readability
    lr = wandb.config["learning_rate"]
    bs = wandb.config["batch_size"]
    epochs = wandb.config["epochs"]

    # Simulate training and logging values to W&B
    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="Learning rate"
    )

    args = parser.parse_args()
    main(args)
```
### Set the configuration throughout your script
You can add more parameters to your config object throughout your script. The proceeding code snippet demonstrates how to add new key-value pairs to your config object:

```python
import wandb

# Define a config dictionary object
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# Pass the config dictionary when you initialize W&B
run = wandb.init(project="config_example", config=config)

# Update config after you initialize W&B
wandb.config["dropout"] = 0.2
wandb.config.epochs = 4
wandb.config["batch_size"] = 32
```
You can update multiple values at a time:

```python
wandb.init(config={"epochs": 4, "batch_size": 32})
# later
wandb.config.update({"lr": 0.1, "channels": 16})
```

### Set the configuration after your Run has finished
Use the [W&B Public API](../../ref/python/public-api/README.md) to update your config (or anything else about from a complete Run) after your Run. This is particularly useful if you forgot to log a value during a Run. 

Provide your `entity`, `project name`, and the `Run ID` to update your configuration after a Run has finished. Find these values directly from the Run object itself `wandb.run` or from the [W&B App UI](../app/intro.md):

```python
api = wandb.Api()

# Access attributes directly from the run object
# or from the W&B App
username = wandb.run.entity
project = wandb.run.project
run_id = wandb.run.id

run = api.run(f"{username}/{project}/{run_id}")
run.config["bar"] = 32
run.update()
```




## `absl.FLAGS`


You can also pass in [`absl` flags](https://abseil.io/docs/python/guides/flags).

```python
flags.DEFINE_string("model", None, "model to run")  # name, default, help

wandb.config.update(flags.FLAGS)  # adds absl flags to config
```

## File-Based Configs
Key-value pairs are automatically passed to `wandb.config` if you create a file called `config-defaults.yaml`.

The proceeding code snippet demonstrates a sample `config-defaults.yaml` YAML file:

```yaml
# config-defaults.yaml
# sample config defaults file
epochs:
  desc: Number of epochs to train over
  value: 100
batch_size:
  desc: Size of each mini-batch
  value: 32
```
You can overwrite automatically passed by a `config-defaults.yaml`. To do so , pass values to the `config` argument of `wandb.init`.

You can also load different config files with the command line argument `--configs`.

### Example use case for file-based configs
Suppose you have a YAML file with some metadata for the run, and then a dictionary of hyperparameters in your Python script. You can save both in the nested `config` object:

```python
hyperparameter_defaults = dict(
    dropout=0.5,
    batch_size=100,
    learning_rate=0.001,
)

config_dictionary = dict(
    yaml=my_yaml_file,
    params=hyperparameter_defaults,
)

wandb.init(config=config_dictionary)
```

## TensorFlow v1 Flags

You can pass TensorFlow flags into the `wandb.config` object directly.

```python
wandb.init()
wandb.config.epochs = 4

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
wandb.config.update(flags.FLAGS)  # add tensorflow flags as config
```

<!-- ## Dataset Identifier

You can add a unique identifier (like a hash or other identifier) in your run's configuration for your dataset by tracking it as input to your experiment using `wandb.config`

```yaml
wandb.config.update({"dataset": "ab131"})
``` -->

<!-- ## Update Config Files

You can use the public API to add values your `config` file, even after the run has finished.

```python
import wandb
api = wandb.Api()
run = api.run("username/project/run_id")
run.config["foo"] = 32
run.update()
``` -->

<!-- ## Key-Value Pairs

You can log any key-value pairs into `wandb.config`. They can be different for every type of model you are training, e.g.`wandb.config.update({"my_param": 10, "learning_rate": 0.3, "model_architecture": "B"})`. -->
