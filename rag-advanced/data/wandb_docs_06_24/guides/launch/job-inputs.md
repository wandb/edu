---
displayed_sidebar: default
---

# Manage job inputs

The core experience of Launch is easily experimenting with different job inputs like hyperparameters and datasets, and routing these jobs to appropriate hardware. Once a job is created, users beyond the original author can adjust these inputs via the W&B GUI or CLI. For information on how job inputs can be set when launching from the CLI or UI, see the [Enqueue jobs](./add-job-to-queue.md) guide.

This section describes how to programmatically control the inputs that can be tweaked for a job.

By default, W&B jobs capture the entire `Run.config` as the inputs to a job, but the Launch SDK provides a function to control select keys in the run config or to specify JSON or YAML files as inputs.


:::info
Launch SDK functions require `wandb-core`. See the [`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md) for more information.
:::

## Reconfigure the `Run` object

The `Run` object returned by `wandb.init` in a job can be reconfigured, by default. The Launch SDK provides a way to customize what parts of the `Run.config` object can be reconfigured when launching the job.


```python
import wandb
from wandb.sdk import launch

# Required for launch sdk use.
wandb.require("core")

config = {
    "trainer": {
        "learning_rate": 0.01,
        "batch_size": 32,
        "model": "resnet",
        "dataset": "cifar10",
        "private": {
            "key": "value",
        },
    },
    "seed": 42,
}


with wandb.init(config=config):
    launch.manage_wandb_config(
        include=["trainer"], 
        exclude=["trainer.private"],
    )
    # Etc.
```

The function `launch.manage_wandb_config` configures the job to accept input values for the `Run.config` object.  The optional `include` and `exclude` options take path prefixes within the nested config object.  This can be useful if, for example, a job uses a library whose options you don't want to expose to end users.  

If `include` prefixes are provided, only paths within the config that match an `include` prefix will accept input values. If `exclude` prefixes are provided, no paths that match the `exclude` list will be filtered out of the input values. If a path matches both an `include` and an `exclude` prefix, the `exclude` prefix will take precedence.

In the preceding example, the path `["trainer.private"]` will filter out the `private` key from the `trainer` object, and the path `["trainer"]` will filter out all keys not under the `trainer` object.

:::tip
Use a `\`-escaped `.` to filter out keys with a `.` in their name. 

For example, `r"trainer\.private"` filters out the `trainer.private` key rather than the `private` key under the `trainer` object.

Note that the `r` prefix above denotes a raw string.
:::

If the code above is packaged and run as a job, the input types of the job will be:

```json
{
    "trainer": {
        "learning_rate": "float",
        "batch_size": "int",
        "model": "str",
        "dataset": "str",
    },
}
```

When launching the job from the W&B CLI or UI, the user will be able to override only the four `trainer` parameters.

### Access run config inputs

Jobs launched with run config inputs can access the input values through the `Run.config`. The `Run` returned by `wandb.init` in the job code will have the input values automatically set. Use 
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```
to load the run config input values anywhere in the job code.

## Reconfigure a file

The Launch SDK also provides a way to manage input values stored in config files in the job code. This is a common pattern in many deep learning and large language model use cases, like this [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) example or this [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml)). 

:::info
[Sweeps on Launch](./sweeps-on-launch.md) does not support the use of config file inputs as sweep parameters. Sweep parameters must be controlled through the `Run.config` object.
:::

The `launch.manage_config_file` function can be used to add a config file as an input to the Launch job, giving you access to edit values within the config file when launching the job.

By default, no run config inputs will be captured if `launch.manage_config_file` is used. Calling `launch.manage_wandb_config` overrides this behavior.

Consider the following example:

```python
import yaml
import wandb
from wandb.sdk import launch

# Required for launch sdk use.
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # Etc.
    pass
```

Imagine the code is run with an adjacent file `config.yaml`:

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

The call to `launch.manage_config_file` will add the `config.yaml` file as an input to the job, making it reconfigurable when launching from the W&B CLI or UI. 

The `include` and `exclude` keyword arugments may be used to filter the acceptable input keys for the config file in the same way as `launch.manage_wandb_config`.


### Access config file inputs

When `launch.manage_config_file` is called in a run created by Launch, `launch` patches the contents of the config file with the input values. The patched config file is available in the job environment.

:::important
Call `launch.manage_config_file` before reading the config file in the job code to ensure input values are used.
:::