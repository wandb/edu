---
description: >-
  Search and stop algorithms locally instead of using the W&B
  cloud-hosted service.
displayed_sidebar: default
---

# Search and stop algorithms locally

<head>
  <title>Search and stop algorithms locally with W&B agents</title>
</head>

The hyper-parameter controller is hosted by Weights & Biased as a cloud service by default. W&B agents communicate with the controller to determine the next set of parameters to use for training. The controller is also responsible for running early stopping algorithms to determine which runs can be stopped.

The local controller feature allows the user to commence search and stop algorithms locally. The local controller gives the user the ability to inspect and instrument the code in order to debug issues as well as develop new features which can be incorporated into the cloud service.

:::caution
This feature is offered to support faster development and debugging of new algorithms for the Sweeps tool. It is not intended for actual hyperparameter optimization workloads.
:::

Before you get start, you must install the W&B SDK(`wandb`). Type the following code snippet into your command line:

```
pip install wandb sweeps 
```

The following examples assume you already have a configuration file and a training loop defined in a python script or Jupyter Notebook. For more information about how to define a configuration file, see [Define sweep configuration](./define-sweep-configuration.md).

### Run the local controller from the command line

Initialize a sweep similarly to how you normally would when you use hyper-parameter controllers hosted by W&B as a cloud service. Specify the controller flag (`controller`) to indicate you want to use the local controller for W&B sweep jobs:

```bash
wandb sweep --controller config.yaml
```

Alternatively, you can separate initializing a sweep and specifying that you want to use a local controller into two steps.

To separate the steps, first add the following key-value to your sweep's YAML configuration file:

```yaml
controller:
  type: local
```

Next, initialize the sweep:

```bash
wandb sweep config.yaml
```

After you initialized the sweep, start a controller with [`wandb controller`](../../ref/python/controller.md):

```bash
# wandb sweep command will print a sweep_id
wandb controller {user}/{entity}/{sweep_id}
```

Once you have specified you want to use a local controller, start one or more Sweep agents to execute the sweep. Start a W&B Sweep similar to how you normally would. See [Start sweep agents](../../guides/sweeps/start-sweep-agents.md), for more information.

```bash
wandb sweep sweep_ID
```

### Run a local controller with W&B Python SDK

The following code snippets demonstrate how to specify and use a local controller with the W&B Python SDK.

The simplest way to use a controller with the Python SDK is to pass the sweep ID to the [`wandb.controller`](../../ref/python/controller.md) method. Next, use the return objects `run` method to start the sweep job:

```python
sweep = wandb.controller(sweep_id)
sweep.run()
```

If you want more control of the controller loop:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)
```

Or even more control over the parameters served:

```python
import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()
```

If you want to specify your sweep entirely with code you can do something like this:

```python
import wandb

sweep = wandb.controller()
sweep.configure_search("grid")
sweep.configure_program("train-dummy.py")
sweep.configure_controller(type="local")
sweep.configure_parameter("param1", value=3)
sweep.create()
sweep.run()
```
