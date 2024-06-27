---
description: Forking a W&B run
displayed_sidebar: default
---

# Fork from a run
:::caution
The ability to fork a run is in private preview. Contact W&B Support at support@wandb.com to request access to this feature.
:::

Use `fork_from` when you initialize a run with [`wandb.init()`](../../ref/python/init.md) to "fork" from an existing W&B run. When you fork from a run, W&B creates a new run using the `run ID` and `step` of the source run.

Forking a run enables you to explore different parameters or models from a specific point in an experiment without impacting the original run.

:::info
* Forking a run requires [`wandb`](https://pypi.org/project/wandb/) SDK version >= 0.16.5
* Forking a run requires monotonically increasing steps. You can not use non-monotonic steps defined with [`define_metric()`](https://docs.wandb.ai/ref/python/run#define_metric) to set a fork point because it would disrupt the essential chronological order of run history and system metrics.
:::


## Start a forked run

To fork a run, use the `fork_from` argument in [`wandb.init()`](../../ref/python/init.md) and specify the source `run ID` and the `step` from the source run to fork from:

```python
import wandb

# Initialize a run to be forked later
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... perform training or logging ...
original_run.finish()

# Fork the run from a specific step
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### Using an immutable run ID

Use an immutable run ID to ensure you have a consistent and unchanging reference to a specific run. Follow these steps to obtain the immutable run ID from the user interface:

1. **Access the Overview Tab:** Navigate to the [**Overview tab**](https://docs.wandb.ai/guides/app/pages/run-page#overview-tab) on the source run's page.

2. **Copy the Immutable Run ID:** Click on the `...` menu (three dots) located in the top-right corner of the **Overview** tab. Select the `Copy Immutable Run ID` option from the dropdown menu.

By following these steps, you will have a stable and unchanging reference to the run, which can be used for forking a run.

## Continue from a forked run
After initializing a forked run, you can continue logging to the new run. You can log the same metrics for continuity and introduce new metrics. 

For example, the following code example shows how to first fork a run and then how to log metrics to the forked run starting from a training step of 200:

```python
import wandb
import math

# Initialize the first run and log some metrics
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# Fork from the first run at a specific step and log the metric starting from step 200
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# Continue logging in the new run
# For the first few steps, log the metric as is from run1
# After step 250, start logging the spikey pattern
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # Continue logging from run1 without spikes
    else:
        # Introduce the spikey behavior starting from step 250
        subtle_spike = i + (2 * math.sin(i / 3.0))  # Apply a subtle spikey pattern
        run2.log({"metric": subtle_spike})
    # Additionally log the new metric at all steps
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

:::tip Rewind and forking compatibility
Forking compliments a [`rewind`](https://docs.wandb.ai/guides/runs/rewind) by providing more flexibility in managing and experimenting with your runs. 

When you fork from a run, W&B creates a new branch off a run at a specific point to try different parameters or models. 

When you  rewind a run, W&B let's you correct or modify the run history itself.
:::
