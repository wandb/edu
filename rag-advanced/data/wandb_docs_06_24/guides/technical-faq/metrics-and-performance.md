---
displayed_sidebar: default
---

# Metrics & Performance

## Metrics

### How often are system metrics collected?

By default, metrics are collected every 2 seconds and averaged over a 15-second period. If you need higher resolution metrics, email us a [contact@wandb.com](mailto:contact@wandb.com).

### Can I just log metrics, no code or dataset examples?

**Dataset Examples**

By default, we don't log any of your dataset examples. You can explicitly turn this feature on to see example predictions in our web interface.

**Code Logging**

There are two ways to turn off code logging:

1. Set `WANDB_DISABLE_CODE` to `true` to turn off all code tracking. We won't pick up the git SHA or the diff patch.
2. Set `WANDB_IGNORE_GLOBS` to `*.patch` to turn off syncing the diff patch to our servers. You'll still have it locally and be able to apply it with the `wandb restore`.
### Can I log metrics on two different time scales? (For example, I want to log training accuracy per batch and validation accuracy per epoch.)

Yes, you can do this by logging your indices (e.g. `batch` and `epoch`) whenever you log your other metrics. So in one step you could call `wandb.log({'train_accuracy': 0.9, 'batch': 200})` and in another step call `wandb.log({'val_accuracy': 0.8, 'epoch': 4})`. Then, in the UI, you can set the appropriate value as the x-axis for each chart. If you want to set the default x-axis of a particular index you can do so using by using [Run.define\_metric()](../../ref/python/run.md#define_metric). In our above example we could do the following:

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```

### How can I log a metric that doesn't change over time such as a final evaluation accuracy?

Using `wandb.log({'final_accuracy': 0.9}` will work fine for this. By default `wandb.log({'final_accuracy'})` will update `wandb.settings['final_accuracy']`, which is the value shown in the runs table.

### How can I log additional metrics after a run completes?

There are several ways to do this.

For complicated workflows, we recommend using multiple runs and setting group parameters in [`wandb.init`](../track/launch.md) to a unique value in all the processes that are run as part of a single experiment. The [runs table](../app/pages/run-page.md) will automatically group the table by the group ID and the visualizations will behave as expected. This will allow you to run multiple experiments and training runs as separate processes log all the results into a single place.

For simpler workflows, you can call `wandb.init` with `resume=True` and `id=UNIQUE_ID` and then later call `wandb.init` with the same `id=UNIQUE_ID`. Then you can log normally with [`wandb.log`](../track/log/intro.md) or `wandb.summary` and the runs values will update.


## Performance

### Will wandb slow down my training?

W&B should have a negligible effect on your training performance if you use it normally. Normal use of wandb means logging less than once a second and logging less than a few megabytes of data at each step. W&B runs in a separate process and the function calls don't block, so if the network goes down briefly or there are intermittent read write issues on disk it should not affect your performance. It is possible to log a huge amount of data quickly, and if you do that you might create disk I/O issues. If you have any questions, please don't hesitate to contact us.

### How many runs to create per project?

We recommend you have roughly 10k runs per project max for performance reasons.

### Best practices to organize hyperparameter searches

If 10k runs per project (approx.) is a reasonable limit then our recommendation would be to set tags in `wandb.init()` and have a unique tag for each search. This means that you'll easily be able to filter the project down to a given search by clicking that tag in the Project Page in the Runs Table. For example `wandb.init(tags='your_tag')`  docs for this can be found [here](../../ref/python/init.md).
