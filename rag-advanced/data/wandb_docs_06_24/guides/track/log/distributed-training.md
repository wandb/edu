---
description: Use W&B to log distributed training experiments with multiple GPUs.
displayed_sidebar: default
---

# Log distributed training experiments

<head>
  <title>Log distributed training experiments</title>
</head>


In distributed training, models are trained using multiple GPUs in parallel. W&B supports two patterns to track distributed training experiments:

1. **One process**: Initialize W&B ([`wandb.init`](../../../ref//python/init.md)) and log experiments ([`wandb.log`](../../../ref//python/log.md)) from a single process. This is a common solution for logging distributed training experiments with the [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) Class. In some cases, users funnel data over from other processes using a multiprocessing queue (or another communication primitive) to the main logging process.
2. **Many processes**: Initialize W&B ([`wandb.init`](../../../ref//python/init.md)) and log experiments ([`wandb.log`](../../../ref//python/log.md)) in every process. Each process is effectively a separate experiment. Use the `group` parameter when you initialize W&B (`wandb.init(group='group-name')`) to define a shared experiment and group the logged values together in the W&B App UI.

The proceeding examples demonstrate how to track metrics with W&B using PyTorch DDP on two GPUs on a single machine. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp\_tutorial.html) (`DistributedDataParallel` in`torch.nn`) is a popular library for distributed training. The basic principles apply to any distributed training setup, but the details of implementation may differ.

:::info
Explore the code behind these examples in the W&B GitHub examples repository [here](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp). Specifically, see the [`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Python script for information on how to implement one process and many process methods.
:::

### Method 1: One process

In this method we track only a rank 0 process. To implement this method, initialize W&B (`wandb.init)`, commence a W&B Run, and log metrics (`wandb.log`) within the rank 0 process. This method is simple and robust, however, this method does not log model metrics from other processes (for example, loss values or inputs from their batches). System metrics, such as usage and memory, are still logged for all GPUs since that information is available to all processes.

:::info
**Use this method to only track metrics available from a single process**. Typical examples include GPU/CPU utilization, behavior on a shared validation set, gradients and parameters, and loss values on representative data examples.
:::

Within our [sample Python script (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py), we check to see if the rank is 0. To do so, we first launch multiple processes with `torch.distributed.launch`. Next, we check the rank with the `--local_rank` command line argument. If the rank is set to 0, we set up `wandb` logging conditionally in the [`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) function. Within our Python script, we use the following check:

```python showLineNumbers
if __name__ == "__main__":
    # Get args
    args = parse_args()

    if args.local_rank == 0:  # only on main process
        # Initialize wandb run
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # Train model with DDP
        train(args, run)
    else:
        train(args)
```

Explore the W&B App UI to view an [example dashboard](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system) of metrics tracked from a single process. The dashboard displays system metrics such as temperature and utilization, that were tracked for both GPUs.

![](/images/track/distributed_training_method1.png)

However, the loss values as a function epoch and batch size were only logged from a single GPU.

![](/images/experiments/loss_function_single_gpu.png)

### Method 2: Many processes

In this method, we track each process in the job, calling `wandb.init()` and `wandb.log()` from each process separately. We suggest you call `wandb.finish()` at the end of training, to mark that the run has completed so that all processes exit properly.

This method makes more information accessible for logging. However, note that multiple W&B Runs are reported in the W&B App UI. It might be difficult to keep track of W&B Runs across multiple experiments. To mitigate this, provide a value to the group parameter when you initialize W&B to keep track of which W&B Run belongs to a given experiment. For more information about how to keep track of training and evaluation W&B Runs in experiments, see [Group Runs](../../runs/grouping.md).

:::info
**Use this method if you want to track metrics from individual processes**. Typical examples include the data and predictions on each node (for debugging data distribution) and metrics on individual batches outside of the main node. This method is not necessary to get system metrics from all nodes nor to get summary statistics available on the main node.
:::

The following Python code snippet demonstrates how to set the group parameter when you initialize W&B:

```python
if __name__ == "__main__":
    # Get args
    args = parse_args()
    # Initialize run
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # all runs for the experiment in one group
    )
    # Train model with DDP
    train(args, run)
```

Explore the W&B App UI to view an [example dashboard](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna) of metrics tracked from multiple processes. Note that there are two W&B Runs grouped together in the left sidebar. Click on a group to view the dedicated group page for the experiment. The dedicated group page displays metrics from each process separately.

![](/images/experiments/dashboard_grouped_runs.png)

The preceding image demonstrates the W&B App UI dashboard. On the sidebar we see two experiments. One labeled 'null' and a second (bound by a yellow box) called 'DPP'. If you expand the group (select the Group dropdown) you will see the W&B Runs that are associated to that experiment.

### Use W&B Service to avoid common distributed training issues.

There are two common issues you might encounter when using W&B and distributed training:

1. **Hanging at the beginning of training** - A `wandb` process can hang if the `wandb` multiprocessing interferes with the multiprocessing from distributed training.
2. **Hanging at the end of training** - A training job might hang if the `wandb` process does not know when it needs to exit. Call the `wandb.finish()` API at the end of your Python script to tell W&B that the Run finished. The wandb.finish() API will finish uploading data and will cause W&B to exit.

We recommend using the `wandb service` to improve the reliability of your distributed jobs. Both of the preceding training issues are commonly found in versions of the W&B SDK where wandb service is unavailable.

### Enable W&B Service

Depending on your version of the W&B SDK, you might already have W&B Service enabled by default.

#### W&B SDK 0.13.0 and above

W&B Service is enabled by default for versions of the W&B SDK `0.13.0` and above.

#### W&B SDK 0.12.5 and above

Modify your Python script to enable W&B Service for W&B SDK version 0.12.5 and above. Use the `wandb.require` method and pass the string `"service"` within your main function:

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # rest-of-your-script-goes-here
```

For optimal experience we do recommend you upgrade to the latest version.

**W&B SDK 0.12.4 and below**

Set the `WANDB_START_METHOD` environment variable to `"thread"` to use multithreading instead if you use a W&B SDK version 0.12.4 and below.

### Example use cases for multiprocessing

The following code snippets demonstrate common methods for advanced distributed use cases.

#### Spawn process

Use the `wandb.setup()[line 8]`method in your main function if you initiate a W&B Run in a spawned process:

```python showLineNumbers
import multiprocessing as mp


def do_work(n):
    run = wandb.init(config=dict(n=n))
    run.log(dict(this=n * n))


def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))


if __name__ == "__main__":
    main()
```

#### Share a W&B Run

Pass a W&B Run object as an argument to share W&B Runs between processes:

```python showLineNumbers
def do_work(run):
    run.log(dict(this=1))


def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()


if __name__ == "__main__":
    main()
```


:::info
Note that we can not guarantee the logging order. Synchronization should be done by the author of the script.
:::
