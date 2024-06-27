---
displayed_sidebar: default
---

# TensorBoard

## Hosted TensorBoard with 1 Line of Code

With Weight & Biases you can easily upload your TensorBoard logs to the cloud, quickly share your results among colleagues and classmates and keep your analysis in one centralized location.

**Get started now in with this Notebook:** [**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard\_and\_Weights\_and\_Biases.ipynb)


![](/images/integrations/tensorboard_oneline_code.webp)


### Just add 1 Line of Code

```python
import wandb

# Start a wandb run with `sync_tensorboard=True`
wandb.init(project="my-project", sync_tensorboard=True)

# Your training code using TensorBoard
...

# [Optional]Finish the wandb run to upload the tensorboard logs to W&B (if running in Notebook)
wandb.finish()
```

[**See here for an example of Tensorboard hosted in Weights & Biases**](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)

Once your wandb run finishes, your TensorBoard event files will then be uploaded to Weights & Biases. These metrics will **also be logged** in native Weights & Biases charts along with a host of useful information such as your machines CPU or GPU utilization, the git state, the terminal command used, and much more.

:::info
Weights & Biases support TensorBoard with all versions of TensorFlow. W&B also supports TensorBoard > 1.14 with PyTorch as well as TensorBoardX.
:::

## Common questions

### How can I log metrics to W&B that aren't logged to TensorBoard?

If you need to log additional custom metrics that aren't being logged to TensorBoard, you can call `wandb.log` in your code `wandb.log({"custom": 0.8})`

Setting the step argument in `wandb.log` is disabled when syncing Tensorboard. If you'd like to set a different step count, you can log the metrics with a step metric as:

`wandb.log({"custom": 0.8, "global_step": global_step})`

### How do I configure Tensorboard when I'm using it with `wandb`?

If you want more control over how TensorBoard is patched you can call `wandb.tensorboard.patch` instead of passing `sync_tensorboard=True` to `wandb.init`.

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# Finish the wandb run to upload the tensorboard logs to W&B (if running in Notebook)
wandb.finish()
```

You can pass `tensorboard_x=False` to this method to ensure vanilla TensorBoard is patched, if you're using TensorBoard > 1.14 with PyTorch you can pass `pytorch=True` to ensure it's patched. Both of these options have smart defaults depending on what versions of these libraries have been imported.

By default, we also sync the `tfevents` files and any `.pbtxt` files. This enables us to launch a TensorBoard instance on your behalf. You will see a [TensorBoard tab](https://www.wandb.com/articles/hosted-tensorboard) on the run page. This behavior can be disabled by passing `save=False` to `wandb.tensorboard.patch`

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# If running in a notebook, finish the wandb run to upload the tensorboard logs to W&B
wandb.finish()
```

:::caution
You must call either `wandb.init` or `wandb.tensorboard.patch` **before** calling `tf.summary.create_file_writer` or constructing a `SummaryWriter` via `torch.utils.tensorboard`.
:::

### Syncing Previous TensorBoard Runs

If you have existing `tfevents` files stored locally and you would like to import them into W&B, you can run `wandb sync log_dir`, where `log_dir` is a local directory containing the `tfevents` files.

### Google Colab, Jupyter and TensorBoard

If running your code in a Jupyter or Colab notebook, make sure to call `wandb.finish()` and the end of your training. This will finish the wandb run and upload the tensorboard logs to W&B so they can be visualized. This is not necessary when running a `.py` script as wandb finishes automatically when a script finishes.

To run shell commands in a notebook environment, you must prepend a `!`, as in `!wandb sync directoryname`.

### PyTorch and TensorBoard

If you use PyTorch's TensorBoard integration, you may need to manually upload the PyTorch Profiler JSON file**:**

```
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```
