---
displayed_sidebar: default
---

# PyTorch

[**Try in a Colab Notebook here →**](http://wandb.me/intro)

PyTorch is one of the most popular frameworks for deep learning in Python, especially among researchers. W&B provides first class support for PyTorch, from logging gradients to profiling your code on the CPU and GPU.

:::info
Try our integration out in a [colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple\_PyTorch\_Integration.ipynb) (with video walkthrough below) or see our [example repo](https://github.com/wandb/examples) for scripts, including one on hyperparameter optimization using [Hyperband](https://arxiv.org/abs/1603.06560) on [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion), plus the [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) it generates.
:::

<!-- {% embed url="https://www.youtube.com/watch?v=G7GH0SeNBMA" %}
Follow along with a video tutorial!
{% endembed %} -->

## Logging gradients with `wandb.watch`

To automatically log gradients, you can call [`wandb.watch`](../../ref/python/watch.md) and pass in your PyTorch model.

```python
import wandb

wandb.init(config=args)

model = ...  # set up your model

# Magic
wandb.watch(model, log_freq=100)

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        wandb.log({"loss": loss})
```

If you need to track multiple models in the same script, you can call `wandb.watch` on each model separately. Reference documentation for this function is [here](../../ref/python/watch.md).

:::caution
Gradients, metrics and the graph won't be logged until `wandb.log` is called after a forward _and_ backward pass.
:::

## Logging images and media

You can pass PyTorch `Tensors` with image data into [`wandb.Image`](../../ref/python/data-types/image.md) and utilities from [`torchvision`](https://pytorch.org/vision/stable/index.html) will be used to convert them to images automatically:

```python
images_t = ...  # generate or load images as PyTorch Tensors
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

For more on logging rich media to W&B in PyTorch and other frameworks, check out our [media logging guide](../track/log/media.md).

If you also want to include information alongside media, like your model's predictions or derived metrics, use a `wandb.Table`.

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# Log your Table to W&B
wandb.log({"mnist_predictions": my_table})
```

![The code above generates a table like this one. This model's looking good!](/images/integrations/pytorch_example_table.png)

For more on logging and visualizing datasets and models, check out our [guide to W&B Tables](../tables/intro.md).

## Profiling PyTorch code

![View detailed traces of PyTorch code execution inside W&B dashboards.](/images/integrations/pytorch_example_dashboard.png)

W&B integrates directly with [PyTorch Kineto](https://github.com/pytorch/kineto)'s [Tensorboard plugin](https://github.com/pytorch/kineto/blob/master/tb\_plugin/README.md) to provide tools for profiling PyTorch code, inspecting the details of CPU and GPU communication, and identifying bottlenecks and optimizations.

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # see the profiler docs for details on scheduling
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # run the code you want to profile here
    # see the profiler docs for detailed usage information

# create a wandb Artifact
profile_art = wandb.Artifact("trace", type="profile")
# add the pt.trace.json files to the Artifact
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# log the artifact
profile_art.save()
```

See and run working example code in [this Colab](http://wandb.me/trace-colab).

:::caution
The interactive trace viewing tool is based on the Chrome Trace Viewer, which works best with the Chrome browser.
:::
