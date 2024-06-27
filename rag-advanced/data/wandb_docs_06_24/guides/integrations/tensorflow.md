---
displayed_sidebar: default
---

# TensorFlow

If you're already using TensorBoard, it's easy to integrate with wandb.

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## Custom Metrics

If you need to log additional custom metrics that aren't being logged to TensorBoard, you can call `wandb.log` in your code `wandb.log({"custom": 0.8}) `

Setting the step argument in `wandb.log` is disabled when syncing Tensorboard. If you'd like to set a different step count, you can log the metrics with a step metric as:

`wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)`

## TensorFlow Hook

If you want more control over what get's logged, wandb also provides a hook for TensorFlow estimators. It will log all `tf.summary` values in the graph.

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## Manual Logging

The simplest way to log metrics in TensorFlow is by logging `tf.summary` with the TensorFlow logger:

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

With TensorFlow 2, the recommended way of training a model with a custom loop is via using `tf.GradientTape`. You can read more about it [here](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough). If you want to incorporate `wandb` to log metrics in your custom TensorFlow training loops you can follow this snippet -

```python
    with tf.GradientTape() as tape:
        # Get the probabilities
        predictions = model(features)
        # Calculate the loss
        loss = loss_func(labels, predictions)

    # Log your metrics
    wandb.log("loss": loss.numpy())
    # Get the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update the weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

A full example is available [here](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2).

## How is W&B different from TensorBoard?

When the cofounders started working on W&B, they were inspired to build a tool for the frustrated TensorBoard users at OpenAI. Here are a few things we've focused on improving:

1. **Reproduce models**: Weights & Biases is good for experimentation, exploration, and reproducing models later. We capture not just the metrics, but also the hyperparameters and version of the code, and we can save your version-control status and model checkpoints for you so your project is reproducible. 
2. **Automatic organization**: Whether you're picking up a project from a collaborator, coming back from a vacation, or dusting off an old project, W&B makes it easy to see all the models that have been tried so no one wastes hours, GPU cycles, or carbon re-running experiments.
3. **Fast, flexible integration**: Add W&B to your project in 5 minutes. Install our free open-source Python package and add a couple of lines to your code, and every time you run your model you'll have nice logged metrics and records.
4. **Persistent, centralized dashboard**: No matter where you train your models, whether on your local machine, in a shared lab cluster, or on spot instances in the cloud, your results are shared to the same centralized dashboard. You don't need to spend your time copying and organizing TensorBoard files from different machines.
5. **Powerful tables**: Search, filter, sort, and group results from different models. It's easy to look over thousands of model versions and find the best performing models for different tasks. TensorBoard isn't built to work well on large projects.
6. **Tools for collaboration**: Use W&B to organize complex machine learning projects. It's easy to share a link to W&B, and you can use private teams to have everyone sending results to a shared project. We also support collaboration via reports— add interactive visualizations and describe your work in markdown. This is a great way to keep a work log, share findings with your supervisor, or present findings to your lab or team.

Get started with a [free account](https://wandb.ai)

## Examples

We've created a few examples for you to see how the integration works:

* [Example on Github](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): MNIST example Using TensorFlow Estimators
* [Example on Github](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): Fashion MNIST example Using Raw TensorFlow
* [Wandb Dashboard](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): View result on W&B
* Customizing Training Loops in TensorFlow 2 - [Article](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [Colab Notebook](https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM) | [Dashboard](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)
