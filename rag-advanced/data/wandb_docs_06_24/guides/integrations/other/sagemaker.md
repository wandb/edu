---
slug: /guides/integrations/sagemaker
description: How to integrate W&B with Amazon SageMaker.
displayed_sidebar: default
---

# SageMaker

## SageMaker Integration

W&B integrates with [Amazon SageMaker](https://aws.amazon.com/sagemaker/), automatically reading hyperparameters, grouping distributed runs, and resuming runs from checkpoints.

### Authentication

W&B looks for a file named `secrets.env` relative to the training script and loads them into the environment when `wandb.init()` is called. You can generate a `secrets.env` file by calling `wandb.sagemaker_auth(path="source_dir")` in the script you use to launch your experiments. Be sure to add this file to your `.gitignore`!

### Existing Estimators

If you're using one of SageMakers preconfigured estimators you need to add a `requirements.txt` to your source directory that includes wandb

```
wandb
```

If you're using an estimator that's running Python 2, you'll need to install psutil directly from a [wheel](https://pythonwheels.com) before installing wandb:

```
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

:::info
A complete example is available on [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) and you can read more on our [blog](https://wandb.ai/site/articles/running-sweeps-with-sagemaker).\
You can also read the [tutorial](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) on deploying a sentiment analyzer using SageMaker and W&B.
:::

:::caution
The W&B sweep agent will not behave as expected in a SageMaker job unless our SageMaker integration is disabled. You can disable the SageMaker integration in your runs by modifying your invocation of `wandb.init` as follows:

```
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
:::
