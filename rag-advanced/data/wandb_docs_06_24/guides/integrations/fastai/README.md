---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fastai

If you're using **fastai** to train your models, W&B has an easy integration using the `WandbCallback`. Explore the details in[ interactive docs with examples →](https://app.wandb.ai/borisd13/demo\_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## Log with W&B

**a)** [Sign up](https://wandb.ai/site) for a free account at [https://wandb.ai/site](https://wandb.ai/site) and then log in to your wandb account.

**b)** Install the wandb library on your machine in a Python 3 environment using `pip`

**c)** log in to the wandb library on your machine. You will find your API key here: [https://wandb.ai/authorize](https://wandb.ai/authorize).

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install wandb
wandb login
```

  </TabItem>
  <TabItem value="notebook">

```notebook
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

Then add the `WandbCallback` to the `learner` or `fit` method:

```python
import wandb
from fastai.callback.wandb import *

# start logging a wandb run
wandb.init(project="my_project")

# To log only during one training phase
learn.fit(..., cbs=WandbCallback())

# To log continuously for all training phases
learn = learner(..., cbs=WandbCallback())
```

:::info
If you use version 1 of Fastai, refer to the [Fastai v1 docs](v1.md).
:::

## WandbCallback Arguments

`WandbCallback` accepts the following arguments:

| Args                     | Description                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | Whether to log the model's: "`gradients`" , "`parameters`", "`all`" or `None` (default). Losses & metrics are always logged.                                                                                                                                 |
| log\_preds               | whether we want to log prediction samples (default to `True`).                                                                                                                                                                                               |
| log\_preds\_every\_epoch | whether to log predictions every epoch or at the end (default to `False`)                                                                                                                                                                                    |
| log\_model               | whether we want to log our model (default to False). This also requires `SaveModelCallback`                                                                                                                                                                  |
| model\_name              | The name of the `file` to save, overrides `SaveModelCallback`                                                                                                                                                                                                |
| log\_dataset             | <ul><li><code>False</code> (default)</li><li><code>True</code> will log folder referenced by learn.dls.path.</li><li>a path can be defined explicitly to reference which folder to log.</li></ul><p><em>Note: subfolder "models" is always ignored.</em></p> |
| dataset\_name            | name of logged dataset (default to `folder name`).                                                                                                                                                                                                           |
| valid\_dl                | `DataLoaders` containing items used for prediction samples (default to random items from `learn.dls.valid`.                                                                                                                                                  |
| n\_preds                 | number of logged predictions (default to 36).                                                                                                                                                                                                                |
| seed                     | used for defining random samples.                                                                                                                                                                                                                            |

For custom workflows, you can manually log your datasets and models:

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_Note: any subfolder "models" will be ignored._

## Distributed Training

`fastai` supports distributed training by using the context manager `distrib_ctx`. W&B supports this automatically and enables you to track your Multi-GPU experiments out of the box.

A minimal example is shown below:

<Tabs
  defaultValue="script"
  values={[
    {label: 'Script', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

<!-- {% code title="train.py" %} -->
```python
import wandb
from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = rank0_first(lambda: untar_data(URLs.PETS) / "images")


def train():
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
    )
    wandb.init("fastai_ddp", entity="capecape")
    cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(sync_bn=False):
        learn.fit(1)


if __name__ == "__main__":
    train()
```

Then, in your terminal you will execute:

```
$ torchrun --nproc_per_node 2 train.py
```

in this case, the machine has 2 GPUs.

  </TabItem>
  <TabItem value="notebook">

You can now run distributed training directly inside a notebook!

```python
import wandb
from fastai.vision.all import *

from accelerate import notebook_launcher
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = untar_data(URLs.PETS) / "images"


def train():
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
    )
    wandb.init("fastai_ddp", entity="capecape")
    cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fit(1)


notebook_launcher(train, num_processes=2)
```

  </TabItem>
</Tabs>

### Logging only on the main process

In the examples above, `wandb` launches one run per process. At the end of the training, you will end up with two runs. This can sometimes be confusing, and you may want to log only on the main process. To do so, you will have to detect in which process you are manually and avoid creating runs (calling `wandb.init` in all other processes)

<Tabs
  defaultValue="script"
  values={[
    {label: 'Script', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
import wandb
from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = rank0_first(lambda: untar_data(URLs.PETS) / "images")


def train():
    cb = []
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
    )
    if rank_distrib() == 0:
        run = wandb.init("fastai_ddp", entity="capecape")
        cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(sync_bn=False):
        learn.fit(1)


if __name__ == "__main__":
    train()
```
in your terminal call:

```
$ torchrun --nproc_per_node 2 train.py
```

  </TabItem>
  <TabItem value="notebook">

```python
import wandb
from fastai.vision.all import *

from accelerate import notebook_launcher
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = untar_data(URLs.PETS) / "images"


def train():
    cb = []
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
    )
    if rank_distrib() == 0:
        run = wandb.init("fastai_ddp", entity="capecape")
        cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fit(1)


notebook_launcher(train, num_processes=2)
```

  </TabItem>
</Tabs>

## Examples

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo\_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): A thoroughly documented walkthrough
* [Image Segmentation on CamVid](http://bit.ly/fastai-wandb): A sample use case of the integration
