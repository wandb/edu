---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# MMEngine

MMEngine by [OpenMMLab](https://github.com/open-mmlab) is a foundational library for training deep learning models based on PyTorch. MMEngine implements a next-generation training architecture for the OpenMMLab algorithm library, providing a unified execution foundation for over 30 algorithm libraries within OpenMMLab. Its core components include the training engine, evaluation engine, and module management.

[Weights and Biases](https://wandb.ai/site) is directly integrated into MMEngine through a dedicated [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) that can be used to
- log training and evaluation metrics.
- log and manage experiment configs.
- log additional records such as graph, images, scalars, etc.

## Getting started

First, you need to install `openmim` and `wandb`. You can then proceed to install `mmengine` and `mmcv` using `openmim`.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install -q -U openmim wandb
mim install -q mmengine mmcv
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install -q -U openmim wandb
!mim install -q mmengine mmcv
```

  </TabItem>
</Tabs>

## Using the `WandbVisBackend` with MMEngine Runner

This section demonstrates a typical workflow using `WandbVisBackend` using [`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner).

First, you need to define a `visualizer` from a visualization config.

```python
from mmengine.visualization import Visualizer

# define the visualization configs
visualization_cfg = dict(
    name="wandb_visualizer",
    vis_backends=[
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(project="mmengine"),
        )
    ],
    save_dir="runs/wandb"
)

# get the visualizer from the visualization configs
visualizer = Visualizer.get_instance(**visualization_cfg)
```

:::info
You pass a dictionary of arguments for [W&B run initialization](https://docs.wandb.ai/ref/python/init) input parameters to `init_kwargs`.
:::

Next, you simply initialize a `runner` with the `visualizer`, and call `runner.train()`.

```python
from mmengine.runner import Runner

# build the mmengine Runner which is a training helper for PyTorch
runner = Runner(
    model,
    work_dir='runs/gan/',
    train_dataloader=train_dataloader,
    train_cfg=train_cfg,
    optim_wrapper=opt_wrapper_dict,
    visualizer=visualizer, # pass the visualizer
)

# start training
runner.train()
```

| ![An example of your experiment tracked using the `WandbVisBackend`](@site/static/images/integrations/mmengine.png) | 
|:--:| 
| **An example of your experiment tracked using the `WandbVisBackend`.** |

## Using the `WandbVisBackend` with OpenMMLab computer vision libraries

The `WandbVisBackend` can also be used easily to track experiments with OpenMMLab computer vision libraries such as [MMDetection](https://mmdetection.readthedocs.io/).

```python
# inherit base configs from the default runtime configs
_base_ = ["../_base_/default_runtime.py"]

# Assign the `WandbVisBackend` config dictionary to the
# `vis_backends` of the `visualizer` from the base configs
_base_.visualizer.vis_backends = [
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': 'mmdet',
            'entity': 'geekyrakshit'
        },
    ),
]
```
