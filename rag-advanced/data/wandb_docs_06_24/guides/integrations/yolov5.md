---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# YOLOv5

[Ultralytics' YOLOv5](https://ultralytics.com/yolov5) ("You Only Look Once") model family enables real-time object detection with convolutional neural networks without all the agonizing pain.

[Weights & Biases](http://wandb.com) is directly integrated into YOLOv5, providing experiment metric tracking, model and dataset versioning, rich model prediction visualization, and more. **It's as easy as running a single `pip install` before you run your YOLO experiments!**

:::info
For a quick overview of the model and data-logging features of our YOLOv5 integration, check out [this Colab](https://wandb.me/yolo-colab) and accompanying video tutorial, linked below.
:::

<!-- {% embed url="https://www.youtube.com/watch?v=yyecuhBmLxE" %} -->

:::info
All W&B logging features are compatible with data-parallel multi-GPU training, e.g. with [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp\_tutorial.html).
:::

## Core Experiment Tracking

Simply by installing `wandb`, you'll activate the built-in W&B [logging features](../track/log/intro.md): system metrics, model metrics, and media logged to interactive [Dashboards](../track/app.md).

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # train a small network on a small dataset
```

Just follow the links printed to the standard out by wandb.

![All these charts and more!](/images/integrations/yolov5_experiment_tracking.png)

## Model Versioning and Data Visualization

But that's not all! By passing a few simple command line arguments to YOLO, you can take advantage of even more W&B features.

* Passing a number to `--save_period` will turn on [model versioning](../model_registry/intro.md). At the end of every `save_period` epochs, the model weights will be saved to W&B. The best-performing model on the validation set will be tagged automatically.
* Turning on the `--upload_dataset` flag will also upload the dataset for data versioning.
* Passing a number to `--bbox_interval` will turn on [data visualization](../intro.md). At the end of every `bbox_interval` epochs, the outputs of the model on the validation set will be uploaded to W&B.

<Tabs
  defaultValue="modelversioning"
  values={[
    {label: 'Model Versioning Only', value: 'modelversioning'},
    {label: 'Model Versioning and Data Visualization', value: 'bothversioning'},
  ]}>
  <TabItem value="modelversioning">

```python
python yolov5/train.py --epochs 20 --save_period 1
```

  </TabItem>
  <TabItem value="bothversioning">

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

  </TabItem>
</Tabs>

:::info
Every W&B account comes with 100 GB of free storage for datasets and models.
:::

Here's what that looks like.

![Model Versioning: the latest and the best versions of the model are identified.](/images/integrations/yolov5_model_versioning.png)

![Data Visualization: compare the input image to the model's outputs and example-wise metrics.](/images/integrations/yolov5_data_visualization.png)

:::info
With data and model versioning, you can resume paused or crashed experiments from any device, no setup necessary! Check out [the Colab ](https://wandb.me/yolo-colab)for details.
:::
