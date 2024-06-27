---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Ultralytics

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb"></CTAButtons>

[Ultralytics](https://github.com/ultralytics/ultralytics) is the home for cutting-edge, state-of-the-art computer vision models for tasks like image classification, object detection, image segmentation, and pose estimation. Not only it hosts [YOLOv8](https://docs.ultralytics.com/models/yolov8/), the latest iteration in the YOLO series of real-time object detection models, but other powerful computer vision models such as [SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/), etc. Besides providing implementations of these models, Ultralytics also provides us with out-of-the-box workflows for training, fine-tuning, and applying these models using an easy-to-use API.

## Getting Started

First, we need to install `ultralytics` and `wandb`.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install --upgrade ultralytics==8.0.238 wandb

# or
# conda install ultralytics
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install --upgrade ultralytics==8.0.238 wandb
```

  </TabItem>
</Tabs>

**Note:** The integration currently has been tested with `ultralyticsv8.0.238` and below. Please report any issues to https://github.com/wandb/wandb/issues with the tag `yolov8`.

## Experiment Tracking and Visualizing Validation Results

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb"></CTAButtons>

This section demonstrates a typical workflow of using an [Ultralytics](https://docs.ultralytics.com/modes/predict/) model for training, fine-tuning, and validation and performing experiment tracking, model-checkpointing, and visualization of the model's performance using [W&B](https://wandb.ai/site).

You can try out the code in Google Colab: [Open In Colab](http://wandb.me/ultralytics-train)

You can also check out about the integration in this report: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

In order to use the W&B integration with Ultralytics, we need to import the `wandb.integration.ultralytics.add_wandb_callback` function.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

Next, we initialize the `YOLO` model of our choice, and invoke the `add_wandb_callback` function on it before performing inference with the model. This would ensure that when we perform training, fine-tuning, validation, or inference, it would automatically log the experiment logs and the images over laid with both ground-truth and the respective prediction results using the [interactive overlays for computer vision tasks](../track/log/media#image-overlays-in-tables) on W&B along with additional insights in a [`wandb.Table`](../tables/intro.md).

```python
# Initialize YOLO Model
model = YOLO("yolov8n.pt")

# Add W&B callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train/fine-tune your model
# At the end of each epoch, predictions on validation batches are logged
# to a W&B table with insightful and interactive overlays for
# computer vision tasks
model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)

# Finish the W&B run
wandb.finish()
```

Here's how experiments tracked using W&B for an Ultralytics training or fine-tuning workflow looks like:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

Here's how epoch-wise validation results are visualized using a [W&B Table](../tables/intro.md):

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## Visualizing Prediction Results

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb"></CTAButtons>

This section demonstrates a typical workflow of using an [Ultralytics](https://docs.ultralytics.com/modes/predict/) model for inference and visualizing the results using [W&B](https://wandb.ai/site).

You can try out the code in Google Colab: [Open in Colab](http://wandb.me/ultralytics-inference).

You can also check out about the integration in this report: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

In order to use the W&B integration with Ultralytics, we need to import the `wandb.integration.ultralytics.add_wandb_callback` function.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

Now, let us download a few images to test the integration on. You can use your own images, videos or camera sources. For more information on inference sources, you can check out the [official docs](https://docs.ultralytics.com/modes/predict/).

```python
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

Next, we initialize a W&B [run](../runs/intro.md) using `wandb.init`.

```python
# Initialize W&B run
wandb.init(project="ultralytics", job_type="inference")
```

Next, we initialize the `YOLO` model of our choice, and invoke the `add_wandb_callback` function on it before performing inference with the model. This would ensure that when we perform inference, it would automatically log the images overlaid with our [interactive overlays for computer vision tasks](../track/log/media#image-overlays-in-tables) along with additional insights in a [`wandb.Table`](../tables/intro.md).

```python
# Initialize YOLO Model
model = YOLO("yolov8n.pt")

# Add W&B callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Perform prediction which automatically logs to a W&B Table
# with interactive overlays for bounding boxes, segmentation masks
model(["./assets/img1.jpeg", "./assets/img3.png", "./assets/img4.jpeg", "./assets/img5.jpeg"])

# Finish the W&B run
wandb.finish()
```

Note: We do not need to explicitly initialize a run using `wandb.init()` in case of a training or fine-tuning workflow. However, tt is necessary to explicitly create a run, if the code only involves prediction.

Here's how the interactive bbox overlay looks:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

You can fine more information on the W&B image overlays [here](../track/log/media.md#image-overlays).

## More Resources

* [Supercharging Ultralytics with Weights & Biases](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)
