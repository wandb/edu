---
slug: /guides/integrations/paddledetection
description: How to integrate W&B with PaddleDetection.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PaddleDetection

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) is an end-to-end object-detection development kit based on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle). It implements varied mainstream object detection, instance segmentation, tracking and keypoint detection algorithms in modular design with configurable modules such as network components, data augmentations and losses.

PaddleDetection now comes with a built in W&B integration which logs all your training and validation metrics, as well as your model checkpoints and their corresponding metadata.

## Example Blog and Colab

[**Read our blog here**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) to see how to train a YOLOX model with PaddleDetection on a subset of the COCO2017 dataset. This also comes with a [**Google Colab**](https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing) and the corresponding live W&B dashboard is available [**here**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/runs/2ry6i2x9?workspace=)

## The PaddleDetection WandbLogger

The PaddleDetection WandbLogger will log your training and evaluation metrics to Weights & Biases as well as your model checkpoints while training.

## Using PaddleDetection with Weights & Biases

### Sign up and log in to W&B

[**Sign up**](https://wandb.ai/site) for a free Weights & Biases account, then pip install the wandb library. To login, you'll need to be signed in to you account at www.wandb.ai. Once signed in **you will find your API key on the** [**Authorize page**](https://wandb.ai/authorize)**.**

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```shell
pip install wandb

wandb login
```
  </TabItem>
  <TabItem value="notebook">

```notebook
!pip install wandb

wandb.login()
```
  </TabItem>
</Tabs>

### Activating the WandbLogger in your Training Script

#### Using the CLI

To use wandb via arguments to `train.py` in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/):

* Add the `--use_wandb` flag
* The first wandb arguments must be preceded by `-o` (you only need to pass this once)
* Each individual wandb argument must contain the prefix `wandb-` . For example any argument to be passed to [`wandb.init`](https://docs.wandb.ai/ref/python/init) would get the `wandb-` prefix

```shell
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```

#### Using a config.yml file

You can also activate wandb via the config file. Add the wandb arguments to the config.yml file under the wandb header like so:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

Once you run your `train.py` file with Weights & Biases turned on, a link will be generated to bring you to your W&B dashboard:

![A Weights & Biases Dashboard](/images/integrations/paddledetection_wb_dashboard.png)

## Feedback or Issues

If you have any feedback or issues about the Weights & Biases integration please open an issue on the [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection) or email support@wandb.com
