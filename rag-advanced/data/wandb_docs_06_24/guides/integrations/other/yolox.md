---
slug: /guides/integrations/yolox
description: How to integrate W&B with YOLOX.
displayed_sidebar: default
---

# YOLOX

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is an anchor-free version of YOLO with strong performance for object detection. YOLOX contains a Weights & Biases integration that enables you to turn on logging of training, validation and system metrics, as well as interactive validation predictions with just 1 command line argument.

## Getting Started

To use YOLOX with Weights & Biases you will first need to sign up for a Weights & Biases account [here](https://wandb.ai/site).

Then just use the `--logger wandb` command line argument to turn on logging with wandb. Optionally you can also pass all of the arguments that [wandb.init](../../track/launch.md) would expect, just prepend `wandb-` to the start of each argument

**Note:** `num_eval_imges` controls the number of validation set images and predictions that will be logged to Weights & Biases Tables for model evaluation.

```shell
# login to wandb
wandb login

# call your yolox training script with the `wandb` logger argument
python tools/train.py .... --logger wandb \
                wandb-project <project-name> \
                wandb-entity <entity>
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
```

## Example

[Example dashboard with YOLOX training and validation metrics ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

![](/images/integrations/yolox_example_dashboard.png)

Any questions or issues about this Weights & Biases integration? Open an issue in the [YOLOX github repository](https://github.com/Megvii-BaseDetection/YOLOX) and we'll catch it and get you an answer :)
