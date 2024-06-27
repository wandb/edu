---
slug: /guides/integrations/paddleocr
description: How to integrate W&B with PaddleOCR.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PaddleOCR

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice implemented in PaddlePaddle. PaddleOCR support a variety of cutting-edge algorithms related to OCR, and developed industrial solution. PaddleOCR now comes with a Weights & Biases integration for logging training and evaluation metrics along with model checkpoints with corresponding metadata.

## Example Blog & Colab

[**Read here**](https://wandb.ai/manan-goel/text\_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw) to see how to train a model with PaddleOCR on the ICDAR2015 dataset. This also comes with a [**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing) and the corresponding live W&B dashboard is available [**here**](https://wandb.ai/manan-goel/text\_detection). There is also a Chinese version of this blog here: [**W&B对您的OCR模型进行训练和调试**](https://wandb.ai/wandb\_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## Using PaddleOCR with Weights & Biases

### 1. Sign up and Log in to wandb

[**Sign up**](https://wandb.ai/site) for a free account, then from the command line install the wandb library in a Python 3 environment. To login, you'll need to be signed in to you account at www.wandb.ai, then **you will find your API key on the** [**Authorize page**](https://wandb.ai/authorize).

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

wandb.login()
```

  </TabItem>
</Tabs>

### 2. Add wandb to your `config.yml` file

PaddleOCR requires configuration variables to be provided using a yaml file. Adding the following snippet at the end of the configuration yaml file will automatically log all training and validation metrics to a W&B dashboard along with model checkpoints:

```python
Global:
    use_wandb: True
```

Any additional, optional arguments that you might like to pass to [`wandb.init`](https://docs.wandb.ai/guides/track/launch) can also be added under the `wandb` header in the yaml file:

```
wandb:  
    project: CoolOCR  # (optional) this is the wandb project name 
    entity: my_team   # (optional) if you're using a wandb team, you can pass the team name here
    name: MyOCRModel  # (optional) this is the name of the wandb run
```

### 3. Pass the `config.yml` file to `train.py`

The yaml file is then provided as an argument to the [training script](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py) available in the PaddleOCR repository.

```
python tools/train.py -c config.yml
```

Once you run your `train.py` file with Weights & Biases turned on, a link will be generated to bring you to your W&B dashboard:

![](/images/integrations/paddleocr_wb_dashboard1.png) ![](/images/integrations/paddleocr_wb_dashboard2.png)

![W&B Dashboard for the Text Detection Model](/images/integrations/paddleocr_wb_dashboard3.png)

## Feedback or Issues?

If you have any feedback or issues about the Weights & Biases integration please open an issue on the [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) or email support@wandb.com
