---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hugging Face Autotrain

[🤗 AutoTrain](https://huggingface.co/docs/autotrain/index) is a no-code tool for training state-of-the-art models for Natural Language Processing (NLP) tasks, for Computer Vision (CV) tasks, and for Speech tasks and even for Tabular tasks.

[Weights & Biases](http://wandb.com/) is directly integrated into 🤗 AutoTrain, providing experiment tracking and config management. It's as easy as using a single parameter in the CLI command for your experiments!

| ![An example of how the metrics of your experiment are logged](@site/static/images/integrations/hf-autotrain-1.png) | 
|:--:| 
| **An example of how the metrics of your experiment are logged.** |

## Getting Started

First, we need to install `autotrain-advanced` and `wandb`.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install --upgrade autotrain-advanced wandb
```

  </TabItem>
  <TabItem value="notebook">

```notebook
!pip install --upgrade autotrain-advanced wandb
```

  </TabItem>
</Tabs>

## Getting Started: Fine-tuning an LLM

To demonstrate these changes we will fine-tune an LLM on a math dataset and try achieving SoTA result in `pass@1` on the [GSM8k Benchmarks](https://github.com/openai/grade-school-math).

### Preparing the Dataset

🤗 AutoTrain expects your CSV custom dataset in a certain format to work properly. Your training file must contain a "text" column on which the training will be done. For best results, the "text" column should have data in the `### Human: Question?### Assistant: Answer.` format. A great example for the kind of dataset AutoTrain Advanced expects would be [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco). However, if you observe the [MetaMathQA dataset](https://huggingface.co/datasets/meta-math/MetaMathQA), there are 3 columns - "query", "response" and "type". We will preprocess this dataset by removing the "type" column and combining the content of the "query" and "response" columns under one "text" column with the `### Human: Query?### Assistant: Response.` format. The resulting dataset is [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath) and it will be used for training.

### Training using Autotrain Advanced

We can start training using the Autotrain Advanced CLI. To leverage the logging functionality, we simply use the `--log` argument. Specifying `--log wandb` will seamlessly log your results to a [W&B run](https://docs.wandb.ai/guides/runs). 

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
autotrain llm \
    --train \
    --model HuggingFaceH4/zephyr-7b-alpha \
    --project-name zephyr-math \
    --log wandb \
    --data-path data/ \
    --text-column text \
    --lr 2e-5 \
    --batch-size 4 \
    --epochs 3 \
    --block-size 1024 \
    --warmup-ratio 0.03 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --weight-decay 0.0 \
    --gradient-accumulation 4 \
    --logging_steps 10 \
    --fp16 \
    --use-peft \
    --use-int4 \
    --merge-adapter \
    --push-to-hub \
    --token <huggingface-token> \
    --repo-id <huggingface-repository-address>
```

  </TabItem>
  <TabItem value="notebook">

```notebook
# Set hyperparameters
learning_rate = 2e-5
num_epochs = 3
batch_size = 4
block_size = 1024
trainer = "sft"
warmup_ratio = 0.03
weight_decay = 0.
gradient_accumulation = 4
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
logging_steps = 10

# Run training
!autotrain llm \
    --train \
    --model "HuggingFaceH4/zephyr-7b-alpha" \
    --project-name "zephyr-math" \
    --log "wandb" \
    --data-path data/ \
    --text-column text \
    --lr str(learning_rate) \
    --batch-size str(batch_size) \
    --epochs str(num_epochs) \
    --block-size str(block_size) \
    --warmup-ratio str(warmup_ratio) \
    --lora-r str(lora_r) \
    --lora-alpha str(lora_alpha) \
    --lora-dropout str(lora_dropout) \
    --weight-decay str(weight_decay) \
    --gradient-accumulation str(gradient_accumulation) \
    --logging-steps str(logging_steps) \
    --fp16 \
    --use-peft \
    --use-int4 \
    --merge-adapter \
    --push-to-hub \
    --token str(hf_token) \
    --repo-id "rishiraj/zephyr-math"
```

  </TabItem>
</Tabs>

| ![An example of how all the configs of your experiment are saved.](@site/static/images/integrations/hf-autotrain-2.gif) | 
|:--:| 
| **An example of how all the configs of your experiment are saved.** |

## More Resources

* [AutoTrain Advanced now supports Experiment Tracking](https://huggingface.co/blog/rishiraj/log-autotrain) by [Rishiraj Acharya](https://huggingface.co/rishiraj).
* [🤗 Autotrain Docs](https://huggingface.co/docs/autotrain/index)