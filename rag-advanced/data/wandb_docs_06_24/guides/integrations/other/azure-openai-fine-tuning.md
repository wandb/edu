---
slug: /guides/integrations/azure-openai-fine-tuning
description: How to Fine-Tune Azure OpenAI models using W&B.
displayed_sidebar: default
---

# Azure OpenAI Fine-Tuning

## Introduction
Fine-tuning GPT-3.5 or GPT-4 models on Microsoft Azure using W&B allows for detailed tracking and analysis of model performance. This guide extends the concepts from the [OpenAI Fine-Tuning guide](/guides/integrations/openai) with specific steps and features for Azure OpenAI.

![](/images/integrations/open_ai_auto_scan.png)

:::info
The Weights and Biases fine-tuning integration works with `openai >= 1.0`. Please install the latest version of `openai` by doing `pip install -U openai`.
:::


## Prerequisites
- Azure OpenAI service set up as per [official Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune).
- Latest versions of `openai`, `wandb`, and other required libraries installed.

## Sync Azure OpenAI fine-tuning results in W&B in 2 lines

```python
from openai import AzureOpenAI

# Connect to Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# Create and validate your training and validation datasets in JSONL format,
# upload them via the client,
# and start a fine-tuning job.

from wandb.integration.openai.fine_tuning import WandbLogger

# Sync your fine-tuning results with W&B!
WandbLogger.sync(
    fine_tune_job_id=job_id, openai_client=client, project="your_project_name"
)
```

### Check out interactive examples

* [Demo Colab](http://wandb.me/azure-openai-colab)

## Visualization and versioning in W&B
- Utilize W&B for versioning and visualizing training and validation data as Tables.
- The datasets and model metadata are versioned as W&B Artifacts, allowing for efficient tracking and version control.

![](/images/integrations/openai_data_artifacts.png)

![](/images/integrations/openai_data_visualization.png)

## Retrieving the fine-tuned model
- The fine-tuned model ID is retrievable from Azure OpenAI and is logged as a part of model metadata in W&B.

![](/images/integrations/openai_model_metadata.png)

## Additional resources
- [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning/)
- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)
- [Demo Colab](http://wandb.me/azure-openai-colab)