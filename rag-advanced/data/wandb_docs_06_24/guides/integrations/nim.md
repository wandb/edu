---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# NVIDIA NeMo Inference Microservice Deploy Job

Deploy a model artifact from W&B to a NVIDIA NeMo Inference Microservice. To do this, use W&B Launch. W&B Launch converts model artifacts to NVIDIA NeMo Model and deploys to a running NIM/Triton server.

W&B Launch currently accepts the following compatible model types:

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (coming soon)


:::info
Deployment time varies by model and machine type. The base Llama2-7b config takes about 1 minute on GCP's `a2-ultragpu-1g`.
:::


## Quickstart

1. [Create a launch queue](../launch/add-job-to-queue.md) if you don't have one already. See an example queue config below.

   ```yaml
   net: host
   gpus: all # can be a specific set of GPUs or `all` to use everything
   runtime: nvidia # also requires nvidia container runtime
   volume:
     - model-store:/model-store/
   ```

   ![image](/images/integrations/nim1.png)

2. Create this job in your project:

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. Launch an agent on your GPU machine:
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. Submit the deployment launch job with your desired configs from the [Launch UI](https://wandb.ai/launch)
   1. You can also submit via the CLI:
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      ![image](/images/integrations/nim2.png)
5. You can track the deployment process in the Launch UI.
   ![image](/images/integrations/nim3.png)
6. Once complete, you can immediately curl the endpoint to test the model. The model name is always `ensemble`.
   ```bash
    #!/bin/bash
    curl -X POST "http://0.0.0.0:9999/v1/completions" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ensemble",
            "prompt": "Tell me a joke",
            "max_tokens": 256,
            "temperature": 0.5,
            "n": 1,
            "stream": false,
            "stop": "string",
            "frequency_penalty": 0.0
            }'
   ```
