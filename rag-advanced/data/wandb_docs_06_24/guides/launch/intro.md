---
slug: /guides/launch
description: Easily scale and manage ML jobs using W&B Launch.
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Launch

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

Easily scale training [runs](../runs/intro.md) from your desktop to a compute resource like Amazon SageMaker, Kubernetes and more with W&B Launch. Once W&B Launch is configured, you can quickly run training scripts, model evaluation suites, prepare models for production inference, and more with a few clicks and commands. 

## How it works

Launch is composed of three fundamental components: **launch jobs**, **queues**, and **agents**.

A [*launch job*](./launch-terminology.md#launch-job) is a blueprint for configuring and running tasks in your ML workflow. Once you have a launch job, you can add it to a [*launch queue*](./launch-terminology.md#launch-queue). A launch queue is a first-in, first-out (FIFO) queue where you can configure and submit your jobs to a particular compute target resource, such as Amazon SageMaker or a Kubernetes cluster. 

As jobs are added to the queue, one or more [*launch agents*](./launch-terminology.md#launch-agent) will poll that queue and execute the job on the system targeted by the queue.

![](/images/launch/launch_overview.png)

Based on your use case, you (or someone on your team) will configure the launch queue according to your chosen [compute resource target](./launch-terminology.md#target-resources) (for example Amazon SageMaker) and deploy a launch agent on your own infrastructure. 


See the [Terms and concepts](./launch-terminology.md) page for more information on launch jobs, how queues work, launch agents, and additional information on how W&B Launch works.

## How to get started

Depending on your use case, explore the following resources to get started with W&B Launch:

* If this is your first time using W&B Launch, we recommend you go through the [Walkthrough](./walkthrough.md) guide.
* Learn how to set up [W&B Launch](./setup-launch.md).
* Create a [launch job](./create-launch-job.md).
* Check out the W&B Launch [public jobs GitHub repository](https://github.com/wandb/launch-jobs) for templates of common tasks like [deploying to Triton](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [evaluating an LLM](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals), or more.
    * View launch jobs created from this repository in this public [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B project.

