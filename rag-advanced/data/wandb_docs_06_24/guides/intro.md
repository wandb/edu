---
description: >-
  An overview of what is W&B along with links on how to get started
  if you are a first time user.
slug: /guides
displayed_sidebar: default
---

# What is W&B?

Weights & Biases (W&B) is the AI developer platform, with tools for training models, fine-tuning models, and leveraging foundation models. 

Set up W&B in 5 minutes, then quickly iterate on your machine learning pipeline with the confidence that your models and data are tracked and versioned in a reliable system of record.

![](@site/static/images/general/architecture.png)

This diagram outlines the relationship between W&B products.

**[W&B Models](/guides/models.md)** is a set of lightweight, interoperable tools for machine learning practitioners training and fine-tuning models.
- [Experiments](/guides/track/intro.md): Machine learning experiment tracking
- [Model Registry](/guides/model_registry/intro.md): Manage production models centrally
- [Launch](/guides/launch/intro.md): Scale and automate workloads
- [Sweeps](/guides/sweeps/intro.md): Hyperparameter tuning and model optimization

**[W&B Weave](https://wandb.github.io/weave/)** is a lightweight toolkit for tracking and evaluating LLM applications.

**[W&B Core](/guides/platform.md)** is a core set of powerful building blocks for tracking and visualizing data and models, and communicating results.
- [Artifacts](/guides/artifacts/intro.md): Version assets and track lineage
- [Tables](/guides/tables/intro.md): Visualize and query tabular data
- [Reports](/guides/reports/intro.md): Document and collaborate on your discoveries
<!-- - [Weave](/guides/app/features/panels/weave) Query and create visualizations of your data -->

## Are you a first-time user of W&B?

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Start exploring W&B with these resources:

1. [Intro Notebook](http://wandb.me/intro): Run quick sample code to track experiments in 5 minutes
2. [Quickstart](../quickstart.md): Read a quick overview of how and where to add W&B to your code
1. Explore our [Integrations guide](./integrations/intro.md) and our [W&B Easy Integration YouTube](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) playlist for information on how to integrate W&B with your preferred machine learning framework.
1. View the [API Reference guide](../ref/README.md) for technical specifications about the W&B Python Library, CLI, and Query Language operations.

## How does W&B work?

We recommend you read the following sections in this order if you are a first-time user of W&B:

1. Learn about [Runs](./runs/intro.md), W&B's basic unit of computation.
2. Create and track machine learning experiments with [Experiments](./track/intro.md).
3. Discover W&B's flexible and lightweight building block for dataset and model versioning with [Artifacts](./artifacts/intro.md).
4. Automate hyperparameter search and explore the space of possible models with [Sweeps](./sweeps/intro.md).
5. Manage the model lifecycle from training to production with [Model Management](./model_registry/intro.md).
6. Visualize predictions across model versions with our [Data Visualization](./tables/intro.md) guide.
7. Organize W&B Runs, embed and automate visualizations, describe your findings, and share updates with collaborators with [Reports](./reports/intro.md).
