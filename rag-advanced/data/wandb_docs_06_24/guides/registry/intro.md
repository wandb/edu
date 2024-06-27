---
slug: /guides/registry
displayed_sidebar: default
---

# Registry

:::info
W&B Registry is in private preview. Contact your account team or support@wandb.com for early access.  
:::


Use W&B Registry to share artifacts from your machine learning pipeline, such as machine learning and datasets, across teams within an organization. 


## How it works
W&B Registry is composed of three main components: registries, collections, and [artifact versions](../artifacts/create-a-new-artifact-version.md).

A *registry* is a repository or catalog for ML assets of the same kind. You can think of a registry as the top most level of a directory. Each registry consists of one or more sub directories called collections. A *collection* is a folder or a set of linked [artifact versions](../artifacts/create-a-new-artifact-version.md) inside a registry. An [*artifact version*](../artifacts/create-a-new-artifact-version.md) is a single, immutable snapshot of an artifact at a particular stage of its development.  A registry belongs to an organization, not a specific team.

![](/images/registry/registry_diagram_homepage.png)

Consider the following example, suppose your company wants to explore brand new cat classification models within your organization. To do this, you create a registry called "Cat Models registry". You are not sure which image classification is best, so you create multiple machine learning experiments with varying hyperparameters and algorithms. To organize your experiments and results, you create a collection for each model algorithm that experiment. Within that collection, you link the best model artifacts from your experiments to that collection.

<!-- To do: Add Raven's new diagram -->

## Migrating from the W&B Model Registry to the W&B Registry

If your team is actively using the existing W&B Model Registry to organize your models, this will still be available through the new Registry App UI. Navigate to the Model Registry from the homepage, and the banner will allow to select a team and visit it's model registry.

![](/images/registry/nav_to_old_model_reg.gif)

Look out for incoming information on a migration we will be making available to migrate contents from the current model registry into the new model registry inside W&B Registry. You can reach out to support@wandb.com with any questions or to speak to our product team about any concerns with the migration.

 

<!-- To do: INSERT -->