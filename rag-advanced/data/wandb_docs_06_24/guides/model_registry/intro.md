---
slug: /guides/model_registry
description: Model registry to manage the model lifecycle from training to production
displayed_sidebar: default
---

# Model registry
The W&B Model Registry houses a team's trained models where ML Practitioners can publish candidates for production to be consumed by downstream teams and stakeholders. It is used to house staged/candidate models and manage workflows associated with staging.

![](/images/models/model_reg_landing_page.png)

With W&B Model Registry, you can:

* [Bookmark your best model versions for each machine learning task.](./link-model-version.md)
* [Automate](./model-registry-automations.md) downstream processes and model CI/CD.
* Move model versions through its ML lifecycle; from staging to production.
* Track a model's lineage and audit the history of changes to production models.

![](/images/models/models_landing_page.png)

## How it works
Track and manage your staged models with a few simple steps.

1. **Log a model version**: In your training script, add a few lines of code to save the model files as an artifact to W&B. 
2. **Compare performance**: Check live charts to compare the metrics and sample predictions from model training and validation. Identify which model version performed the best.
3. **Link to registry**: Bookmark the best model version by linking it to a registered model, either programmatically in Python or interactively in the W&B UI.

The following code snippet demonstrates how to log and link a model to the Model Registry:

```python showLineNumbers
import wandb
import random

# Start a new W&B run
run = wandb.init(project="models_quickstart")

# Simulate logging model metrics
run.log({"acc": random.random()})

# Create a simulated model file
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# Log and link the model to the Model Registry
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **Connect model transitions to CI/DC workflows**: transition candidate models through workflow stages and [automate downstream actions](./model-registry-automations.md) with webhooks or jobs.


## How to get started
Depending on your use case, explore the following resources to get started with W&B Models:

* Check out the two-part video series:
  1. [Logging and registering models](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. [Consuming models and automating downstream processes](https://www.youtube.com/watch?v=8PFCrDSeHzw) in the Model Registry.
* Read the [models walkthrough](./walkthrough.md) for a step-by-step outline of the W&B Python SDK commands you could use to create, track, and use a dataset artifact.
* Learn about:
   * [Protected models and access control](./access_controls.md).
   * [How to connect the Model Registry to CI/CD processes](./model-registry-automations.md).
   * Set up [Slack notifications](./notifications.md) when a new model version is linked to a registered model.
* Review [this](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) report on how the Model Registry fits into your ML workflow and the benefits of using one for model management. 
* Take the W&B [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) course and learn how to:
  * Use the W&B Model Registry to manage and version your models, track lineage, and promote models through different lifecycle stages
  * Automate your model management workflows using webhooks and launch jobs.
  * See how the Model Registry integrates with external ML systems and tools in your model development lifecycle for model evaluation, monitoring, and deployment.
