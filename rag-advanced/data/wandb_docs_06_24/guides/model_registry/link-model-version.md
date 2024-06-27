---
description: ''
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Link a model version

Link a model version to a registered model with the W&B App or programmatically with the Python SDK.

## Programmatically link a model

Use the [`link_model`](../../ref/python/run.md#link_model) method to programmatically log model files to a W&B run and link it to the [W&B Model Registry](./intro.md). 

Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

W&B creates a registered model for you if the name you specify for the `registered-model-name` parameter does not already exist. 

For example, suppose you have an existing registered model named "Fine-Tuned-Review-Autocompletion"(`registered-model-name="Fine-Tuned-Review-Autocompletion"`) in your Model Registry. And suppose that a few model versions are linked to it: `v0`, `v1`, `v2`. If you programmatically link a new model and use the same registered model name (`registered-model-name="Fine-Tuned-Review-Autocompletion"`), W&B links this model to the existing registered model and assigns it a model version `v3`. If no registered model with this name exists, a new one registered model is created and it will have a model version `v0`.

See an example ["Fine-Tuned-Review-Autocompletion" registered model here](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models).

## Interactively link a model
Interactively link a model with the Model Registry or with the Artifact browser.

<Tabs
  defaultValue="model_ui"
  values={[
    {label: 'Model Registry', value: 'model_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="model_ui">

1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Hover your mouse next to the name of the registered model you want to link a new model to. 
3. Select the meatball menu icon (three horizontal dots) next to  **View details**.
4. From the dropdown, select **Link new version**.
5. From the **Project** dropdown, select the name of the project that contains your model. 
6. From the **Model Artifact** dropdown, select the name of the model artifact. 
7. From the **Version** dropdown, select the model version you want to link to the registered model.

![](/images/models/link_model_wmodel_reg.gif)

  </TabItem>
  <TabItem value="artifacts_ui">

1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Select the Artifacts icon on the left sidebar.
3. Click on the model version you want to link to your registry.
4. Within the **Version overview** section, click the **Link to registry** button.
5. From the modal that appears on the right of the screen, select a registered model from the **Select a register model** menu dropdown. 
6. Click **Next step**.
7. (Optional) Select an alias from the **Aliases** dropdown. 
8. Click **Link to registry**. 

![](/images/models/manual_linking.gif)

  </TabItem>
</Tabs>





## View the source of linked models

There are two ways to view the source of linked models: The artifact browser within the project that the model is logged to and the W&B Model Registry.

A pointer connects a specific model version in the model registry to the source model artifact (located within the project the model is logged to). The source model artifact also has a pointer to the model registry.

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. Navigate to your model registry at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
![](/images/models/create_registered_model_1.png)
2. Select **View details** next the name of your registered model.
3. Within the **Versions** section, select **View** next to the model version you want to investigate.
4. Click on the **Version** tab within the right panel.
5. Within the **Version overview** section there is a row that contains a **Source Version** field. The **Source Version** field shows both the name of the model and the model's version.

For example, the following image shows a `v0` model version called `mnist_model` (see **Source version** field `mnist_model:v0`), linked to a registered model called `MNIST-dev`.

![](/images/models/view_linked_model_registry.png)

  </TabItem>
  <TabItem value="browser">

1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Select the Artifacts icon on the left sidebar.
3. Expand the **model** dropdown menu from the Artifacts panel.
4. Select the name and version of the model linked to the model registry.
5. Click on the **Version** tab within the right panel.
6. Within the **Version overview** section there is a row that contains a **Linked To** field. The **Linked To** field shows both the name of the registered model and the version it possesses(`registered-model-name:version`). 

For example, in the following image, there is a registered model called `MNIST-dev` (see the **Linked To** field). A model version called `mnist_model` with a version `v0`(`mnist_model:v0`) points to the `MNIST-dev` registered model.


![](/images/models/view_linked_model_artifacts_browser.png)


  </TabItem>
</Tabs>