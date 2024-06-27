---
description: ''
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Download a model version

Use the W&B Python SDK to download a model artifact that you linked to the Model Registry. Downloading a model is particularly useful if you want to load or consume a model in a future to evaluate a model's performance, make predictions with a dataset, or use ship the model to production. 

:::info
You are responsible for providing additional Python functions, API calls to reconstruct, deserialize your model into a form that you can work with. 

W&B suggests that you document information on how to load models into memory with model cards. For more information, see the [Document machine learning models](./create-model-cards.md) page. 
:::


Replace values within `<>` with your own:

```python
import wandb

# Initialize a run
run = wandb.init(project="<project>", entity="<entity>")

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name="<your-model-name>")
```

Reference a model version with one of following formats listed:

* `latest` - Use `latest` alias to specify the model version that is most recently linked.
* `v#` - Use `v0`, `v1`, `v2`, and so on to fetch a specific version in the Registered Model
* `alias` - Specify the custom alias that you and your team assigned to your model version

See [`use_model`](../../ref/python/run.md#use_model) in the API Reference guide for more information on possible parameters and return type.

<details>

<summary>Example: Download and use a logged model</summary>

For example, in the proceeding code snippet a user called the `use_model` API. They specified the name of the model artifact they want to fetch and they also provided a version/alias. They then stored the path that returned from the API to the `downloaded_model_path` variable.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # semantic nickname or identifier for the model version
model_artifact_name = "fine-tuned-model"

# Initialize a run
run = wandb.init()
# Access and download model. Returns path to downloaded artifact

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

<!-- <Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="cli">

Replace values within `<>` with your own:

```python
import wandb

# Initialize a run
run = wandb.init(project="<project>", entity="<entity>")

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name="<your-model-name>")
```

Reference a model version with one of following formats listed:

* `latest` - Use `latest` alias to specify the model version that is most recently linked.
* `v#` - Use `v0`, `v1`, `v2`, and so on to fetch a specific version in the Registered Model
* `alias` - Specify the custom alias that you and your team assigned to your model version

See [`use_model`](../../ref/python/run.md#use_model) in the API Reference guide for more information on possible parameters and return type.




  </TabItem>
  <TabItem value="app">

1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select **View details** next to the name of the registered model that contains the model you want to download.
3. Within the Versions section, select the View button next to the model version you want to download.
4. Select the **Files** tab. 
5. Click on the download button next to the model file you want to download. 

![](/images/models/download_model_ui.gif)

  </TabItem>
</Tabs> -->







