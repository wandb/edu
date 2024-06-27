---
description: Guide on how to integrate W&B with Dagster.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dagster

Use Dagster and W&B (W&B) to orchestrate your MLOps pipelines and maintain ML assets. The integration with W&B makes it easy within Dagster to:

* Use and create [W&B Artifacts](../artifacts/intro.md).
* Use and create Registered Models in [W&B Model Registry](../model_registry/intro.md).
* Run training jobs on dedicated compute using [W&B Launch](../launch/intro.md).
* Use the [wandb](../../ref/python/README.md) client in ops and assets.

The W&B Dagster integration provides a W&B-specific Dagster resource and IO Manager:

* `wandb_resource`: a Dagster resource used to authenticate and communicate to the W&B API. 
* `wandb_artifacts_io_manager`: a Dagster IO Manager used to consume W&B Artifacts.

The following guide demonstrates how to satisfy prerequisites to use W&B in Dagster, how to create and use W&B Artifacts in ops and assets, how to use W&B Launch and recommended best practices.

## Before you get started
You will need the following resources to use Dagster within Weights and Biases:
1. **W&B API Key**.
2. **W&B entity (user or team)**: An entity is a username or team name where you send W&B Runs and Artifacts. Make sure to create your account or team entity in the W&B App UI before you log runs. If you do not specify ain entity, the run will be sent to your default entity, which is usually your username. Change your default entity in your settings under **Project Defaults**.
3. **W&B project**: The name of the project where [W&B Runs](../runs/intro.md) are stored.

Find your W&B entity by checking the profile page for that user or team in the W&B App. You can use a pre-existing W&B project or create a new one. New projects can be created on the W&B App homepage or on user/team profile page. If a project does not exist it will be automatically created when you first use it. The proceeding instructions demonstrate how to get an API key: 

### How to get an API key
1. [Log in to W&B](https://wandb.ai/login). Note: if you are using W&B Server ask your admin for the instance host name.
2. Collect your API key by navigating to the [authorize page](https://wandb.ai/authorize) or in your user/team settings. For a production environment we recommend using a [service account](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful) to own that key. 
3. Set an environment variable for that API key export `WANDB_API_KEY=YOUR_KEY`.


The proceeding examples demonstrate where to specify your API key in your Dagster code. Make sure to specify your entity and project name within the `wandb_config` nested dictionary. You can pass different `wandb_config` values to different ops/assets if you want to use a different W&B Project. For more information about possible keys you can pass, see the Configuration section below.


<Tabs
  defaultValue="job"
  values={[
    {label: 'configuration for @job', value: 'job'},
    {label: 'configuration for @repository using assets', value: 'repository'},
  ]}>
  <TabItem value="job">

Example: configuration for `@job`
```python
# add this to your config.yaml
# alternatively you can set the config in Dagit's Launchpad or JobDefinition.execute_in_process
# Reference: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # replace this with your W&B entity
     project: my_project # replace this with your W&B project


@job(
   resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
       "io_manager": wandb_artifacts_io_manager,
   }
)
def simple_job_example():
   my_op()
```

  </TabItem>
  <TabItem value="repository">


Example: configuration for `@repository` using assets

```python
from dagster_wandb import wandb_artifacts_io_manager, wandb_resource
from dagster import (
   load_assets_from_package_module,
   make_values_resource,
   repository,
   with_resources,
)

from . import assets

@repository
def my_repository():
   return [
       *with_resources(
           load_assets_from_package_module(assets),
           resource_defs={
               "wandb_config": make_values_resource(
                   entity=str,
                   project=str,
               ),
               "wandb_resource": wandb_resource.configured(
                   {"api_key": {"env": "WANDB_API_KEY"}}
               ),
               "wandb_artifacts_manager": wandb_artifacts_io_manager.configured(
                   {"cache_duration_in_minutes": 60} # only cache files for one hour
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # replace this with your W&B entity
                       "project": "my_project", # replace this with your W&B project
                   }
               }
           },
       ),
   ]
```
Note that we are configuring the IO Manager cache duration in this example contrary to the example for `@job`.

  </TabItem>
</Tabs>


### Configuration
The proceeding configuration options are used as settings on the W&B-specific Dagster resource and IO Manager provided by the integration.

* `wandb_resource`: Dagster [resource](https://docs.dagster.io/concepts/resources) used to communicate with the W&B API. It automatically authenticates using the provided API key. Properties:
    * `api_key`: (str, required): a W&B API key necessary to communicate with the W&B API.
    * `host`: (str, optional): the API host server you wish to use. Only required if you are using W&B Server. It defaults to the Public Cloud host: [https://api.wandb.ai](https://api.wandb.ai)
* `wandb_artifacts_io_manager`: Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers) to consume W&B Artifacts. Properties:
    * `base_dir`: (int, optional) Base directory used for local storage and caching. W&B Artifacts and W&B Run logs will be written and read from that directory. By default, it’s using the `DAGSTER_HOME` directory.
    * `cache_duration_in_minutes`: (int, optional) to define the amount of time W&B Artifacts and W&B Run logs should be kept in the local storage. Only files and directories that were not opened for that amount of time are removed from the cache. Cache purging happens at the end of an IO Manager execution. You can set it to 0, if you want to disable caching completely. Caching improves speed when an Artifact is reused between jobs running on the same machine. It defaults to 30 days.
    * `run_id`: (str, optional): A unique ID for this run, used for resuming. It must be unique in the project, and if you delete a run you can't reuse the ID. Use the name field for a short descriptive name, or config for saving hyperparameters to compare across runs. The ID cannot contain the following special characters: `/\#?%:..` You need to set the Run ID when you are doing experiment tracking inside Dagster to allow the IO Manager to resume the run. By default it’s set to the Dagster Run ID e.g `7e4df022-1bf2-44b5-a383-bb852df4077e`.
    * `run_name`: (str, optional) A short display name for this run, which is how you'll identify this run in the UI. By default, it’s set to a string with the following format dagster-run-[8 first characters of the Dagster Run ID] e.g. `dagster-run-7e4df022`.
    * `run_tags`: (list[str], optional): A list of strings, which will populate the list of tags on this run in the UI. Tags are useful for organizing runs together, or applying temporary labels like "baseline" or "production". It's easy to add and remove tags in the UI, or filter down to just runs with a specific tag. Any W&B Run used by the integration will have the `dagster_wandb` tag.

## Use W&B Artifacts

The integration with W&B Artifact relies on a Dagster IO Manager.

[IO Managers](https://docs.dagster.io/concepts/io-management/io-managers) are user-provided objects that are responsible for storing the output of an asset or op and loading it as input to downstream assets or ops. For example, an IO Manager might store and load objects from files on a filesystem.

The integration provides an IO Manager for W&B Artifacts. This allows any Dagster `@op` or `@asset` to create and consume W&B Artifacts natively. Here’s a simple example of an `@asset` producing a W&B Artifact of type dataset containing a Python list.

```python
@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
    return [1, 2, 3] # this will be stored in an Artifact
```

You can annotate your `@op`, `@asset` and `@multi_asset` with a metadata configuration in order to write Artifacts. Similarly you can also consume W&B Artifacts even if they were created outside Dagster. 

## Write W&B Artifacts
Before continuing, we recommend you to have a good understanding of how to use W&B Artifacts. Consider reading the [Guide on Artifacts](../artifacts/intro.md).

Return an object from a Python function to write a W&B Artifact. The following objects are supported by W&B:
* Python objects (int, dict, list…)
* W&B objects (Table, Image, Graph…)
* W&B Artifact objects

The proceeding examples demonstrate how to write W&B Artifacts with Dagster assets (`@asset`):


<Tabs
  defaultValue="python_objects"
  values={[
    {label: 'Python objects', value: 'python_objects'},
    {label: 'W&B object', value: 'wb_object'},
    {label: 'W&B Artifacts', value: 'wb_artifact'},
  ]}>
  <TabItem value="python_objects">

Anything that can be serialized with the [pickle](https://docs.python.org/3/library/pickle.html) module is pickled and added to an Artifact created by the integration. The content is unpickled when you read that Artifact inside Dagster (see [Read artifacts](#read-wb-artifacts) for more details). 

```python
@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
    return [1, 2, 3]
```


W&B supports multiple Pickle-based serialization modules ([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)). You can also use more advanced serialization like [ONNX](https://onnx.ai/) or [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language). Please refer to the [Serialization](#serialization-configuration) section for more information.

  </TabItem>
  <TabItem value="wb_object">

any native W&B object (e.g [Table](../../ref/python/data-types/table.md), [Image](../../ref/python/data-types/image.md), [Graph](../../ref/python/data-types/graph.md)) is added to an Artifact created by the integration. Here’s an example using a Table.

```python
import wandb

@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset_in_table():
    return wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
```

  </TabItem>
  <TabItem value="wb_artifact">

For complex use cases, it might be necessary to build your own Artifact object. The integration still provides useful additional features like augmenting the metadata on both sides of the integration.

```python
import wandb

MY_ASSET = "my_asset"

@asset(
    name=MY_ASSET,
    io_manager_key="wandb_artifacts_manager",
)
def create_artifact():
   artifact = wandb.Artifact(MY_ASSET, "dataset")
   table = wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
   artifact.add(table, "my_table")
   return artifact
```

  </TabItem>
</Tabs>


### Configuration
A configuration dictionary called wandb_artifact_configuration can be set on an `@op`, `@asset` and `@multi_asset`. This dictionary must be passed in the decorator arguments as metadata. This configuration is required to control the IO Manager reads and writes of W&B Artifacts.

For `@op`, it’s located in the output metadata through the [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) metadata argument. 
For `@asset`, it’s located in the metadata argument on the asset. 
For `@multi_asset`, it’s located in each output metadata through the [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) metadata arguments.

The proceeding code examples demonstrate how to configure a dictionary on an `@op`, `@asset` and `@multi_asset` computations:

<Tabs
  defaultValue="op"
  values={[
    {label: 'Example for @op', value: 'op'},
    {label: 'Example for @asset', value: 'asset'},
    {label: 'Example for @multi_asset', value: 'multi_asset'},
  ]}>
  <TabItem value="op">

Example for `@op`:
```python 
@op(
   out=Out(
       metadata={
           "wandb_artifact_configuration": {
               "name": "my_artifact",
               "type": "dataset",
           }
       }
   )
)
def create_dataset():
   return [1, 2, 3]
```

  </TabItem>
  <TabItem value="asset">

Example for `@asset`:
```python
@asset(
   name="my_artifact",
   metadata={
       "wandb_artifact_configuration": {
           "type": "dataset",
       }
   },
   io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
   return [1, 2, 3]
```

You do not need to pass a name through the configuration because the @asset already has a name. The integration sets the Artifact name as the asset name.

  </TabItem>
  <TabItem value="multi_asset">

Example for `@multi_asset`:

```python
@multi_asset(
   name="create_datasets",
   outs={
       "first_table": AssetOut(
           metadata={
               "wandb_artifact_configuration": {
                   "type": "training_dataset",
               }
           },
           io_manager_key="wandb_artifacts_manager",
       ),
       "second_table": AssetOut(
           metadata={
               "wandb_artifact_configuration": {
                   "type": "validation_dataset",
               }
           },
           io_manager_key="wandb_artifacts_manager",
       ),
   },
   group_name="my_multi_asset_group",
)
def create_datasets():
   first_table = wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
   second_table = wandb.Table(columns=["d", "e"], data=[[4, 5]])

   return first_table, second_table
```

  </TabItem>
</Tabs>



Supported properties:
* `name`: (str) human-readable name for this artifact, which is how you can identify this artifact in the UI or reference it in use_artifact calls. Names can contain letters, numbers, underscores, hyphens, and dots. The name must be unique across a project. Required for `@op`.
* `type`: (str) The type of the artifact, which is used to organize and differentiate artifacts. Common types include dataset or model, but you can use any string containing letters, numbers, underscores, hyphens, and dots. Required when the output is not already an Artifact.
* `description`: (str) Free text that offers a description of the artifact. The description is markdown rendered in the UI, so this is a good place to place tables, links, etc.
* `aliases`: (list[str]) An array containing one or more aliases you want to apply on the Artifact. The integration will also add the “latest” tag to that list whether it’s set or not. This is an effective way for you to manage versioning of models and datasets.
* [`add_dirs`](../../ref/python/artifact.md#add_dir): (list[dict[str, Any]]): An array containing configuration for each local directory to include in the Artifact. It supports the same arguments as the homonymous method in the SDK.
* [`add_files`](../../ref/python/artifact.md#add_file): (list[dict[str, Any]]): An array containing configuration for each local file to include in the Artifact. It supports the same arguments as the homonymous method in the SDK.
* [`add_references`](../../ref/python/artifact.md#add_reference): (list[dict[str, Any]]): An array containing configuration for each external reference to include in the Artifact. It supports the same arguments as the homonymous method in the SDK.
* `serialization_module`: (dict) Configuration of the serialization module to be used. Refer to the Serialization section for more information.
    * `name`: (str) Name of the serialization module. Accepted values: `pickle`, `dill`, `cloudpickle`, `joblib`. The module needs to be available locally.
    * `parameters`: (dict[str, Any]) Optional arguments passed to the serialization function. It accepts the same parameters as the dump method for that module e.g. `{"compress": 3, "protocol": 4}`.

Advanced example:
```python
@asset(
   name="my_advanced_artifact",
   metadata={
       "wandb_artifact_configuration": {
           "type": "dataset",
           "description": "My *Markdown* description",
           "aliases": ["my_first_alias", "my_second_alias"],
           "add_dirs": [
               {
                   "name": "My directory",
                   "local_path": "path/to/directory",
               }
           ],
           "add_files": [
               {
                   "name": "validation_dataset",
                   "local_path": "path/to/data.json",
               },
               {
                   "is_tmp": True,
                   "local_path": "path/to/temp",
               },
           ],
           "add_references": [
               {
                   "uri": "https://picsum.photos/200/300",
                   "name": "External HTTP reference to an image",
               },
               {
                   "uri": "s3://my-bucket/datasets/mnist",
                   "name": "External S3 reference",
               },
           ],
       }
   },
   io_manager_key="wandb_artifacts_manager",
)
def create_advanced_artifact():
   return [1, 2, 3]
```



The asset is materialized with useful metadata on both sides of the integration:
* W&B side: the source integration name and version, the python version used, the pickle protocol version and more.
* Dagster side: 
    * Dagster Run ID
    * W&B Run: ID, name, path, URL
    * W&B Artifact: ID, name, type, version, size, URL
    * W&B Entity
    * W&B Project

The proceeding image demonstrates the metadata from W&B that was added to the Dagster asset. This information would not be available without the integration.

![](/images/integrations/dagster_wb_metadata.png)

The following image demonstrates how  the provided configuration was enriched with useful metadata on the W&B Artifact. This information should help for reproducibility and maintenance. It would not be available without the integration.

![](/images/integrations/dagster_inte_1.png)
![](/images/integrations/dagster_inte_2.png)
![](/images/integrations/dagster_inte_3.png)


:::info
If you use a static type checker like mypy, import the configuration type definition object using: 

```python
from dagster_wandb import WandbArtifactConfiguration
```
:::

### Using partitions

The integration natively supports [Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions).

The following is an example with a partitioned using `DailyPartitionsDefinition`.
```python
@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2023-01-01", end_date="2023-02-01"),
    name="my_daily_partitioned_asset",
    compute_kind="wandb",
    metadata={
        "wandb_artifact_configuration": {
            "type": "dataset",
        }
    },
)
def create_my_daily_partitioned_asset(context):
    partition_key = context.asset_partition_key_for_output()
    context.log.info(f"Creating partitioned asset for {partition_key}")
    return random.randint(0, 100)
```
This code will produce one W&B Artifact for each partition. They can be found in the Artifact panel (UI) under the asset name, which will be appended with the partition key, for example `my_daily_partitioned_asset.2023-01-01`, `my_daily_partitioned_asset.2023-01-02`, `my_daily_partitioned_asset.2023-01-03` and so on. Assets that are partitioned across multiple dimensions will have each dimension divided by a dot e.g. `my_asset.car.blue`.

:::caution
The integration does not allow for the materialization of multiple partitions within one run. You will need to carry out multiple runs to materialize your assets. This can be executed in Dagit when you're materializing your assets.

![](/images/integrations/dagster_multiple_runs.png)
:::

#### Advanced usage
- [Partitioned job](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [Simple partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [Multi-partitioned asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [Advanced partitioned usage](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)


## Read W&B Artifacts
Reading W&B Artifacts is similar to writing them. A configuration dictionary called `wandb_artifact_configuration` can be set on an `@op` or `@asset`. The only difference is that we must set the configuration on the input instead of the output.

For `@op`, it’s located in the input metadata through the [In](https://docs.dagster.io/_apidocs/ops#dagster.In) metadata argument. You need to 
explicitly pass the name of the Artifact.

For `@asset`, it’s located in the input metadata through the [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In metadata argument. You should not pass an Artifact name because the name of the parent asset should match it. 

If you want to have a dependency on an Artifact created outside the integration you will need to use [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset). It will always read the latest version of that asset.

The following examples demonstrate how to read an Artifact from various ops.

<Tabs
  defaultValue="op"
  values={[
    {label: 'From an @op', value: 'op'},
    {label: 'Created by another @asset', value: 'asset'},
    {label: 'Artifact created outside Dagster', value: 'outside_dagster'},
  ]}>
  <TabItem value="op">

Reading an artifact from an `@op`
```python
@op(
   ins={
       "artifact": In(
           metadata={
               "wandb_artifact_configuration": {
                   "name": "my_artifact",
               }
           }
       )
   },
   io_manager_key="wandb_artifacts_manager"
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```

  </TabItem>
  <TabItem value="asset">

Reading an artifact created by another `@asset`
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # if you don't want to rename the input argument you can remove 'key'
           key="parent_dagster_asset_name",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```

  </TabItem>
  <TabItem value="outside_dagster">

Reading an Artifact created outside Dagster:

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # the name of the W&B Artifact
   description="Artifact created outside Dagster",
   io_manager_key="wandb_artifacts_manager",
)


@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```

  </TabItem>
</Tabs>


### Configuration
The proceeding configuration is used to indicate what the IO Manager should collect and provide as inputs to the decorated functions. The following read patterns are supported.

1. To get an named object contained within an Artifact use get:

```python
@asset(
   ins={
       "table": AssetIn(
           key="my_artifact_with_table",
           metadata={
               "wandb_artifact_configuration": {
                   "get": "my_table",
               }
           },
           input_manager_key="wandb_artifacts_manager",
       )
   }
)
def get_table(context, table):
   context.log.info(table.get_column("a"))
```


2. To get the local path of a downloaded file contained within an Artifact use get_path:

```python
@asset(
   ins={
       "path": AssetIn(
           key="my_artifact_with_file",
           metadata={
               "wandb_artifact_configuration": {
                   "get_path": "name_of_file",
               }
           },
           input_manager_key="wandb_artifacts_manager",
       )
   }
)
def get_path(context, path):
   context.log.info(path)
```

3. To get the entire Artifact object (with the content downloaded locally):
```python
@asset(
   ins={
       "artifact": AssetIn(
           key="my_artifact",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def get_artifact(context, artifact):
   context.log.info(artifact.name)
```


Supported properties
* `get`: (str) Gets the W&B object located at the artifact relative name.
* `get_path`: (str) Gets the path to the file located at the artifact relative name.

### Serialization configuration
By default, the integration will use the standard [pickle](https://docs.python.org/3/library/pickle.html) module, but some objects are not compatible with it. For example, functions with yield will raise an error if you try to pickle them. 

We support more Pickle-based serialization modules ([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)). You can also use more advanced serialization like [ONNX](https://onnx.ai/) or [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) by returning a serialized string or creating an Artifact directly. The right choice will depend on your use case, please refer to the available literature on this subject.	

### Pickle-based serialization modules

:::caution
Pickling is known to be insecure. If security is a concern please only use W&B objects. We recommend signing your data and storing the hash keys in your own systems. For more complex use cases don’t hesitate to contact us, we will be happy to help.
:::

You can configure the serialization used through the `serialization_module` dictionary in the `wandb_artifact_configuration`. Please make sure the module is available on the machine running Dagster.

The integration will automatically know which serialization module to use when you read that Artifact.

The currently supported modules are pickle, dill, cloudpickle and joblib.

Here’s a simplified example where we create a “model” serialized with joblib and then use it for inference.

```python
@asset(
    name="my_joblib_serialized_model",
    compute_kind="Python",
    metadata={
        "wandb_artifact_configuration": {
            "type": "model",
            "serialization_module": {
                "name": "joblib"
            },
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_model_serialized_with_joblib():
    # This is not a real ML model but this would not be possible with the pickle module
    return lambda x, y: x + y

@asset(
    name="inference_result_from_joblib_serialized_model",
    compute_kind="Python",
    ins={
        "my_joblib_serialized_model": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        )
    },
    metadata={
        "wandb_artifact_configuration": {
            "type": "results",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def use_model_serialized_with_joblib(
    context: OpExecutionContext, my_joblib_serialized_model
):
    inference_result = my_joblib_serialized_model(1, 2)
    context.log.info(inference_result)  # Prints: 3
    return inference_result
```

### Advanced serialization formats (ONNX, PMML)
It’s common to use interchange file formats like ONNX and PMML. The integration supports those formats but it requires a bit more work than for Pickle-based serialization.

There are two different methods to use those formats.
1. Convert your model to the selected format, then return the string representation of that format as if it were a normal Python objects. The integration will pickle that string. You can then rebuild your model using that string.
2. Create a new local file with your serialized model, then build a custom Artifact with that file using the add_file configuration.

Here’s an example of a Scikit-learn model being serialized using ONNX.

```python
import numpy
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dagster import AssetIn, AssetOut, asset, multi_asset

@multi_asset(
    compute_kind="Python",
    outs={
        "my_onnx_model": AssetOut(
            metadata={
                "wandb_artifact_configuration": {
                    "type": "model",
                }
            },
            io_manager_key="wandb_artifacts_manager",
        ),
        "my_test_set": AssetOut(
            metadata={
                "wandb_artifact_configuration": {
                    "type": "test_set",
                }
            },
            io_manager_key="wandb_artifacts_manager",
        ),
    },
    group_name="onnx_example",
)
def create_onnx_model():
    # Inspired from https://onnx.ai/sklearn-onnx/

    # Train a model.
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # Write artifacts (model + test_set)
    return onx.SerializeToString(), {"X_test": X_test, "y_test": y_test}

@asset(
    name="experiment_results",
    compute_kind="Python",
    ins={
        "my_onnx_model": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        ),
        "my_test_set": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        ),
    },
    group_name="onnx_example",
)
def use_onnx_model(context, my_onnx_model, my_test_set):
    # Inspired from https://onnx.ai/sklearn-onnx/

    # Compute the prediction with ONNX Runtime
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

### Using partitions

The integration natively supports [Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions).

You can selectively read one, multiple or all partitions of an asset.

All partitions are provided in a dictionary, with the key and value representing the partition key and the Artifact content, respectively.

<!-- 
<Tabs
  defaultValue="op"
  values={[
    {label: 'From an @op', value: 'all'},
  ]}>
  <TabItem value="all">
  Read all partitions
```python
@asset(
    compute_kind="wandb",
    ins={"my_daily_partitioned_asset": AssetIn()},
    output_required=False,
)
def read_all_partitions(context, my_daily_partitioned_asset):
    for partition, content in my_daily_partitioned_asset.items():
        context.log.info(f"partition={partition}, content={content}")
```
  </TabItem>
</Tabs> -->

<Tabs
  defaultValue="all"
  values={[
    {label: 'Read all partitions', value: 'all'},
    {label: 'Read specific partitions', value: 'specific'},
  ]}>
  <TabItem value="all">

It reads all partitions of the upstream `@asset`, which are given as a dictionary. In this dictionary, the key and value correlate to the partition key and the Artifact content, respectively.
```python
@asset(
    compute_kind="wandb",
    ins={"my_daily_partitioned_asset": AssetIn()},
    output_required=False,
)
def read_all_partitions(context, my_daily_partitioned_asset):
    for partition, content in my_daily_partitioned_asset.items():
        context.log.info(f"partition={partition}, content={content}")
```
  </TabItem>
  <TabItem value="specific">

The `AssetIn`'s `partition_mapping` configuration allows you to choose specific partitions. In this case, we are employing the `TimeWindowPartitionMapping`.
```python
@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2023-01-01", end_date="2023-02-01"),
    compute_kind="wandb",
    ins={
        "my_daily_partitioned_asset": AssetIn(
            partition_mapping=TimeWindowPartitionMapping(start_offset=-1)
        )
    },
    output_required=False,
)
def read_specific_partitions(context, my_daily_partitioned_asset):
    for partition, content in my_daily_partitioned_asset.items():
        context.log.info(f"partition={partition}, content={content}")
```
  </TabItem>
</Tabs>

The configuration object, `metadata`, is used to configure how Weights & Biases (wandb) interacts with different artifact partitions in your project.

The object `metadata` contains a key named `wandb_artifact_configuration` which further contains a nested object `partitions`.

The `partitions` object maps the name of each partition to its configuration. The configuration for each partition can specify how to retrieve data from it. These configurations can contain different keys, namely `get`, `version`, and `alias`, depending on the requirements of each partition.

**Configuration keys**

1. `get`:
The `get` key specifies the name of the W&B Object (Table, Image...) where to fetch the data. 
2. `version`:
The `version` key is used when you want to fetch a specific version for the Artifact. 
3. `alias`:
The `alias` key allows you to get the Artifact by its alias.

**Wildcard configuration**

The wildcard `"*"` stands for all non-configured partitions. This provides a default configuration for partitions that are not explicitly mentioned in the `partitions` object. 

For example, 

```python
"*": {
    "get": "default_table_name",
},
```
This configuration means that for all partitions not explicitly configured, data is fetched from the table named `default_table_name`.

**Specific partition configuration**

You can override the wildcard configuration for specific partitions by providing their specific configurations using their keys.

For example,

```python
"yellow": {
    "get": "custom_table_name",
},
```

This configuration means that for the partition named `yellow`, data will be fetched from the table named `custom_table_name`, overriding the wildcard configuration.

**Versioning and aliasing**

For versioning and aliasing purposes, you can provide specific `version` and `alias` keys in your configuration.

For versions,

```python
"orange": {
    "version": "v0",
},
```

This configuration will fetch data from the version `v0` of the `orange` Artifact partition.

For aliases,

```python
"blue": {
    "alias": "special_alias",
},
```

This configuration will fetch data from the table `default_table_name` of the Artifact partition with the alias `special_alias` (referred to as `blue` in the configuration).

### Advanced usage
To view advanced usage of the integration please refer to the following full code examples:
* [Advanced usage example for assets](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py) 
* [Partitioned job example](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [Linking a model to the Model Registry](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)


## Using W&B Launch

:::caution
Beta product in active development
Interested in Launch? Reach out to your account team to talk about joining the customer pilot program for W&B Launch.
Pilot customers need to use AWS EKS or SageMaker to qualify for the beta program. We ultimately plan to support additional platforms.
:::

Before continuing, we recommend you to have a good understanding of how to use W&B Launch. Consider, reading the Guide on Launch: https://docs.wandb.ai/guides/launch.

The Dagster integration helps with:
* Running one or multiple Launch agents in your Dagster instance.
* Executing local Launch jobs within your Dagster instance.
* Remote Launch jobs on-prem or in a cloud.

### Launch agents
The integration provides an importable `@op` called `run_launch_agent`. It starts a Launch Agent and runs it as a long running process until stopped manually.

Agents are processes that poll launch queues and execute the jobs (or dispatch them to external services to be executed) in order.

Refer to the [reference documentation](../launch/intro.md) for configuration

You can also view useful descriptions for all properties in Launchpad.

![](/images/integrations/dagster_launch_agents.png)

Simple example
```python
# add this to your config.yaml
# alternatively you can set the config in Dagit's Launchpad or JobDefinition.execute_in_process
# Reference: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # replace this with your W&B entity
     project: my_project # replace this with your W&B project
ops:
 run_launch_agent:
   config:
     max_jobs: -1
     queues: 
       - my_dagster_queue

from dagster_wandb.launch.ops import run_launch_agent
from dagster_wandb.resources import wandb_resource

from dagster import job, make_values_resource

@job(
   resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
   },
)
def run_launch_agent_example():
   run_launch_agent()
```

### Launch jobs
The integration provides an importable `@op` called `run_launch_job`. It executes your Launch job.

A Launch job is assigned to a queue in order to be executed. You can create a queue or use the default one. Make sure you have an active agent listening to that queue. You can run an agent inside your Dagster instance but can also consider using a deployable agent in Kubernetes.

Refer to the [reference documentation](../launch/intro.md) for configuration.

You can also view useful descriptions for all properties in Launchpad.

![](/images/integrations/dagster_launch_jobs.png)


Simple example
```python
# add this to your config.yaml
# alternatively you can set the config in Dagit's Launchpad or JobDefinition.execute_in_process
# Reference: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # replace this with your W&B entity
     project: my_project # replace this with your W&B project
ops:
 my_launched_job:
   config:
     entry_point:
       - python
       - train.py
     queue: my_dagster_queue
     uri: https://github.com/wandb/example-dagster-integration-with-launch


from dagster_wandb.launch.ops import run_launch_job
from dagster_wandb.resources import wandb_resource

from dagster import job, make_values_resource


@job(resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
   },
)
def run_launch_job_example():
   run_launch_job.alias("my_launched_job")() # we rename the job with an alias
```

## Best practices

1. Use the IO Manager to read and write Artifacts. 
You should never need to use [`Artifact.download()`](../../ref/python/artifact.md#download) or [`Run.log_artifact()`](../../ref/python/run.md#log_artifact) directly. Those methods are handled by integration. Simply return the data you wish to store in Artifact and let the integration do the rest. This will provide better lineage for the Artifact in W&B.

2. Only build an Artifact object yourself for complex use cases.
Python objects and W&B objects should be returned from your ops/assets. The integration handles bundling the Artifact.
For complex use cases, you can build an Artifact directly in a Dagster job. We recommend you pass an Artifact object to the integration for metadata enrichment such as the source integration name and version, the python version used, the pickle protocol version and more.

3. Add files, directories and external references to your Artifacts through the metadata.
Use the integration `wandb_artifact_configuration` object to add any file, directory or external references (Amazon S3, GCS, HTTP…). See the advanced example in the [Artifact configuration section](#configuration-1) for more information.

4. Use an @asset instead of an @op when an Artifact is produced.
Artifacts are assets. It is recommended to use an asset when Dagster maintains that asset. This will provide better observability in the Dagit Asset Catalog.

5. Use a SourceAsset to consume an Artifact created outside Dagster.
This allows you to take advantage of the integration to read externally created Artifacts. Otherwise, you can only use Artifacts created by the integration.

6. Use W&B Launch to orchestrate training on dedicated compute for large models.
You can train small models inside your Dagster cluster and you can run Dagster in a Kubernetes cluster with GPU nodes. We recommend using W&B Launch for large model training. This will prevent overloading your instance and provide access to more adequate compute. 

7. When experiment tracking within Dagster, set your W&B Run ID to the value of your Dagster Run ID.
We recommend that you both: make the [Run resumable](../runs/resuming.md) and set the W&B Run ID to the Dagster Run ID or to a string of your choice. Following this recommendation ensures your W&B metrics and W&B Artifacts are stored in the same W&B Run when you train models inside of Dagster.


Either set the W&B Run ID to the Dagster Run ID.
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```


Or choose your own W&B Run ID and pass it to the IO Manager configuration.
```python
wandb.init(
    id="my_resumable_run_id",
    resume="allow",
    ...
)

@job(
   resource_defs={
       "io_manager": wandb_artifacts_io_manager.configured(
           {"wandb_run_id": "my_resumable_run_id"}
       ),
   }
)
```

8. Only collect data you need with get or get_path for large W&B Artifacts.
By default, the integration will download an entire Artifact. If you are using very large artifacts you might want to only collect the specific files or objects you need. This will improve speed and resource utilization.

9. For Python objects adapt the pickling module to your use case.
By default, the W&B integration will use the standard [pickle](https://docs.python.org/3/library/pickle.html) module. But some objects are not compatible with it. For example, functions with yield will raise an error if you try to pickle them. W&B supports other Pickle-based serialization modules ([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib)). 

You can also use more advanced serialization like [ONNX](https://onnx.ai/) or [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) by returning a serialized string or creating an Artifact directly. The right choice will depend on your use case, refer to the available literature on this subject.
