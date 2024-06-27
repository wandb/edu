---
displayed_sidebar: default
---

# Advanced queue set up
The following page describes how to configure additional launch queue options.

## Set up queue config templates
Administer and manage guardrails on compute consumption with Queue Config Templates. Set defaults, minimums, and maximum values for fields such as memory consumption, GPU, and runtime duration.

After you configure a queue with config templates, members of your team can alter fields you defined only within the specified range you defined.

### Configure queue template
You can configure a queue template on an existing queue or create a new queue.  

1. Navigate to the Launch App at [https://wandb.ai/launch](https://wandb.ai/launch).
2. Select **View queue** next to the name of the queue you want to add a template to.
3. Select the **Config** tab. This will show information about your queue such as when the queue was created, the queue config, and existing launch-time overrides.
4. Navigate to the **Queue config** section.
5. Identify the config key-values you want to create a template for. 
6. Replace the value in the config with a template field. Template fields take the form of `{{variable-name}}`. 
7. Click on the **Parse configuration** button. When you parse your configuration, W&B will automatically create tiles below the queue config for each template you created.
8. For each tile generated, you must first specify the data type (string, integer, or float) the queue config can allow. To do this, select the data type from the **Type** dropdown menu.
9. Based on your data type, complete the fields that appear within each tile.
10. Click on **Save config**.


For example, suppose you want to create a template that limits which AWS instances your team can use. Before you add a template field, your queue config might look something similar to:

```yaml title="launch config"
RoleArn: arn:aws:iam:region:account-id:resource-type/resource-id
ResourceConfig:
  InstanceType: ml.m4.xlarge
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: s3://bucketname
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

When you add a template field for the `InstanceType`, your config will look like:

```yaml title="launch config"
RoleArn: arn:aws:iam:region:account-id:resource-type/resource-id
ResourceConfig:
  InstanceType: "{{aws_instance}}"
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: s3://bucketname
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```


Next, you click on **Parse configuration**. A new tile labeled `aws-instance` will appear underneath the **Queue config**. 

From there, you select String as the datatype from the **Type** dropdown. This will populate fields where you can specify values a user can choose from. For example, in the following image the admin of the team configured two different AWS instance types that users can choose from (`ml.m4.xlarge` and `ml.p3.xlarge`):

![](/images/launch/aws_template_example.png)



## Dynamically configure launch jobs
Queue configs can be dynamically configured using macros that are evaluated when the agent dequeues a job from the queue. You can set the following macros:

| Macro             | Description                                           |
|-------------------|-------------------------------------------------------|
| `${project_name}` | The name of the project the run is being launched to. |
| `${entity_name}`  | The owner of the project the run being launched to.   |
| `${run_id}`       | The id of the run being launched.                     |
| `${run_name}`     | The name of the run that is launching.                |
| `${image_uri}`    | The URI of the container image for this run.          |

:::info
Any custom macro not listed in the preceding table (for example `${MY_ENV_VAR}`), is substituted with an environment variable from the agent's environment.
:::

## Use the launch agent to build images that execute on accelerators (GPUs)
You might need to specify an accelerator base image if you use launch to build images that are executed in an accelerator environment.

This accelerator base image must satisfy the following requirements:

- Debian compatibility (the Launch Dockerfile uses apt-get to fetch python)
- Compatibility CPU & GPU hardware instruction set (Make sure your CUDA version is supported by the GPU you intend on using)
- Compatibility between the accelerator version you provide and the packages installed in your ML algorithm
- Packages installed that require extra steps for setting up compatibility with hardware

### How to use GPUs with TensorFlow

Ensure TensorFlow properly utilizes your GPU. To accomplish this, specify a Docker image and its image tag for the `builder.accelerator.base_image` key in the queue resource configuration.

For example, the `tensorflow/tensorflow:latest-gpu` base image ensures TensorFlow properly uses your GPU. This can be configured using the resource configuration in the queue.

The following JSON snippet demonstrates how to specify the TensorFlow base image in your queue config:

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```