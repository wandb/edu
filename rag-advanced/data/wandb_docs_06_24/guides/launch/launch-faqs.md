---
description: Answers to frequently asked question about W&B Launch.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch FAQs

<head>
  <title>Frequently Asked Questions About Launch</title>
</head>


## Getting Started


### I do not want W&B to build a container for me, can I still use Launch?
  
Yes. Run the following to launch a pre-built docker image. Replace the items in the `<>` with your information:

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```  

This will build a job when you create a run.

Or you can make a job from an image:

```bash
wandb job create image <image-name> -p <project> -e <entity>
```

### Are there best practices for using Launch effectively?

  1. Create your queue before you start your agent, so that you can set your agent to point to it easily.  If you don’t do this, your agent will give errors and not work until you add a queue.
  2. Create a W&B service account to start up the agent, so that it's not tied to an individual user account.
  3. Use `wandb.config` to read and write your hyperparameters, so that they can be overwritten when re-running a job. Check out [this guide](https://docs.wandb.ai/guides/launch/create-launch-job#making-your-code-job-friendly) if you use argsparse.

### I do not like clicking- can I use Launch without going through the UI?
  
  Yes. The standard `wandb` CLI includes a `launch` subcommand that you can use to launch your jobs. For more info, try running

  ```bash
  wandb launch --help
  ```

### Can Launch automatically provision (and spin down) compute resources for me in the target environment?

This depends on the environment, we are able to provision resources in SageMaker, and Vertex. In Kubernetes, autoscalers can be used to automatically spin up and spin down resources when required. The Solution Architects at W&B are happy to work with you to configure your underlying Kubernetes infrastructure to facilitate retries, autoscaling, and use of spot instance node pools. Reach out to support@wandb.com or your shared Slack channel.

### Is `wandb launch -d` or `wandb job create image` uploading a whole docker artifact and not pulling from a registry? 

No. The  `wandb launch -d` command will not upload to a registry for you. You need to upload your image to a registry yourself. Here are the general steps:

1. Build an image. 
2. Push the image to a registry.

The workflow looks like:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

From there, the launch agent will spin up a job pointing to that container.  See [Advanced agent setup](./setup-agent-advanced.md#agent-configuration) for examples of how to give the agent access to pull an image from a container registry.

For Kubernetes, the Kubernetes cluster pods will need access to the registry you are pushing to. 


### Can I specify a Dockerfile and let W&B build a Docker image for me?
Yes. This is particularly useful if you have a lot of requirements that do not change often, but you have a codebase that does change often.

:::important
Ensure your Dockerfile is formatted to use mounts. For more information, see [Mounts documentation on the Docker Docs website](https://docs.docker.com/build/guide/mounts/). 
:::

Once your Dockerfile is configured, you can then specify your Dockerfile in one of three ways to W&B:

* Use Dockerfile.wandb
* W&B CLI
* W&B App


<Tabs
  defaultValue="dockerfile"
  values={[
    {label: 'Dockerfile.wandb', value: 'dockerfile'},
    {label: 'W&B CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="dockerfile">

Include a file called `Dockerfile.wandb` in the  same directory as the W&B run’s entrypoint.  W&B will use `Dockerfile.wandb` instead of W&B’s built-in Dockerfile.


  </TabItem>
  <TabItem value="cli">

Provide the `--dockerfile` flag when you call queue a launch job with the [`wandb launch`](../../ref/cli/wandb-launch.md) command:

```bash
wandb launch --dockerfile path/to/Dockerfile
```


  </TabItem>
  <TabItem value="app">


When you add a job to a queue on the W&B App, provide the path to your Dockerfile in the **Overrides** section. More specifically, provide it as a key-value pair where `"dockerfile"` is the key and the value is the path to your Dockerfile. 

For example, the following JSON shows how to include a Dockerfile that is within a local directory:

```json title="Launch job W&B App"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```

  </TabItem>
</Tabs>



## Permissions and Resources

### How do I control who can push to a queue?

Queues are scoped to a team of users. You define the owning entity when you create the queue.  To restrict access, you can change the team membership.

### What permissions does the agent require in Kubernetes?
“The following kubernetes manifest will create a role named
`wandb-launch-agent` in the`wandb`namespace. This role will allow the agent to create pods, configmaps, secrets, and pods/log in the `wandb` namespace. The `wandb-cluster-role` will allow the agent to create pods, pods/log, secrets, jobs, and jobs/status in any namespace of your choice.”

### Does Launch support parallelization?  How can I limit the resources consumed by a job?
   
Yes, Launch supports scaling jobs across mulitple GPUs and multiple nodes.  See [this guide](https://docs.wandb.ai/tutorials/volcano) for details.

On an inter-job level, an individual launch agent is configured with a `max_jobs` parameter that determines how many jobs that agent can run simultaneously. Additionally, you can point to as many agents as you want at a particular queue, so long as those agents are connected to an infrastructure that they can launch into.
  
You can limit the CPU/GPU, memory, and other requirements at the launch queue or job run level, in the resource config. For more information about setting up queues with resource limits on Kubernetes see [here](https://docs.wandb.ai/guides/launch/kubernetes#queue-configuration). 

   
For sweeps, in the SDK you can add a block to the queue config

```yaml title="queue config"
  scheduler:
    num_workers: 4
```
To limit the number of concurrent runs from a sweep that will be run in parallel.

### When using Docker queues to run multiple jobs that download the same artifact with `use_artifact`, do we re-download the artifact for every single run of the job, or is there any caching going on under the hood?

There is no caching; each job is independent.  However, there are ways to configure your queue/agent where it mounts a shared cache.  You can achieve this via docker args in the queue config.

As a special case, you can also mount the W&B artifacts cache as a persistent volume.


### Can you specify secrets for jobs/automations? For instance, an API key which you do not wish to be directly visible to users?

Yes. The suggested way is:

  1. Add the secret as a vanilla k8s secret in the namespace where the runs will be created. something like `kubectl create secret -n <namespace> generic <secret_name> <secret value>`

 2. Once that secret is created, you can specify a queue config to inject the secret when runs start. The end users cannot see the secret, only cluster admins can.

### How can admins restrict what ML engineers have access to modify? For example, changing an image tag may be fine but other job settings may not be.
  
This can be controlled by [queue config templates](./setup-queue-advanced.md), which expose certain queuefields for non-team-admin users to edit within limits defined by admin users. Only team admins can create or edit queues, including defining which fields are exposed and the limits for them. 

### How does W&B Launch build images?

The steps taken to build an image vary depending on the source of the job being run, and whether the resource configuration specifies an accelerator base image.

:::note
When specifying a queue config, or submitting a job, a base accelerator image can be provided in the queue or job resource configuration:
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
:::

During the build process the following actions are taken dependant on the type of job and accelerator base image provided:

|                                                     | Install python using apt | Install python packages | Create a user and workdir | Copy code into image | Set entrypoint |
|-----------------------------------------------------|:------------------------:|:-----------------------:|:-------------------------:|:--------------------:|:--------------:|
| Job sourced from git                                |                          |            X            |             X             |           X          |        X       |
| Job sourced from code                               |                          |            X            |             X             |           X          |        X       |
| Job sourced from git and provided accelerator image |             X            |            X            |             X             |           X          |        X       |
| Job sourced from code and provided accelerator image|             X            |            X            |             X             |           X          |        X       |
| Job sourced from image                              |                          |                         |                           |                      |                |


### What requirements does the accelerator base image have?
For jobs that use an accelerator, an accelerator base image with the required accelerator components installed can be provided. Other requirements for the provided accelerator image include:
- Debian compatibility (the Launch Dockerfile uses apt-get to fetch python )
- Compatibility CPU & GPU hardware instruction set (Make sure your CUDA version is supported by the GPU you intend on using)
- Compatibility between the accelerator version you provide and the packages installed in your ML algorithm
- Packages installed that require extra steps for setting up compatibility with hardware

### How do I make W&B Launch work with Tensorflow on GPU?
For jobs that use tensorflow on GPU, you may also need to specify a custom base image for the container build that the agent will perform in order for your runs to properly utilize GPUs. This can be done by adding an image tag under the `builder.accelerator.base_image` key to the resource configuration. For example:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

Note prior to wandb version: 0.15.6 use `cuda` instead of `accelerator` as the parent key to `base_image`.

### Can you use a custom repository for packages when Launch builds the image?

Yes. To do so, add the following line to your `requirements.txt` and replace the values passed to `index-url` and `extra-index-url` with your own values:

```text
----index-url=https://xyz@<your-repo-host> --extra-index-url=https://pypi.org/simple
```
 
The `requirements.txt` needs to be defined at the base root of the job.

## Automatic run re-queuing on preemption

In some cases, it can be useful to set up jobs to be resumed after they are interrupted.  For example, you might run broad hyperparameter sweeps on spot instances, and want them to pick up again when more spot instances spin up.  Launch can support this configuration on Kubernetes clusters.

If your Kubernetes queue is running a job on a node that’s pre-empted by a scheduler, the job will be automatically added back to the end of the queue so it can resume later. This resumed run will have the same name as the original, and can be followed from the same page in the UI as the original. A job can be automatically re-queued this way up to five times. 

Launch detects whether a pod is preempted by a scheduler by checking if the pod has the condition `DisruptionTarget` with one of the following reasons:

- `EvictionByEvictionAPI`
- `PreemptionByScheduler`
- `TerminationByKubelet`

If your job’s code is structured to allow resuming, it will enable these re-queued runs to pick up where they left off. Otherwise, runs will start from the beginning when they are re-queued. See our guide for [resuming runs](../runs/resuming.md) for more info.   

There is currently no way to opt out of automatic run re-queuing for preempted nodes. However, if you delete a run from the UI or delete the node directly, it will not be re-queued.

Automatic run re-queuing is currently only available on Kubernetes queues; Sagemaker and Vertex are not yet supported.
