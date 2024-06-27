---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up Launch

This page describes the high-level steps required to set up W&B Launch:

1. **Set up a queue**: Queues are FIFO and possess a queue configuration. A queue's configuration controls where and how jobs are executed on a target resource.
2. **Set up an agent**: Agents run on your machine/infrastructure and poll one or more queues for launch jobs. When a job is pulled, the agent ensures that the image is built and available. The agent then submits the job to the target resource.


## Set up a queue
Launch queues must be configured to point to a specific target resource along with any additional configuration specific to that resource. For example, a launch queue that points to a Kubernetes cluster might include environment variables or set a custom namespace its launch queue configuration. When you create a queue, you will specify both the target resource you want to use and the configuration for that resource to use.

When an agent receives a job from a queue, it also receives the queue configuration. When the agent submits the job to the target resource, it includes the queue configuration along with any overrides from the job itself. For example, you can use a job configuration to specify the Amazon SageMaker instance type for that job instance only. In this case, it is common to use [queue config templates](./setup-queue-advanced.md#configure-queue-template) as the end user interface. 

### Create a queue
1. Navigate to Launch App at [wandb.ai/launch](https://wandb.ai/launch). 
2. Click the **create queue** button on the top right of the screen. 

![](/images/launch/create-queue.gif)

3. From the **Entity** dropdown menu, select the entity the queue will belong to. 
  :::tip
  If you choose a team entity, all members of the team will be able to send jobs to this queue. If you choose a personal entity (associated with a username), W&B will create a private queue that only that user can use.
  :::
4. Provide a name for your queue in the **Queue** field. 
5. From the **Resource** dropdown, select the compute resource you want jobs added to this queue to use.
6. Choose whether to allow **Prioritization** for this queue.  If prioritization is enabled, a user on your team can define a priority for their launch job when they enqueue them.  Higher priority jobs are executed before lower priority jobs.
7. Provide a resource configuration in either JSON or YAML format in the **Configuration** field. The structure and semantics of your configuration document will depend on the resource type that the queue is pointing to. For more details, see the dedicated set up page for your target resource.




## Set up a launch agent
Launch agents are long running processes that poll one or more launch queues for jobs. Launch agents dequeue jobs in first in, first out (FIFO) order or in priority order depending on the queues they pull from.  When an agent dequeues a job from a queue, it optionally builds an image for that job. The agent then submits the job to the target resource along with configuration options specified in the queue configuration.

<!-- Future: Insert image -->

:::info
Agents are highly flexible and can be configured to support a wide variety of use cases. The required configuration for your agent will depend on your specific use case. See the dedicated page for [Docker](./setup-launch-docker.md), [Amazon SageMaker](./setup-launch-sagemaker.md), [Kubernetes](./setup-launch-kubernetes.md), or [Vertex AI](./setup-vertex.md).
:::

:::tip
W&B recommends you start agents with a [service account's](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful) API key, rather than a specific user's API key. There are two benefits to using a service account's API key:
1. The agent isn't dependent on an individual user.
2. The author associated with a run created through Launch is viewed by Launch as the user who submitted the launch job, rather than the user associated with the agent.
:::

### Agent configuration
Configure the launch agent with a YAML file named `launch-config.yaml`. By default, W&B checks for the config file in `~/.config/wandb/launch-config.yaml`. You can optionally specify a different directory when you activate the launch agent.

The contents of your launch agent's configuration file will depend on your launch agent's environment, the launch queue's target resource, Docker builder requirements, cloud registry requirements, and so forth. 

Independent of your use case, there are core configurable options for the launch agent:
* `max_jobs`: maximum number of jobs the agent can execute in parallel 
* `entity`: the entity that the queue belongs to
* `queues`: the name of one or more queues for the agent to watch

:::tip
You can use the W&B CLI to specify universal configurable options for the launch agent (instead of the config YAML file): maximum number of jobs, W&B entity, and launch queues. See the [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) command for more information.
:::


The following YAML snippet shows how to specify core launch agent config keys:

```yaml title="launch-config.yaml"
# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

entity: <entity-name>

# List of queues to poll.
queues:
  - <queue-name>
```


### Configure a container builder
The launch agent can be configured to build images. You must configure the agent to use a container builder if you intend to use launch jobs created from git repositories or code artifacts. See the [Create a launch job](./create-launch-job.md) for more information on how to create a launch job. 

W&B Launch supports three builder options:

* Docker: The Docker builder uses a local Docker daemon to build images.
* [Kaniko](https://github.com/GoogleContainerTools/kaniko):  Kaniko is a Google project that enables image building in environments where a Docker daemon is unavailable. 
* Noop: The agent will not try to build jobs, and instead only pull pre-built images.

:::tip
Use the Kaniko builder if your agent is polling in an environment where a Docker daemon is unavailable (for example, a Kubernetes cluster).

See the [Set up Kubernetes](./setup-launch-kubernetes.md) for details about the Kaniko builder.
:::

To specify an image builder, include the builder key in your agent configuration. For example, the following code snippet shows a portion of the launch config (`launch-config.yaml`) that specifies to use Docker or Kaniko:

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### Configure a container registry
In some cases, you might want to connect a launch agent to a cloud registry. Common scenarios where you might want to connect a launch agent to a cloud registry include:

* You want to run a job in an envirnoment other than where you built it, such as a powerful workstation or cluster.
* You want to use the agent to build images and run these images on Amazon SageMaker or VertexAI.
* You want the launch agent to provide credentials to pull from an image repository.

To learn more about how to configure the agent to interact with a container registry, see the [Advanced agent set](./setup-agent-advanced.md) up page.

## Activate the launch agent
Activate the launch agent with the `launch-agent` W&B CLI command:

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

In some use cases, you might want to have a launch agent polling queues from within a Kubernetes cluster. See the [Advanced queue set up page](./setup-queue-advanced.md) for more information. 


