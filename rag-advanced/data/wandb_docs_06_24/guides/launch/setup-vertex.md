---
displayed_sidebar: default
---

# Set up Vertex AI

You can use W&B Launch to submit jobs for execution as Vertex AI training jobs. With Vertex AI training jobs, you can train machine learning models using either provided, or custom algorithms on the Vertex AI platform. Once a launch job is initiated, Vertex AI manages the underlying infrastructure, scaling, and orchestration.

W&B Launch works with Vertex AI through the `CustomJob` class in the `google-cloud-aiplatform` SDK. The parameters of a `CustomJob` can be controlled with the launch queue configuration. Vertex AI cannot be configured to pull images from a private registry outside of GCP. This means that you must store container images in GCP or in a public registry if you want to use Vertex AI with W&B Launch. See the Vertex AI documentation for more information on making container images accessible to Vertex jobs.

<!-- Component Diagram of Launch in Vertex AI -->

## Prerequisites

1. **Create or access a GCP project with the Vertex AI API enabled.** See the [GCP API Console docs](https://support.google.com/googleapi/answer/6158841?hl=en) for more information on enabling an API.
2. **Create a GCP Artifact Registry repository** to store images you want to execute on Vertex. See the [GCP Artifact Registry documentation](https://cloud.google.com/artifact-registry/docs/overview) for more information.
3. **Create a staging GCS bucket** for Vertex AI to store its metadata. Note that this bucket must be in the same region as your Vertex AI workloads in order to be used as a staging bucket. The same bucket can be used for staging and build contexts.
4. **Create a service account** with the necessary permissions to spin up Vertex AI jobs. See the [GCP IAM documentation](https://cloud.google.com/iam/docs/creating-managing-service-accounts) for more information on assigning permissions to service accounts.
5. **Grant your service account permission to manage Vertex jobs**

| Permission                     | Resource Scope        | Description                                                                              |
| ------------------------------ | --------------------- | ---------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create` | Specified GCP Project | Allows creation of new machine learning jobs within the project.                         |
| `aiplatform.customJobs.list`   | Specified GCP Project | Allows listing of machine learning jobs within the project.                              |
| `aiplatform.customJobs.get`    | Specified GCP Project | Allows retrieval of information about specific machine learning jobs within the project. |

:::info
If you want your Vertex AI workloads to assume the identity of a non-standard service account, refer to the Vertex AI documentation for instructions on service account creation and necessary permissions. The `spec.service_account` field of the launch queue configuration can be used to select a custom service account for your W&B runs.
:::

## Configure a queue for Vertex AI

The queue configuration for Vertex AI resources specify inputs to the `CustomJob` constructor in the Vertex AI Python SDK, and the `run` method of the `CustomJob`. Resource configurations are stored under the `spec` and `run` keys:

- The `spec` key contains values for the named arguments of the [`CustomJob` constructor](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1beta1/projects.locations.customJobs#CustomJob.FIELDS.spec) in the Vertex AI Python SDK.
- The `run` key contains values for the named arguments of the `run` method of the `CustomJob` class in the Vertex AI Python SDK.

Customizations of the execution environment happens primarily in the `spec.worker_pool_specs` list. A worker pool spec defines a group of workers that will run your job. The worker spec in the default config asks for a single `n1-standard-4` machine with no accelerators. You can change the machine type, accelerator type and count to suit your needs.

For more information on available machine types and accelerator types, see the [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec).

## Create a queue

Create a queue in the W&B App that uses Vertex AI as its compute resource:

1. Navigate to the [Launch page](https://wandb.ai/launch).
2. Click on the **Create Queue** button.
3. Select the **Entity** you would like to create the queue in.
4. Provide a name for your queue in the **Name** field.
5. Select **GCP Vertex** as the **Resource**.
6. Within the **Configuration** field, provide information about your Vertex AI `CustomJob` you defined in the previous section. By default, W&B will populate a YAML and JSON request body similar to the following:

```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-4
        accelerator_type: ACCELERATOR_TYPE_UNSPECIFIED
        accelerator_count: 0
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}
  staging_bucket: <REQUIRED>
run:
  restart_job_on_worker_restart: false
```

7. After you configure your queue, click on the **Create Queue** button.

You must at minimum specify:

- `spec.worker_pool_specs` : non-empty list of worker pool specifications.
- `spec.staging_bucket` : GCS bucket to be used for staging Vertex AI assets and metadata.

:::caution
Some of the Vertex AI docs show worker pool specifications with all keys in camel case,for example, ` workerPoolSpecs`. The Vertex AI Python SDK uses snake case for these keys, for example `worker_pool_specs`.

Every key in the launch queue configuration should use snake case.
:::

## Configure a launch agent

The launch agent is configurable through a config file that is, by default, located at `~/.config/wandb/launch-config.yaml`.

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

If you want the launch agent to build images for you that are executed in Vertex AI, see [Advanced agent set up](./setup-agent-advanced.md).

## Set up agent permissions

There are multiple methods to authenticate as this service account. This can be achieved through Workload Identity, a downloaded service account JSON, environment variables, the Google Cloud Platform command-line tool, or a combination of these methods.
