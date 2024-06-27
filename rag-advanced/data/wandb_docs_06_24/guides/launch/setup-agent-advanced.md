---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Advanced agent setup

This guide provides information on how to set up the W&B Launch agent to build container images in different environments.

:::info
Build is only required for git and code artifact jobs. Image jobs do not require build.

See [Create a launch job](./create-launch-job.md) for more information on job types.
:::

## Builders

The Launch agent can build images using [Docker](https://docs.docker.com/) or [Kaniko](https://github.com/GoogleContainerTools/kaniko).

* Kaniko: builds a container image in Kubernetes without running the build as a privileged container.
* Docker: builds a container image by executing a `docker build` command locally.

The builder type can be controlled by the `builder.type` key in the launch agent config to either `docker`, `kaniko`, or `noop` to disable build. By default, the agent helm chart sets the `builder.type` to `noop`. Additional keys in the `builder` section will be used to configure the build process.

If no builder is specified in the agent config and a working `docker` CLI is found, the agent will default to using Docker. If Docker is not available the agent will default to `noop`.

:::tip
Use Kaniko for building images in a Kubernetes cluster. Use Docker for all other cases.
:::


## Pushing to a container registry

The launch agent tags all images it builds with a unique source hash. The agent pushes the image to the registry specified in the `builder.destination` key.

For example, if the `builder.destination` key is set to `my-registry.example.com/my-repository`, the agent will tag and push the image to `my-registry.example.com/my-repository:<source-hash>`. If the image exists in the registry, the build is skipped.

### Agent configuration

If you are deploying the agent via our Helm chart, the agent config should be provided in the `agentConfig` key in the `values.yaml` file.

If you are invoking the agent yourself with `wandb launch-agent`, you can provide the agent config as a path to a YAML file with the `--config` flag. By default, the config will be loaded from `~/.config/wandb/launch-config.yaml`.

Within your launch agent config (`launch-config.yaml`), provide the name of the target resource environment and the container registry for the `environment` and `registry` keys, respectively.

The following tabs demonstrates how to configure the launch agent based on your environment and registry.

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

The AWS environment configuration requires the region key. The region should be the AWS region that the agent runs in. 

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # URI of the ECR repository where the agent will store images.
  # Make sure the region matches what you have configured in your
  # environment.
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # If using Kaniko, specify the S3 bucket where the agent will store the
  # build context.
  build-context-store: s3://<bucket-name>/<path>
```

The agent uses boto3 to load the default AWS credentials. See the [boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) for more information on how to configure default AWS credentials.

  </TabItem>
  <TabItem value="gcp">

The Google Cloud environment requires region and project keys. Set `region` to the region that the agent runs in. Set `project` to the Google Cloud project that the agent runs in. The agent uses `google.auth.default()` in Python to load the default credentials.

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # URI of the Artifact Registry repository and image name where the agent
  # will store images. Make sure the region and project match what you have
  # configured in your environment.
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # If using Kaniko, specify the GCS bucket where the agent will store the
  # build context.
  build-context-store: gs://<bucket-name>/<path>
```

See the [`google-auth` documentation](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) for more information on how to configure default GCP credentials so they are available to the agent.

  </TabItem>
  <TabItem value="azure">

The Azure environment does not require any additional keys. When the agent starts, it use `azure.identity.DefaultAzureCredential()` to load the default Azure credentials.

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # URI of the Azure Container Registry repository where the agent will store images.
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # If using Kaniko, specify the Azure Blob Storage container where the agent
  # will store the build context.
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

See the [`azure-identity` documentation](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) for more information on how to configure default Azure credentials.

  </TabItem>
</Tabs>

## Agent permissions

The agent permissions required vary by use case.

### Cloud registry permissions

Below are the permissions that are generally required by launch agents to interact with cloud registries.

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

```yaml
{
  'Version': '2012-10-17',
  'Statement':
    [
      {
        'Effect': 'Allow',
        'Action':
          [
            'ecr:CreateRepository',
            'ecr:UploadLayerPart',
            'ecr:PutImage',
            'ecr:CompleteLayerUpload',
            'ecr:InitiateLayerUpload',
            'ecr:DescribeRepositories',
            'ecr:DescribeImages',
            'ecr:BatchCheckLayerAvailability',
            'ecr:BatchDeleteImage',
          ],
        'Resource': 'arn:aws:ecr:<region>:<account-id>:repository/<repository>',
      },
      {
        'Effect': 'Allow',
        'Action': 'ecr:GetAuthorizationToken',
        'Resource': '*',
      },
    ],
}
```

  </TabItem>
  <TabItem value="gcp">

```js
artifactregistry.dockerimages.list;
artifactregistry.repositories.downloadArtifacts;
artifactregistry.repositories.list;
artifactregistry.repositories.uploadArtifacts;
```

  </TabItem>
  <TabItem value="azure">

Add the [`AcrPush` role](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush) if you use the Kaniko builder.

</TabItem>
</Tabs>

### Storage permissions for Kaniko

The launch agent requires permission to push to cloud storage if the agent uses the Kaniko builder. Kaniko uses a context store outside of the pod running the build job.

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

The recommended context store for the Kaniko builder on AWS is Amazon S3. The following policy can be used to give the agent access to an S3 bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListObjectsInBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>"]
    },
    {
      "Sid": "AllObjectActions",
      "Effect": "Allow",
      "Action": "s3:*Object",
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>/*"]
    }
  ]
}
```

  </TabItem>
  <TabItem value="gcp">

On GCP, the following IAM permissions are required for the agent to upload build contexts to GCS:

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

  </TabItem>
  <TabItem value="azure">

The [Storage Blob Data Contributor](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) role is required in order for the agent to upload build contexts to Azure Blob Storage.

  </TabItem>
</Tabs>


## Customizing the Kaniko build

Specify the Kubernetes Job spec that the Kaniko job uses in the `builder.kaniko-config` key of the agent configuration. For example:

```yaml title="launch-config.yaml"
builder:
  type: kaniko
  build-context-store: <my-build-context-store>
  destination: <my-image-destination>
  build-job-name: wandb-image-build
  kaniko-config:
    spec:
      template:
        spec:
          containers:
          - args:
            - "--cache=false" # Args must be in the format "key=value"
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## Deploy Launch agent into CoreWeave 
Optionally deploy the W&B Launch agent to CoreWeave Cloud infrastructure. CoreWeave is a cloud infrastructure that is purpose built for GPU-accelerated workloads.

For information on how to deploy the Launch agent to CoreWeave, see the [CoreWeave documentation](https://docs.coreweave.com/partners/weights-and-biases#integration). 

:::note
You will need to create a [CoreWeave account](https://cloud.coreweave.com/login) in order to deploy the Launch agent into a CoreWeave infrastructure. 
:::

