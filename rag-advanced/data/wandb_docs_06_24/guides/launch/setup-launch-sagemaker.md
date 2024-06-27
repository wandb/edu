---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up for SageMaker

<!-- You can use W&B Launch to submit jobs to run as SageMaker training jobs. Amazon SageMaker training jobs allow users to train machine learning models using provided or custom algorithms on the SageMaker platform. Once initiated, SageMaker handles the underlying infrastructure, scaling, and orchestration. -->

You can use W&B Launch to submit launch jobs to Amazon SageMaker to train machine learning models using provided or custom algorithms on the SageMaker platform. SageMaker takes care of spinning up and releasing compute resources, so it can be a good choice for teams without an EKS cluster.

Launch jobs sent to a W&B Launch queue connected to Amazon SageMaker are executed as SageMaker Training Jobs with the [CreateTrainingJob API](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_CreateTrainingJob.html). Use the launch queue configuration to control arguments sent to the `CreateTrainingJob` API.

Amazon SageMaker [uses Docker images to execute training jobs](https://docs.aws.amazon.com/SageMaker/latest/dg/your-algorithms-training-algo-dockerfile.html). Images pulled by SageMaker must be stored in the Amazon Elastic Container Registry (ECR). This means that the image you use for training must be stored on ECR. 

:::note
This guide shows how to execute SageMaker Training Jobs. For information on how to deploy to models for inference on Amazon SageMaker, see [this example Launch job](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints).
:::


## Prerequisites

Before you get started, ensure you satisfy the following prerequisites:

* [Decide if you want the Launch agent to build a Docker image for you.](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)
* [Set up AWS resources and gather information about S3, ECR, and Sagemaker IAM roles.](#set-up-aws-resources)
* [Create an IAM role for the Launch agent](#create-an-iam-role-for-launch-agent).

### Decide if you want the Launch agent to build a Docker images

Decide if you want the W&B Launch agent to build a Docker image for you. There are two options you can choose from: 

* Permit the launch agent build a Docker image, push the image to Amazon ECR, and submit [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) jobs for you. This option can offer some simplicity to ML Engineers rapidly iterating over training code.  
* The launch agent uses an existing Docker image that contains your training or inference scripts. This option works well with existing CI systems. If you choose this option, you will need to manually upload your Docker image to your container registry on Amazon ECR.


### Set up AWS resources

Ensure you have the following AWS resources configured in your preferred AWS region:

1. An [ECR repository](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) to store container images.
2. One or more [S3 buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) to store inputs and outputs for your SageMaker Training jobs.
3. An IAM role for Amazon SageMaker that permits SageMaker to run training jobs and interact with Amazon ECR and Amazon S3.

Make a note of the ARNs for these resources. You will need the ARNs when you define the [Launch queue configuration](#configure-a-queue-for-sagemaker). 

<!-- If you don't have these resources, create them in AWS or follow our walkthrough tutorial [[link]]. -->

### Create an IAM role for Launch agent

The Launch agent needs permission to create Amazon SageMaker training jobs. Follow the procedure below to create an IAM role:

1. From the IAM screen in AWS, create a new role. 
2. For **Trusted Entity**, select **AWS Account** (or another option that suits your organization's policies).
3. Scroll through the permissions screen and click **Next**. 
4. Give the role a name and description.
5. Select **Create role**.
6. Under **Add permissions**, select **Create inline policy**.
7. Toggle to the JSON policy editor, then paste the following policy based on your use case. Substitute values enclosed with `<>` with your own values:

<Tabs
  defaultValue="build"
  values={[
    {label: 'Agent builds and submits Docker image', value: 'build'},
    {label: 'Agent submits pre-built Docker image', value: 'no-build'},
  ]}>
  <TabItem value="no-build">

  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "logs:DescribeLogStreams",
          "SageMaker:AddTags",
          "SageMaker:CreateTrainingJob",
          "SageMaker:DescribeTrainingJob"
        ],
        "Resource": "arn:aws:SageMaker:<region>:<account-id>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/<RoleArn-from-queue-config>"
      },
    {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<ARN-OF-KMS-KEY>",
        "Condition": {
          "StringEquals": {
            "kms:ViaService": "SageMaker.<region>.amazonaws.com",
            "kms:GrantIsForAWSResource": "true"
          }
        }
      }
    ]
  }
  ```
  </TabItem>
  <TabItem value="build">

  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "logs:DescribeLogStreams",
          "SageMaker:AddTags",
          "SageMaker:CreateTrainingJob",
          "SageMaker:DescribeTrainingJob"
        ],
        "Resource": "arn:aws:SageMaker:<region>:<account-id>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/<RoleArn-from-queue-config>"
      },
       {
      "Effect": "Allow",
      "Action": [
        "ecr:CreateRepository",
        "ecr:UploadLayerPart",
        "ecr:PutImage",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:DescribeImages",
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchDeleteImage"
      ],
      "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<repository>"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<ARN-OF-KMS-KEY>",
        "Condition": {
          "StringEquals": {
            "kms:ViaService": "SageMaker.<region>.amazonaws.com",
            "kms:GrantIsForAWSResource": "true"
          }
        }
      }
    ]
  }
  ```
  </TabItem>
</Tabs>

8. Click **Next**.
9. Note the ARN for the role. You will specify the ARN when you set up the launch agent.

For more information on how to create IAM role, see the [AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html).

:::info
* If you want the launch agent to build images, see the [Advanced agent set up](./setup-agent-advanced.md) for additional permissions required.
* The `kms:CreateGrant` permission for SageMaker queues is required only if the associated ResourceConfig has a specified VolumeKmsKeyId and the associated role does not have a policy that permits this action.
:::



## Configure launch queue for SageMaker

Next, create a queue in the W&B App that uses SageMaker as its compute resource:

1. Navigate to the [Launch App](https://wandb.ai/launch).
3. Click on the **Create Queue** button.
4. Select the **Entity** you would like to create the queue in.
5. Provide a name for your queue in the **Name** field.
6. Select **SageMaker** as the **Resource**.
7. Within the **Configuration** field, provide information about your SageMaker job. By default, W&B will populate a YAML and JSON `CreateTrainingJob` request body:
```json
{
  "RoleArn": "<REQUIRED>", 
  "ResourceConfig": {
      "InstanceType": "ml.m4.xlarge",
      "InstanceCount": 1,
      "VolumeSizeInGB": 2
  },
  "OutputDataConfig": {
      "S3OutputPath": "<REQUIRED>"
  },
  "StoppingCondition": {
      "MaxRuntimeInSeconds": 3600
  }
}
```
You must at minimum specify:

- `RoleArn` : ARN of the SageMaker execution IAM role (see [prerequisites](#prerequisites)). Not to be confused with the launch **agent** IAM role.
- `OutputDataConfig.S3OutputPath` : An Amazon S3 URI specifying where SageMaker outputs will be stored.
- `ResourceConfig`: Required specification of a resource config. Options for resource config are outlined [here](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_ResourceConfig.html).
- `StoppingCondition`: Required specification of the stopping conditions for the training job. Options outlined [here](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_StoppingCondition.html).
7. Click on the **Create Queue** button.


## Set up the launch agent

The following section describes where you can deploy your agent and how to configure your agent based on where it is deployed.

There are [several options for how the Launch agent is deployed for a Amazon SageMaker](#decide-where-to-run-the-launch-agent) queue: on a local machine, on an EC2 instance, or in an EKS cluster. [Configure your launch agent appropriately](#configure-a-launch-agent) based on the where you deploy your agent.


### Decide where to run the Launch agent

For production workloads and for customers who already have an EKS cluster, W&B recommends deploying the Launch agent to the EKS cluster using this Helm chart.

For production workloads without an current EKS cluster, an EC2 instance is a good option. Though the launch agent instance will keep running all the time, the agent doesn't need more than a `t2.micro` sized EC2 instance which is relatively affordable.

For experimental or solo use cases, running the Launch agent on your local machine can be a fast way to get started.

Based on your use case, follow the instructions provided in the following tabs to properly configure up your launch agent: 
<Tabs
  defaultValue="eks"
  values={[
    {label: 'EKS', value: 'eks'},
    {label: 'EC2', value: 'ec2'},
    {label: 'Local machine', value: 'local'},
  ]}>
  <TabItem value="eks">

W&B strongly encourages that you use the[ W&B managed helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) to install the agent in an EKS cluster.

</TabItem>
  <TabItem value="ec2">

Navigate to the Amazon EC2 Dashboard and complete the following steps:

1. Click **Launch instance**.
2. Provide a name for the **Name** field. Optionally add a tag.
2. From the **Instance type**, select an instance type for your EC2 container. You do not need more than 1vCPU and 1GiB of memory (for example a t2.micro). 
3. Create a key pair for your organization within the **Key pair (login)** field. You will use this key pair to [connect to your EC2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html) with SSH client at a later step.
2. Within **Network settings**, select an appropriate security group for your organization. 
3. Expand **Advanced details**. For **IAM instance profile**, select the launch agent IAM role you created above.
2. Review the **Summary** field. If correct, select **Launch instance**. 

Navigate to **Instances** within the left panel of the EC2 Dashboard on AWS. Ensure that the EC2 instance you created is running (see the **Instance state** column). Once you confirm your EC2 instance is running, navigate to your local machine's terminal and complete the following:

1. Select **Connect**. 
2. Select the **SSH client** tab and following the instructions outlined to connect to your EC2 instance.
3. Within your EC2 instance, install the following packages:
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. Next, install and start Docker within your EC2 instance:
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

Now you can proceed to setting up the Launch agent config.

  </TabItem>
  <TabItem value="local">

Use the AWS config files located at `~/.aws/config`  and `~/.aws/credentials` to associate a role with an agent that is polling on a local machine. Provide the IAM role ARN that you created for the launch agent in the previous step.
 
```yaml title="~/.aws/config"
[profile SageMaker-agent]
role_arn = arn:aws:iam::<account-id>:role/<agent-role-name>
source_profile = default                                                                   
```

```yaml title="~/.aws/credentials"
[default]
aws_access_key_id=<access-key-id>
aws_secret_access_key=<secret-access-key>
aws_session_token=<session-token>
```

Note that session tokens have a [max length](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description) of 1 hour or 3 days depending on the principal they are associated with.

  </TabItem>
</Tabs>


### Configure a launch agent
Configure the launch agent with a YAML config file named `launch-config.yaml`. 

By default, W&B will check for the config file in `~/.config/wandb/launch-config.yaml`. You can optionally specify a different directory when you activate the launch agent with the `-c` flag.

The following YAML snippet demonstrates how to specify the core config agent options:

```yaml title="launch-config.yaml"
max_jobs: -1
queues:
  - <queue-name>
environment:
  type: aws
  region: <your-region>
registry:
  type: ecr
  uri: <ecr-repo-arn>
builder: 
  type: docker

```

Now start the agent with `wandb launch-agent`


 ## (Optional) Push your launch job Docker image to Amazon ECR

:::info
This section applies only if your launch agent uses existing Docker images that contain your training or inference logic. [There are two options on how your launch agent behaves.](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)  
:::

Upload your Docker image that contains your launch job to your Amazon ECR repo. Your Docker image needs to be in your ECR registry before you submit new launch jobs if you are using image-based jobs.


<!--  
The full URI to the image can then be used in job creation e.g.

```bash
wandb job create image <your-image-uri> --p <project> ...
``` -->



<!--
## Launch jobs from W&B

If you go to the W&B GUI, your SageMaker Launch queue will now be active.  You can push jobs to it from the UI or CLI.

 -->
