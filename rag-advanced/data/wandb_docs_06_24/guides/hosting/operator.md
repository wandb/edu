---
description: Hosting W&B Server via Kubernetes Operator
displayed_sidebar: default
---

# W&B Kubernetes Operator

Use the W&B Kubernetes Operator to simplify deploying, administering, troubleshooting, and scaling your W&B Server deployments on Kubernetes. You can think of the operator as a smart assistant for your W&B instance.

The W&B Server architecture and design continuously evolve to expand the AI developer tooling capabilities for users, and to have appropriate primitives for high performance, better scalability, and easier administration. That evolution applies to the compute services, relevant storage and the connectivity between them. W&B plans to use the operator to roll out such improvements to users across deployment types.

:::info
W&B uses the operator to deploy and manage Dedicated Cloud instances on AWS, GCP and Azure public clouds.
:::

For more general information about the operator, see [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) in the Kubernetes documentation.


## Migrate self-managed instances to W&B Kubernetes Operator
W&B recommends that you use the operator if you self-manage your W&B Server instance. That would enable W&B to roll out the newer services and products to your instance more seamlessly, and provide better troubleshooting and support.

:::note
Operator for self-managed W&B Server deployments is in private preview. In future at or after GA time, W&B will deprecate deployment mechanisms that do not use the operator. Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions.
:::

## Pre-baked in Cloud Terraform modules

The W&B Kubernetes Operator is pre-baked with official W&B cloud-specific Terraform Modules with the following versions:

| Terraform Module                                 | Version |
| ------------------------------------------------ | ------- |
| https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

This integration ensures that W&B Kubernetes Operator is ready to use for your instance with minimal setup, providing a streamlined path to deploying and managing W&B Server in your cloud environment.

## Deploying with Helm Terraform module

Install the W&B Kubernetes Operator with the official W&B Terraform Module [terraform-helm-wandb](https://github.com/wandb/terraform-helm-wandb).

This method allows for customized deployments tailored to specific requirements, leveraging Terraform's infrastructure-as-code approach for consistency and repeatability.

:::note
For detailed instructions on how to use the operator, reach out to [Customer Support](mailto:support@wandb.com) or your W&B team.
:::
