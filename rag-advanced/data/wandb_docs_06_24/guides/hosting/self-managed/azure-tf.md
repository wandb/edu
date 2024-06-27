---
title: Azure
description: Hosting W&B Server on Azure.
displayed_sidebar: default
---

:::info
W&B recommends fully managed deployment options such as [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) or [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) deployment types. W&B fully managed services are simple and secure to use, with minimum to no configuration required.
:::


If you've determined to self-managed W&B Server, W&B recommends using the [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) to deploy the platform on Azure.

The module documentation is extensive and contains all available options that can be used. We will cover some deployment options in this document.

Before you start, we recommend you choose one of the [remote backends](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) available for Terraform to store the [State File](https://developer.hashicorp.com/terraform/language/state).

The State File is the necessary resource to roll out upgrades or make changes in your deployment without recreating all components.

The Terraform Module will deploy the following `mandatory` components:

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Fliexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

Other deployment options can also include the following optional components:

- Azure Cache for Redis
- Azure Event Grid

## **Pre-requisite permissions**

The simplest way to get the AzureRM provider configured is via [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) but the incase of automation using [Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) can also be useful.
Regardless the authentication method used, the account that will run the Terraform needs to be able to create all components described in the Introduction.

## General steps
The steps on this topic are common for any deployment option covered by this documentation.

1. Prepare the development environment.
  * Install [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
  * We recommend creating a Git repository with the code that will be used, but you can keep your files locally.

2. **Create the `terraform.tfvars` file** The `tvfars` file content can be customized according to the installation type, but the minimum recommended will look like the example below.

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   The variables defined here need to be decided before the deployment because. The `namespace` variable will be a string that will prefix all resources created by Terraform.

   The combination of `subdomain` and `domain` will form the FQDN that W&B will be configured. In the example above, the W&B FQDN will be `wandb-aws.wandb.ml` and the DNS `zone_id` where the FQDN record will be created.

3. **Create the file `versions.tf`** This file will contain the Terraform and Terraform provider versions required to deploy W&B in AWS
  ```bash
  terraform {
    required_version = "~> 1.3"

    required_providers {
      azurerm = {
        source  = "hashicorp/azurerm"
        version = "~> 3.17"
      }
    }
  }
  ```

  Refer to the [Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) to configure the AWS provider.

  Optionally, **but highly recommended**, you can add the [remote backend configuration](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) mentioned at the beginning of this documentation.

4. **Create the file** `variables.tf`. For every option configured in the `terraform.tfvars` Terraform requires a correspondent variable declaration.

   ```bash
    variable "namespace" {
      type        = string
      description = "String used for prefix resources."
    }

    variable "location" {
      type        = string
      description = "Azure Resource Group location"
    }

    variable "domain_name" {
      type        = string
      description = "Domain for accessing the Weights & Biases UI."
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Subdomain for accessing the Weights & Biases UI. Default creates record at Route53 Route."
    }

    variable "license" {
      type        = string
      description = "Your wandb/local license"
    }
  ```

## Deployment - Recommended (~20 mins)

This is the most straightforward deployment option configuration that will create all `Mandatory` components and install in the `Kubernetes Cluster` the latest version of `W&B`.

1. **Create the `main.tf`** In the same directory where you created the files in the `General Steps`, create a file `main.tf` with the following content:

  ```bash
  provider "azurerm" {
    features {}
  }

  provider "kubernetes" {
    host                   = module.wandb.cluster_host
    cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
    client_key             = base64decode(module.wandb.cluster_client_key)
    client_certificate     = base64decode(module.wandb.cluster_client_certificate)
  }

  provider "helm" {
    kubernetes {
      host                   = module.wandb.cluster_host
      cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
      client_key             = base64decode(module.wandb.cluster_client_key)
      client_certificate     = base64decode(module.wandb.cluster_client_certificate)
    }
  }

  # Spin up all required services
  module "wandb" {
    source  = "wandb/wandb/azurerm"
    version = "~> 1.2"

    namespace   = var.namespace
    location    = var.location
    license     = var.license
    domain_name = var.domain_name
    subdomain   = var.subdomain

    deletion_protection = false

    tags = {
      "Example" : "PublicDns"
    }
  }

  output "address" {
    value = module.wandb.address
  }

  output "url" {
    value = module.wandb.url
  }
  ```

2. **Deploy to W&B** To deploy W&B, execute the following commands:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## Deployment with REDIS Cache

Another deployment option uses `Redis` to cache the SQL queries and speed up the application response when loading the metrics for the experiments.

You need to add the option `create_redis = true` to the same `main.tf` file we worked on in [Deployment Recommended](azure-tf.md#deployment---recommended-20-mins) to enable the cache.

```bash
# Spin up all required services
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"


  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  create_redis       = true # Create Redis
  [...]
```

## Deployment with External Queue

Deployment option 3 consists of enabling the external `message broker`. This is optional because the W&B brings embedded a broker. This option doesn't bring a performance improvement.

The Azure resource that provides the message broker is the `Azure Event Grid`, and to enable it, you will need to add the option `use_internal_queue = false` to the same `main.tf` that we worked on the [Deployment Recommended](azure-tf.md#deployment---recommended-20-mins)
```bash
# Spin up all required services
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"


  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  use_internal_queue       = false # Enable Azure Event Grid
  [...]
}
```

## Other deployment options

You can combine all three deployment options adding all configurations to the same file.
The [Terraform Module](https://github.com/wandb/terraform-azure-wandb) provides several options that can be combined along with the standard options and the minimal configuration found in [Deployment Recommended](azure-tf.md#deployment---recommended-20-mins)
