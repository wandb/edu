---
description: Guide for updating W&B (Weights & Biases) version and license across different installation methods.
displayed_sidebar: default
---

# Update W&B License and Version

Update your W&B Server Version and License with the same method you installed W&B Server with. The following table lists how to update your license and version based on different deployment methods:


| Release Type    | Description         |
| ---------------- | ------------------ |
| [Terraform](#update-with-terraform) | W&B supports three public Terraform modules for cloud deployment: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), and [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest). |
| [Helm](#update-with-helm)              | You can use the [Helm Chart](https://github.com/wandb/helm-charts) to install W&B into an existing Kubernetes cluster.  |
| [Docker](#update-with-docker-container)     | Docker latest docker image can found in the [W&B Docker Registry](https://hub.docker.com/r/wandb/local/tags).  |

## Update with Terraform

Update your license and version with Terraform. The proceeding table lists W&B managed Terraform modules based cloud platform.

|Cloud provider| Terraform module|
|-----|-----|
|AWS|[AWS Terraform module](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform module](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. First, navigate to the W&B maintained Terraform module for your appropriate cloud provider. See the preceding table to find the appropriate Terraform module based on your cloud provider.
2. Within your Terraform configuration, update `wandb_version` and `license` in your Terraform `wandb_app` module configuration:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # Your new license key
       wandb_version = "new_wandb_version" # Desired W&B version
       ...
   }
   ```
3. Apply the Terraform configuration with `terraform plan` and `terraform apply`.
   ```bash
   terraform init
   terraform apply
   ```

4. (Optional) If you use a `terraform.tfvars` or other `.tfvars` file:
   1. Update or create a `terraform.tfvars` file with the new W&B version and license key.
   2. Apply the configuration. In your Terraform workspace directory execute:  
   ```bash
   terraform plan -var-file="terraform.tfvars"
   terraform apply -var-file="terraform.tfvars"
   ```
## Update with Helm

### Update W&B with spec

1. Specify a new version by modifying the `image.tag` and/or `license` values in your Helm chart `*.yaml` configuration file:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. Execute the Helm upgrade with the following command:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### Update license and version directly

1. Set the new license key and image tag as environment variables:

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. Upgrade your Helm release with the command below, merging the new values with the existing configuration:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

For more details, see the [upgrade guide](https://github.com/wandb/helm-charts/blob/main/UPGRADE.md) in the public repository.

## Update with Docker container

1. Choose a new version from the [W&B Docker Registry](https://hub.docker.com/r/wandb/local/tags).
2. Pull the new Docker image version with:

   ```bash
   docker pull wandb/local:<new_version>
   ```

3. Update your Docker container to run the new image version, ensuring you follow best practices for container deployment and management.

## Update with admin UI

This method is only works for updating licenses that are not set with an environment variable in the W&B server container, typically in self-hosted Docker installations.

1. Obtain a new license from the [W&B Deployment Page](https://deploy.wandb.ai/), ensuring it matches the correct organization and deployment ID for the deployment you are looking to upgrade.
2. Access the W&B Admin UI at `<host-url>/system-settings`.
3. Navigate to the license management section.
4. Enter the new license key and save your changes.

