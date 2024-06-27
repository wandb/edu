---
displayed_sidebar: default
---

# Dedicated cloud (Single-tenant SaaS)

W&B Dedicated Cloud is a single-tenant, fully-managed platform deployed in W&B's AWS, GCP or Azure cloud accounts. Each Dedicated Cloud instance has its own isolated network, compute and storage from other W&B Dedicated Cloud instances. Your W&B specific metadata and data is stored in an isolated cloud storage and is processed using isolated cloud compute services. 

W&B Dedicated Cloud is available in [multiple global regions for each cloud provider](./dedicated_regions.md)

## Data security 
You can bring your own bucket (BYOB) using the [secure storage connector](../data-security/secure-storage-connector.md) at the [instance and team levels](../data-security/secure-storage-connector.md#configuration-options) to store your files such as models, datasets, and more.

Similar to W&B Multi-tenant Cloud, you can configure a single bucket for multiple teams or you can use separate buckets for different teams. If you do not configure secure storage connector for a team, that data is stored in the instance level bucket.

![](/images/hosting/dedicated_cloud_arch.png)

In addition to BYOB with secure storage connector, you can utilize [IP allowlisting](../data-security/ip-allowlisting.md) to restrict access to your Dedicated Cloud instance from specific network locations. 

You can also connect to your Dedicated Cloud instance using [cloud provider's secure connectivity solution](../data-security/private-connectivity.md). This feature is currently available for AWS instances of Dedicated Cloud with [AWS PrivateLink](https://aws.amazon.com/privatelink/).

## Identity and access management (IAM)
Use the identity and access management capabilities for secure authentication and effective authorization in your W&B Organization. The following features are available for IAM in Dedicated Cloud instances:

* Authenticate with [SSO using OpenID Connect (OIDC)](../iam/sso.md) or with [LDAP](../iam/ldap.md).
* [Configure appropriate user roles](../iam/manage-users.md) at the scope of the organization and within a team.
* Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it with [restricted projects](../iam/restricted-projects.md).

## Monitor
Use [Audit logs](../monitoring-usage/audit-logging.md) to track user activity within your teams and to conform to your enterprise governance requirements. Also, you can view organization usage in our Dedicated Cloud instance with [W&B Organization Dashboard](../monitoring-usage/org_dashboard.md).

## Maintenance
Similar to W&B Multi-tenant Cloud, you do not incur the overhead and costs of provisioning and maintaining the W&B platform with Dedicated Cloud.

To understand how W&B manages updates on Dedicated Cloud, refer to the [server release process](../server-release-process.md).

## Compliance 
Security controls for W&B Dedicated Cloud are periodically audited internally and externally. Refer to the [W&B Security Portal](https://security.wandb.ai/) to request the SOC2 report and other security and compliance documents.

## Migration options
Migration to Dedicated Cloud from a [Self-managed instance](./self-managed.md) or [Multi-tenant Cloud](./saas_cloud.md) is supported.

## Next steps
Submit [this form](https://wandb.ai/site/for-enterprise/dedicated-saas-trial) if you are interested in using Dedicated Cloud.

