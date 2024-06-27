---
displayed_sidebar: default
---

# Secure private connectivity for Dedicated Cloud

You can connect to your [Dedicated Cloud](../hosting-options/dedicated_cloud.md) instance over the cloud provider's secure private network. This applies to the access from your desktop to the W&B UI application and from your AI training workloads to the W&B SDK API. If using this option, the relevant requests and responses do not transit through the public network / internet.

:::info
Secure private connectivity is available in preview as an advanced security option with Dedicated Cloud.
:::

Secure private connectivity is currently available on AWS instances of Dedicated Cloud using [AWS Privatelink](https://aws.amazon.com/privatelink/). If enabled, W&B creates a private endpoint service for your instance and provides you the relevant DNS URI to connect to. With that, you can create interface endpoints in your AWS accounts that route the relevant traffic to the private endpoint service. It is easier to setup for your AI training workloads running within your AWS VPCs. To use the same mechanism for traffic from user desktop to the W&B UI application, you must configure appropriate DNS based routing from your corporate network to the interface endpoints in your AWS accounts.

If you are interested in this feature for your instance on GCP or Azure, contact your W&B team.

You can use Secure private connectivity with [IP allowlisting](./ip-allowlisting.md). In such a case, W&B recommends using secure private connectivity for majority of the traffic from your AI workloads and user desktops, and IP allowlisting for instance administration from privileged locations.