---
displayed_sidebar: default
---

# IP allowlisting for Dedicated Cloud

You can restrict access to your [Dedicated Cloud](../hosting-options/dedicated_cloud.md) instance from only an authorized list of IP addresses. This applies to the access from your desktop to the W&B UI application and from your AI training workloads to the W&B SDK API. Once you configure IP allowlisting for your Dedicated Cloud instance, any requests from other unauthorized locations will be denied.

IP allowlisting is available on AWS and GCP Dedicated Cloud instances.

You can use IP allowlisting with [Secure private connectivity](./private-connectivity.md). In such a case, W&B recommends using secure private connectivity for majority of the traffic from your AI workloads and user desktops, and IP allowlisting for instance administration from privileged locations.

:::important
W&B strongly recommends to use [CIDR blocks](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) assigned to your corporate / business egress gateways rather than individual `/32` IP addresses. Using individual IP addresses is not scalable and has strict limits per cloud.
:::