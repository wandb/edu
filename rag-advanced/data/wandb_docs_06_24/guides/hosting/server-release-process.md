---
description: Release process for W&B Server
displayed_sidebar: default
---

# Server release process

## Frequency and deployment types
W&B Server releases apply to the **Dedicated Cloud** and **Self-managed** deployments. There are three kinds of server releases:

| Release type | Description |
|--------------|-------------|
| Monthly | Monthly releases include new features, enhancements, plus medium and low severity bug fixes. |
| Patch | Patch releases include critical and high severity bug fixes. Patches are only rarely released, as needed. |
| Feature | The feature release targets a specific release date for a new product feature, which occasionally happens before the standard monthly release. |

All releases are immediately deployed to all **Dedicated Cloud** instances once the acceptance testing phase is complete. It keeps those managed instances fully updated, making the latest features and fixes available to relevant customers. Customers with **Self-managed** instances are responsible for the [update process](./server-upgrade-process.md) on their own schedule, where they can use [the latest Docker image](https://hub.docker.com/r/wandb/local). Refer to [release support and end of life](#release-support-and-end-of-life).

:::info
Some advanced features are available only with the enterprise license. So even if you get the latest docker image but don't have an enterprise license, you would not be able to take advantage of the relevant advanced capabilities.
:::

:::note
Some new features start in private preview, which means they are only available to design partners or early adopters. You may not have such a feature until the W&B team enables it for your instance.
:::

## Release notes
The release notes for all releases are available at [W&B Server Releases on GitHub](https://github.com/wandb/server/releases). Customers who use Slack can receive automatic release announcements in their W&B Slack channel. Ask your W&B team to enable these updates.

## Release update and downtime
A server release does not generally require instance downtime for **Dedicated Cloud** instances and for customers with **Self-managed** deployments who have implemented a proper rolling update process.

Downtime might occur for the following scenarios:
* A new feature or enhancement requires changes to the underlying infrastructure such as compute, storage or network. W&B tries to send relevant advance notifications to **Dedicated Cloud** customers.
* An infrastructure change due to a security patch or to avoid `support end-of-life` for a particular version. For urgent changes, **Dedicated Cloud** customers may not receive advance notifications. The priority here is to keep the fleet secure and fully supported.

For both cases, updates roll out to all **Dedicated Cloud** instances without exception. Customers with **Self-managed** instances are responsible to manage such updates on their own schedule. Refer to [release support and end of life](#release-support-and-end-of-life).

## Release support and end of life policy
W&B supports every server release for six months from the release date. **Dedicated Cloud** instances are automatically updated. Customers with **Self-managed** instances are responsible to update their deployments in time to comply with the support policy. Avoid staying on a version older than six months as it would significantly limit support from W&B.

:::note
W&B strongly recommends customers with **Self-managed** instances to update their deployments with the latest release at least every quarter. This ensures that you are using the latest and greatest capabilities while also keeping well ahead of the release end of life.
:::