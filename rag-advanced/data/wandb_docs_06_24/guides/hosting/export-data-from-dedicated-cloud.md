---
description: Export data from Dedicated Cloud
displayed_sidebar: default
---

# Export data from Dedicated Cloud

If you would like to export all the data managed in your Dedicated Cloud instance, you may use the W&B SDK API to extract the runs, metrics, artifacts etc. and log those to another cloud or on-premises storage using API relevant to that storage. 

Refer to data export use cases at [Import and Export Data](../track/public-api-guide#export-data). Another use case would be if you are planning to end your agreement to use Dedicated Cloud, you may want to export the pertinent data before W&B terminates the instance.

Refer to the table below for data export API and pointers to relevant documentation:

| Purpose | Documentation |
|---------|---------------|
| Export project metadata | [Projects API](../../ref/python/public-api/api#projects) |
| Export runs in a project | [Runs API](../../ref/python/public-api/api#runs), [Export run data](../track/public-api-guide#export-run-data), [Querying multiple runs](../track/public-api-guide#querying-multiple-runs) |
| Export reports | [Reports API](../../ref/python/public-api/api#reports) |
| Export artifacts | [Artifact API](../../ref/python/public-api/api#artifact), [Explore and traverse an artifact graph](../artifacts/explore-and-traverse-an-artifact-graph#traverse-an-artifact-programmatically), [Download and use an artifact](../artifacts/download-and-use-an-artifact#download-and-use-an-artifact-stored-on-wb) |

:::info
You manage artifacts stored in the Dedicated Cloud with [Secure Storage Connector](./data-security/secure-storage-connector). In that case, you may not need to export the artifacts using the W&B SDK API.
:::

:::note
Using W&B SDK API to export all of your data can be slow if you have a large number of runs, artifacts etc. W&B recommends running the export process in appropriately sized batches so as not to overwhelm your Dedicated Cloud instance.
:::