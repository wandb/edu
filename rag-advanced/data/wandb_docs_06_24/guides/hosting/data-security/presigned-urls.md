---
displayed_sidebar: default
---

# Pre-signed URLs

W&B uses pre-signed URLs to simplify access to blob storage from your AI workloads or user browsers. For basic information on pre-signed URLs, refer to [Pre-signed URLs for AWS S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html), [Signed URLs for Google Cloud Storage](https://cloud.google.com/storage/docs/access-control/signed-urls) and [Shared Access Signature for Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview).

When needed, AI workloads or user browser clients within your network request pre-signed URLs from the W&B Platform. W&B Platform then access the relevant blob storage to generate the pre-signed URL with required permissions, and returns it back to the client. The client then uses the pre-signed URL to access the blob storage for object upload or retrieval operations. URL expiry time for object downloads is 1 hour, and it is 24 hours for object uploads as some large objects may need more time to upload in chunks.

## Team-level access control

Each pre-signed URL is restricted to specific bucket(s) based on [team level access control](../iam/manage-users.md#manage-a-team) in the W&B platform. If a user is part of a team which is mapped to a blob storage bucket using [secure storage connector](./secure-storage-connector.md), and if that user is part of only that team, then the pre-signed URLs generated for their requests would not have permissions to access blob storage buckets mapped to other teams. 

:::info
W&B recommends adding users to only the teams that they are supposed to be a part of.
:::

## Network restriction

W&B recommends restricting the networks that can use pre-signed URLs to access the blob storage, by using IAM policy based restrictions on the buckets. 

In case of AWS, one can use [VPC or IP address based network restriction](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities). It ensures that your W&B specific buckets are accessed only from networks where your AI workloads are running, or from gateway IP addresses that map to your user machines if your users access artifacts using the W&B UI.

## Audit logs

W&B also recommends to use [W&B audit logs](../monitoring-usage/audit-logging.md) in addition to blob storage specific audit logs. For latter, refer to [AWS S3 access logs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html),[Google Cloud Storage audit logs](https://cloud.google.com/storage/docs/audit-logging) and [Monitor Azure blob storage](https://learn.microsoft.com/en-us/azure/storage/blobs/monitor-blob-storage). With audit logs, admin and security teams can keep track of which user is doing what in the W&B product, and take necessary action if they determine that some operations need to be limited for certain users.

:::note
Pre-signed URLs are the only supported blob storage access mechanism in W&B. W&B recommends configuring some or all of the above list of security controls depending on your risk appetite.
:::