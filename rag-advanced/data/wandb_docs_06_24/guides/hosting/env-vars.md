---
description: How to configure the W&B Server installation
displayed_sidebar: default
---

# Environment variables

In addition to configuring instance level settings via the System Settings admin UI, W&B also provides a way to configure these values via code using Environment Variables. Also, refer to [advanced configuration for IAM](./iam/advanced_env_vars.md).

## Configuration as code

| Environment Variable             | Description                                                                                                                                                                              |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | Your wandb/local license                                                                                                                                                                 |
| MYSQL                            | The MySQL connection string                                                                                                                                                              |
| BUCKET                           | The S3 / GCS bucket for storing data                                                                                                                                                     |
| BUCKET_QUEUE                     | The SQS / Google PubSub queue for object creation events                                                                                                                                 |
| NOTIFICATIONS_QUEUE              | The SQS queue on which to publish run events                                                                                                                                             |
| AWS_REGION                       | The AWS Region where your bucket lives                                                                                                                                                   |
| HOST                             | The FQD of your instance, i.e. [https://my.domain.net](https://my.domain.net)                                                                                                            |
| OIDC_ISSUER                      | A url to your Open ID Connect identity provider, i.e. [https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd](https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd) |
| OIDC_CLIENT_ID                   | The Client ID of application in your identity provider                                                                                                                                   |
| OIDC_AUTH_METHOD                 | Implicit (default) or pkce, see below for more context                                                                                                                                   |
| SLACK_CLIENT_ID                  | The client ID of the Slack application you want to use for alerts                                                                                                                        |
| SLACK_SECRET                     | The secret of the Slack application you want to use for alerts                                                                                                                           |
| LOCAL_RESTORE                    | You can temporarily set this to true if you're unable to access your instance. Check the logs from the container for temporary credentials.                                              |
| REDIS                            | Can be used to setup an external REDIS instance with W&B.                                                                                                                                |
| LOGGING_ENABLED                  | When set to true, access logs are streamed to stdout. You can also mount a sidecar container and tail `/var/log/gorilla.log` without setting this variable.                              |
| GORILLA_ALLOW_USER_TEAM_CREATION | When set to true, allows non-admin users to create a new team. False by default.                                                                                                         |
| GORILLA_DATA_RETENTION_PERIOD | How long to retain deleted data from runs in hours. Deleted run data is unrecoverable. Append an `h` to the input value. For example, `"24h"`. |


:::info

Use the GORILLA_DATA_RETENTION_PERIOD environment variable cautiously. Data is removed immediately once the environment variable is set. We also recommend that you backup both the database and the storage bucket before you enable this flag.

:::

## Advanced Reliability Settings

#### Redis

While configuring an external redis server is optional, it's highly recommended for production systems. Redis will improve the reliability of the service and enable caching which will decrease load times, especially in large projects. We recommend using a managed redis service (ex: ElastiCache) with high availability(HA) and the following specs:

- Minimum 4GB of memory, suggested 8GB
- Redis version 6.x
- In transit encryption
- Authentication enabled

#### Configuring REDIS in the W&B server

To configure the redis instance with W&B, you can navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/system-admin`. Enable the "Use an external Redis instance" option, and fill in the `redis` connection string in the following format:

![Configuring REDIS in W&B](/images/hosting/configure_redis.png)

You can also configure `redis` using the environment variable `REDIS` on the container or in your Kubernetes deployment. Alternatively, you could also setup `REDIS` as a Kubernetes secret.

The above assumes the `redis` instance is running at the default port of `6379`. If you configure a different port, setup authentication and also want to have TLS enabled on the `redis` instance the connection string format would look something like: `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`
