---
title: Install on on-prem infra
description: Hosting W&B Server on on-premises infrastructure
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info
W&B recommends fully managed deployment options such as [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) or [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) deployment types. W&B fully managed services are simple and secure to use, with minimum to no configuration required.
:::


You can run W&B Server on your on-premises infrastructure if Multi-tenant Cloud or Dedicated Cloud are not a good fit for your organization.

Reach out to the W&B Sales Team for related question: [contact@wandb.com](mailto:contact@wandb.com).

## Infrastructure Guidelines

The following infrastructure guidelines section outline W&B recommendations to take into consideration when you set up your application server, database server, and object storage.

:::tip
W&B strongly recommends to deploy W&B Server into a Kubernetes cluster using the W&B Kubernetes Operator. Deploying to a Kubernetes cluster with the operator ensures that you can use all the existing and latest W&B features.
:::

:::caution
W&B application performance depends on scalable data stores that your operations team must configure and manage. The team must provide a MySQL 8 database cluster and an AWS S3 compatible object store for the application to scale properly.
:::

### Application Server

We recommend deploying W&B Server into its own namespace and a two availability zone node group with the following specifications to provide the best performance, reliability, and availability:

| Specification              | Value                             |
|----------------------------|-----------------------------------|
| Bandwidth                  | Dual 10 Gigabit+ Ethernet Network |
| Root Disk Bandwidth (Mbps) | 4,750+                            |
| Root Disk Provision (GB)   | 100+                              |
| Core Count                 | 4                                 |
| Memory (GiB)               | 8                                 |

This ensures that W&B Server has sufficient disk space to process the application data and store temporary logs before they are externalized. It also ensures fast and reliable data transfer, the necessary processing power and memory for smooth operation, and that W&B will not be affected by any noisy neighbors. 

It is important to keep in mind that these specifications are minimum requirements, and actual resource needs may vary depending on the specific usage and workload of the W&B application. Monitoring the resource usage and performance of the application is critical to ensure that it operates optimally and to make adjustments as necessary.

### Database Server

W&B recommends a [MySQL 8](../self-managed/bare-metal.md#mysql-80) database as a metadata store. The shape of the ML practitioners parameters and metadata will greatly affect the performance of the database. The database is typically incrementally written to as practitioners track their training runs and is more read heavy when queries are executed in reports and dashboard.

To ensure optimal performance we recommend deploying the W&B database on to a server with the following starting specs:

| Specification              | Value                             |
|--------------------------- |-----------------------------------|
| Bandwidth                  | Dual 10 Gigabit+ Ethernet Network |
| Root Disk Bandwidth (Mbps) | 4,750+                            |
| Root Disk Provision (GB)   | 1000+                              |
| Core Count                 | 4                                 |
| Memory (GiB)               | 32                                |

Again, we recommend monitoring the resource usage and performance of the database to ensure that it operates optimally and to make adjustments as necessary.

Additionally, we recommend the following [parameter overrides](../self-managed/bare-metal.md#mysql-80) to tune the DB for MySQL 8.

### Object Storage

W&B is compatible with an object storage that supports S3 API interface, Signed URLs and CORS. We recommend specing the storage array to the current needs of your practitioners and to capacity plan on a regular cadence.

More details on object store configuration can be found in the [how-to section](../self-managed/bare-metal.md#object-store).

Some tested and working providers:
- [MinIO](https://min.io/)
- [Ceph](https://ceph.io/)
- [NetApp](https://www.netapp.com/)
- [Pure Storage](https://www.purestorage.com/)

##### Secure Storage Connector

The [Secure Storage Connector](../data-security/secure-storage-connector.md) is not available for teams at this time for bare metal deployments.

## MySQL Database

:::caution
W&B does not recommend using MySQL 5.7. If you are using MySQL 5.7, migrate to MySQL 8 for best compatibility with latest versions of W&B Server. The W&B Server currently only supports `MySQL 8` versions `8.0.28` and above.
:::

There are a number of enterprise services that make operating a scalable MySQL database simpler. We suggest looking into one of the following solutions:

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

Satisfy the conditions below if you run W&B Server MySQL 8.0 or when you upgrade from MySQL 5.7 to 8.0:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

Due to some changes in the way that MySQL 8.0 handles `sort_buffer_size`, you might need to update the `sort_buffer_size` parameter from its default value of `262144`. Our recommendation is to set the value to `33554432(32MiB)` in order for the database to efficiently work with the W&B application. Note that, this only works with MySQL versions 8.0.28 and above.

### Database considerations

Consider the following when you run your own MySQL database:

1. **Backups**. You should  periodically back up the database to a separate facility. We suggest daily backups with at least 1 week of retention.
2. **Performance.** The disk the server is running on should be fast. We suggest running the database on an SSD or accelerated NAS.
3. **Monitoring.** The database should be monitored for load. If CPU usage is sustained at > 40% of the system for more than 5 minutes it is likely a good indication the server is resource starved.
4. **Availability.** Depending on your availability and durability requirements you may want to configure a hot standby on a separate machine that streams all updates in realtime from the primary server and can be used to failover to incase the primary server crashes or become corrupted.

Create a database and a user with the following SQL query. Replace `SOME_PASSWORD` with password of your choice:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

#### Parameter Group Configuration

Ensure that the following parameter groups are set to tune the database performance:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 33554432
```

## Object Store
The object store can be externally hosted on a [Minio cluster](https://docs.min.io/minio/k8s/), or any Amazon S3 compatible object store that has support for signed urls. Run the [following script](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) to check if your object store supports signed urls.

Additionally, the following CORS policy needs to be applied to the object store.

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<CORSRule>
    <AllowedOrigin>http://YOUR-W&B-SERVER-IP</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedMethod>HEAD</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
</CORSRule>
</CORSConfiguration>
```

You can specify your credentials in a connection string when you connect to an Amazon S3 compatible object store. For example, you can specify the following: 

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

You can optionally tell W&B to only connect over TLS if you configure a trusted SSL certificate for your object store. To do so, add the `tls` query parameter to the url. For example, the following URL example demonstrates how to add the TLS query parameter to an Amazon S3 URI:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

:::caution
This will only work if the SSL certificate is trusted. W&B does not support self-signed certificates.
:::

Set `BUCKET_QUEUE` to `internal://` if you use third-party object stores.  This tells the W&B server to manage all object notifications internally instead of depending on an external SQS queue or equivalent.

The most important things to consider when running your own object store are:

1. **Storage capacity and performance**. It's fine to use magnetic disks, but you should be monitoring the capacity of these disks. Average W&B usage results in 10's to 100's of Gigabytes. Heavy usage could result in Petabytes of storage consumption.
2. **Fault tolerance.** At a minimum, the physical disk storing the objects should be on a RAID array. If you use minio, consider running it in [distributed mode](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed).
3. **Availability.** Monitoring should be configured to ensure the storage is available.

There are many enterprise alternatives to running your own object storage service such as:

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO setup

If you use minio, you can run the following commands to create a bucket.

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## Kubernetes deployment

The following k8s yaml can be customized but should serve as a basic foundation for configuring local in Kubernetes.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wandb
  labels:
    app: wandb
spec:
  strategy:
    type: RollingUpdate
  replicas: 1
  selector:
    matchLabels:
      app: wandb
  template:
    metadata:
      labels:
        app: wandb
    spec:
      containers:
        - name: wandb
          env:
            - name: HOST
              value: https://YOUR_DNS_NAME
            - name: LICENSE
              value: XXXXXXXXXXXXXXX
            - name: BUCKET
              value: s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
            - name: BUCKET_QUEUE
              value: internal://
            - name: AWS_REGION
              value: us-east-1
            - name: MYSQL
              value: mysql://$USERNAME:$PASSWORD@$HOSTNAME/$DATABASE
          imagePullPolicy: IfNotPresent
          image: wandb/local:latest
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
          readinessProbe:
            httpGet:
              path: /ready
              port: http
          startupProbe:
            httpGet:
              path: /ready
              port: http
            failureThreshold: 60 # allow 10 minutes for migrations
          resources:
            requests:
              cpu: '2000m'
              memory: 4G
            limits:
              cpu: '4000m'
              memory: 8G
---
apiVersion: v1
kind: Service
metadata:
  name: wandb-service
spec:
  type: NodePort
  selector:
    app: wandb
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: wandb-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  defaultBackend:
    service:
      name: wandb-service
      port:
        number: 80
```

The k8s YAML above should work in most on-premises installations. However the details of your Ingress and optional SSL termination will vary. See [networking](#networking) below.

### Helm Chart

W&B also supports deploying via a Helm Chart. The official W&B helm chart can be [found here](https://github.com/wandb/helm-charts).

### Openshift

W&B supports operating from within an [Openshift kubernetes cluster](https://www.redhat.com/en/technologies/cloud-computing/openshift). Simply follow the instructions in the kubernetes deployment section above.

#### Run the container as an un-privileged user

By default, containers use a `$UID` of 999. Specify `$UID` >= 100000 and a `$GID` of 0 if your orchestrator requires the container run with a non-root user.

:::note
W&B  must start as the root group (`$GID=0`) for file system permissions to function properly.
:::

An example security context for Kubernetes looks similar to the following:

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## Networking

### Load Balancer

Run a load balancer that terminates network requests at the appropriate network boundary. 

Common load balancers include:
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

Ensure that all machines used to execute machine learning payloads, and the devices used to access the service through web browsers, can communicate to this endpoint. 


### SSL / TLS

W&B Server does not terminate SSL. If your security policies require SSL communication within your trusted networks consider using a tool like Istio and [side car containers](https://istio.io/latest/docs/reference/config/networking/sidecar/). The load balancer itself should terminate SSL with a valid certificate. Using self-signed certificates is not supported and will cause a number of challenges for users. If possible using a service like [Let's Encrypt](https://letsencrypt.org) is a great way to provided trusted certificates to your load balancer. Services like Caddy and Cloudflare manage SSL for you.

### Example Nginx Configuration

The following is an example configuration using nginx as a reverse proxy.

```nginx
events {}
http {
    # If we receive X-Forwarded-Proto, pass it through; otherwise, pass along the
    # scheme used to connect to this server
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # Also, in the above case, force HTTPS
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # If we receive X-Forwarded-Host, pass it though; otherwise, pass along $http_host
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # If we receive X-Forwarded-Port, pass it through; otherwise, pass along the
    # server port the client connected to
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # If we receive Upgrade, set Connection to "upgrade"; otherwise, delete any
    # Connection header that may have been passed to this server
    map $http_upgrade $proxy_connection {
        default upgrade;
        '' close;
    }

    server {
        listen 443 ssl;
        server_name         www.example.com;
        ssl_certificate     www.example.com.crt;
        ssl_certificate_key www.example.com.key;

        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $proxy_connection;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $proxy_x_forwarded_proto;
        proxy_set_header X-Forwarded-Host $proxy_x_forwarded_host;

        location / {
            proxy_pass  http://$YOUR_UPSTREAM_SERVER_IP:8080/;
        }

        keepalive_timeout 10;
    }
}
```

## Verify your installation

Very your W&B Server is configured properly. Run the following commands in your terminal:

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

Check log files to view any errors the W&B Server hits at startup. Run the following commands based on whether if you use Docker or Kubernetes: 

<Tabs
  defaultValue="docker"
  values={[
    {label: 'Docker', value: 'docker'},
    {label: 'Kubernetes', value: 'kubernetes'},
  ]}>
  <TabItem value="docker">

```bash
docker logs wandb-local
```

  </TabItem>
  <TabItem value="kubernetes">

```bash
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

  </TabItem>
</Tabs>


Contact W&B Support if you encounter errors. 