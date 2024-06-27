---
description: Run Weights and Biases on your own machines using Docker
displayed_sidebar: default
---

# Getting started

Follow this "Hello, world!" example to learn the general workflow to install W&B Server for Dedicated Cloud and Self Managed hosting options. By the end of this demo, you will know how to host W&B Server on your local machine using a Trial Mode W&B license. 

For demonstration purposes, this demo uses a local development server on port `8080` (`localhost:8080`).

:::tip
**Trial Mode vs. Production Setup**

In Trial Mode, you run the Docker container on a single machine. This setup is ideal for testing the product, but it is not scalable.

For production work, set up a scalable file system to avoid data loss. W&B strongly recommends that you:
* Allocate extra space in advance, 
* Resize the file system proactively as you log more data
* Configure external metadata and object stores for backup.
:::

## Prerequisites
Before you get started, ensure your local machine satisfies the following requirements: 

1. Install [Python](https://www.python.org)
2. Install [Docker](https://www.docker.com) and ensure it is running
3. Install or upgrade the latest version of W&B:
   ```bash
   pip install --upgrade wandb
   ```
##  1. Pull the W&B Docker image

Run the following in your terminal:

```bash
wandb server start
```

This command pulls the latest W&B Docker image [`wandb/local`](https://hub.docker.com/r/wandb/local).


## 2. Create a W&B account
Navigate to `http://localhost:8080/signup` and create an initial user account. Provide a name, email address, a username, and a password: 

![](/images/hosting/signup_localhost.png)

Click the **Sign Up** button to create a W&B account. 

:::note
For this demo, create a new W&B account even if you already have a W&B account. 
:::


### Copy your API key
After you create an account, navigate to `http://localhost:8080/authorize`.  

Copy the W&B API key that appears on the screen. At a later step, you will need this key at a later step to verify your login credentials.

![](/images/hosting/copy_api_key.png)

## 3. Generate a license
Navigate to the W&B Deploy Manager at https://deploy.wandb.ai/deploy to generate a Trial Mode W&B license.

1. Select Docker as your provider
![](/images/hosting/deploy_manager_platform.png)
2. Click **Next**.
3. Select a license owner from the **Owner of license** dropdown.
![](/images/hosting/deploy_manager_info.png)
4. Click **Next**.
5. Provide a name for your license in the **Name of Instance** field.
6. (Optional) Provide a description about your license in the **Description** field. 
7. Click the **Generate License Key** button.
![](/images/hosting/deploy_manager_generate.png)

After you click **Generate License Key**, W&B redirects you to a Deployment License page. Within the Deployment License page you can view information about your license instance such as the Deployment ID, the organization the license belongs to, and more.

:::tip
View a specific license instance in one of two ways:
1. Navigate to the Deploy Manager UI and then click the name of the license instance.
2. Directly navigate to a specific license instance at `https://deploy.wandb.ai/DeploymentID` where `DeploymentID` is the unique ID assigned to your license instance.
:::

## 4. Add trial license to your local host
1. Within the Deployment License page of your license instance, click the **Copy License** button.
![](/images/hosting/deploy_manager_get_license.png)
2. Navigate to `http://localhost:8080/system-admin/`
3. Paste your license into to **License field**.
![](/images/hosting/License.gif)
4. Click the **Update settings** button.

## 5. Check your browser is running the W&B App UI
Check that W&B is running on your local machine. Navigate to `http://localhost:8080/home`. You should see the W&B App UI in your browser.

![](/images/hosting/check_local_host.png)

## 6. Add programmatic access to your local W&B instance

1. Navigate to `http://localhost:8080/authorize` to obtain your API key.
2. Within your terminal, execute the following:
   ```bash
   wandb login --host=http://localhost:8080/
   ```
   If you are already logged into W&B with a different count, add the `relogin` flag:
   ```bash
   wandb login --relogin --host=http://localhost:8080
   ```
3. Paste your API key when prompted.

W&B appends a `localhost` profile and your API key to your .netrc profile at `/Users/username/.netrc` for future automatic logins.

## Add a volume to retain data

All metadata and files you log to W&B are temporarily stored in the `https://deploy.wandb.ai/vol` directory. 

Mount a volume, or external storage, to your Docker container to retain files and metadata you store in your local W&B instance. W&B recommends that you store metadata in an external MySQL database and files in an external storage bucket such as Amazon S3.

:::info
Recall that your local W&B instance (created using a Trial W&B License), uses Docker to run W&B in your local browser. By default, data is not retained if a Docker container no longer exists. Data is lost when a Docker process dies if you do not mount a volume at `https://deploy.wandb.ai/vol`.
:::

For more information on how to mount a volume and for information on how Docker manages data, see [Manage data in Docker](https://docs.docker.com/storage/) page in the Docker documentation.

### Volume considerations
The underlying file store should be resizable.
W&B recommends that you set up alerts to inform you when you are close to reaching minimum storage thresholds so you can resize the underlying file system. 


:::info
For enterprise trials, W&B recommends at least 100 GB free space in the volume for non-image/video/audio heavy workloads.
:::

<!-- ## Next steps -->



