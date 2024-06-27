import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Spin up a single node GPU cluster with Minikube

Set up W&B Launch on a Minikube cluster that can schedule and run GPU workloads. 

:::info
This tutorial is intended to guide users with direct access to a machine that has multiple GPUs. This tutorial is not intended for users who rent a cloud machine.

W&B recommends you create a Kubernetes cluster with GPU support that uses your cloud provider, if you want to set up a minikube cluster on a cloud machine. For example, AWS, GCP, Azure, Coreweave, and other cloud providers have tools to create Kubernetes clusters with GPU support.

W&B recommends you use a [Launch Docker queue](/guides/launch/setup-launch-docker) if you want to set up a minikube cluster for scheduling GPUs on a machine that has a single GPU. You can still follow the tutorial for fun, but the GPU scheduling will not be very useful.
:::

## Background

<!-- Paraphrase commented paragraph below in more clear wording. -->

The [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) has made it easy to run GPU-enabled workflows on Docker. One limitation is a lack of native support for scheduling GPU by volume. If you want to use a GPU with the `docker run` command you must either request specific GPU by ID or all GPU present, which makes many distributed GPU enabled workloads impractical. Kubernetes offers support for scheduling by a volume request, but setting up a local Kubernetes cluster with GPU scheduling can take considerable time and effort, until recently. Minikube, one of the most popular tools for running single node Kubernetes clusters, recently released [support for GPU scheduling](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) 🎉 In this tutorial, we will create a Minikube cluster on a multi-GPU machine and launch concurrent stable diffusion inference jobs to the cluster using W&B Launch 🚀

## Prerequisites

Before getting started, you will need:

1. A W&B account.
2. Linux machine with the following installed and running:
   1. Docker runtime
   2. Drivers for any GPU you want to use
   3. Nvidia container toolkit

:::note
For testing and creating this tutorial, we used an `n1-standard-16` Google Cloud Compute Engine instance with 4 NVIDIA Tesla T4 GPU connected.
:::

## Create a queue for launch jobs

First, create a launch queue for our launch jobs. 

1. Navigate to [wandb.ai/launch](https://wandb.ai/launch) (or `<your-wandb-url>/launch` if you use a private W&B server).
2. In the top right corner of your screen, click the blue **Create a queue** button. A queue creation drawer will slide out from the right side of your screen.
3. Select an entity, enter a name, and select **Kubernetes** as the type for your queue.
4. The **Config** section of the drawer is where you will enter a [Kubernetes job specification](https://kubernetes.io/docs/concepts/workloads/controllers/job/) for the launch queue. Any runs launched from this queue will be created using this job specification, so you can modify this configuration as needed to customize your jobs. For this tutorial, you can copy and paste the sample config below in your queue config as YAML or JSON:

<Tabs
defaultValue="yaml"
values={[
{ label: "YAML", value: "yaml", },
{ label: "JSON", value: "json", },
]}>

<TabItem value="yaml">

```yaml
spec:
  template:
    spec:
      containers:
        - image: ${image_uri}
          resources:
            limits:
              cpu: 4
              memory: 12Gi
              nvidia.com/gpu: '{{gpus}}'
      restartPolicy: Never
  backoffLimit: 0
```

</TabItem>

<TabItem value="json">

```json
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "image": "${image_uri}",
            "resources": {
              "limits": {
                "cpu": 4,
                "memory": "12Gi",
                "nvidia.com/gpu": "{{gpus}}"
              }
            }
          }
        ],
        "restartPolicy": "Never"
      }
    },
    "backoffLimit": 0
  }
}
```

</TabItem>
</Tabs>

For more information about queue configurations, see the [Set up Launch on Kubernetes](/guides/launch/setup-launch-kubernetes.md) and the [Advanced queue setup guide](/guides/launch/setup-queue-advanced.md).   


The `${image_uri}` and `{{gpus}}` strings are examples of the two kinds of
variable templates that you can use in your queue configuration. The `${image_uri}`
template will be replaced with the image URI of the job you are launching by the
agent. The `{{gpus}}` template will be used to create a template variable that
you can override from the launch UI, CLI, or SDK when submitting a job. These values
are placed in the job specification so that they will modify the correct fields
to control the image and GPU resources used by the job.

5. Click the **Parse configuration** button to begin customizing your `gpus` template
variable. 
6. Set the **Type** to `Integer` and the **Default**, **Min**, and **Max** to values of your choosing.
Attempts to submit a run to this queue which violate the constraints of the template variable will
be rejected.

![Image of queue creation drawer with gpus template variable](/images/tutorials/minikube_gpu/create_queue.png)

7. Click **Create queue** to create your queue. You will be redirected to the queue page for your new queue.


In the next section, we will set up an agent that can pull and execute jobs from the queue you created.

## Setup Docker + NVIDIA CTK

If you already have Docker and the Nvidia container toolkit setup on your machine, you can skip this section.

Refer to [Docker’s documentation](https://docs.docker.com/engine/install/) for instructions on setting up the Docker container engine on your system.

Once you have Docker installed, install the Nvidia container toolkit [following the instructions in Nvidia’s documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

To validate that your container runtime has access to your GPU, you can run:

```bash
docker run --gpus all ubuntu nvidia-smi
```

You should see `nvidia-smi` output describing the GPU connected to your machine. For example, on our setup the output looks like this:

```
Wed Nov  8 23:25:53 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   38C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla T4            Off  | 00000000:00:05.0 Off |                    0 |
| N/A   38C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla T4            Off  | 00000000:00:06.0 Off |                    0 |
| N/A   40C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla T4            Off  | 00000000:00:07.0 Off |                    0 |
| N/A   39C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Setup Minikube

Minikube’s GPU support requires version `v1.32.0` or later. Refer to [Minikube’s install documentation](https://minikube.sigs.k8s.io/docs/start/) for up to date installation help. For this tutorial, we installed the latest Minikube release using the command:

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

The next step is to start a minikube cluster using your GPU. On your machine, run:

```yaml
minikube start --gpus all
```

The output of the command above will indicate whether a cluster has been successfully created.

## Start launch agent

The launch agent for your new cluster can either be started by invoking `wandb launch-agent` directly or by deploying the launch agent using a [helm chart managed by W&B](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent). 

In this tutorial we will run the agent directly on our host machine. 

:::tip
Running the agent outside of a container also means we can use the local Docker host to build images for our cluster to run.
:::

To run the agent locally, make sure your default Kubernetes API context refers to the Minikube cluster. Then, execute the following:

```bash
pip install wandb[launch]
```

to install the agent’s dependencies. To setup authentication for the agent, run `wandb login` or set the `WANDB_API_KEY` environment variable.

To start the agent, type and execute the following:

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

Within your terminal you should see the launch agent start to print polling message. 

Congratulations, you have a launch agent polling your launch queue! When a job is added to your queue, your agent will pick it up and schedule it to run on your Minikube cluster.

## Launch a job

Let's send a job to our agent. You can launch a simple "hello world" from a terminal logged into your W&B account with:

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

You can test with any job or image you like, but make sure your cluster can pull your image.  See [Minikube’s documentation](https://minikube.sigs.k8s.io/docs/handbook/registry/) for additional guidance. You can also [test using one of our public jobs](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman).

## (Optional) Model and data caching with NFS

For ML workloads we will often want multiple jobs to have access to the same data. For example, you might want to have a shared cache to avoid repeatedly downloading large assets like datasets or model weights. Kubernetes supports this through [persistent volumes and persistent volume claims](https://kubernetes.io/docs/concepts/storage/persistent-volumes/). Persistent volumes can be used to create `volumeMounts` in our Kubernetes workloads, providing direct filesystem access to the shared cache.

In this step, we will set up a network file system (NFS) server that can be used as a shared cache for model weights. The first step is to install and configure NFS. This process varies by operating system. Since our VM is running Ubuntu, we installed nfs-kernel-server and configured an export at `/srv/nfs/kubedata`:

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

Keep note of the export location of the server in your host filesystem, as well as the local IP address of your NFS server, i.e. your host machine. We will need this pieces of information in the next step.

Next, you will need to create a persistent volume and persistent volume claim for this NFS. Persistent volumes are highly customizable, but we will use straightforward configuration here for the sake of simplicity.

Copy the yaml below into a file named `nfs-persistent-volume.yaml` , making sure to fill out your desired volume capacity and claim request. The `PersistentVolume.spec.capcity.storage` field controls the maximum size of the underlying volume. The `PersistentVolumeClaim.spec.resources.requests.stroage` can be used to limit the volume capacity allotted for a particular claim. For our use case, it makes sense to use the same value for each.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # Set this to your desired capacity.
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: Fill this in.
    path: '/srv/nfs/kubedata' # Or your custom path
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nfs-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi # Set this to your desired capacity.
  storageClassName: ''
  volumeName: nfs-pv
```

Create the resources in your cluster with:

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

In order for our runs to make use of this cache, we will need to add `volumes` and `volumeMounts` to our launch queue config. To edit the launch config, head back to [wandb.ai/launch](http://wandb.ai/launch) (or \<your-wandb-url\>/launch for users on wandb server), find your queue, click to the queue page, and then click the **Edit config** tab. The original config can be modified to:

<Tabs
defaultValue="yaml"
values={[
{ label: "YAML", value: "yaml", },
{ label: "JSON", value: "json", },
]}>

<TabItem value="yaml">

```yaml
spec:
  template:
    spec:
      containers:
        - image: ${image_uri}
          resources:
            limits:
              cpu: 4
              memory: 12Gi
              nvidia.com/gpu: "{{gpus}}"
					volumeMounts:
            - name: nfs-storage
              mountPath: /root/.cache
      restartPolicy: Never
			volumes:
        - name: nfs-storage
          persistentVolumeClaim:
            claimName: nfs-pvc
  backoffLimit: 0
```

</TabItem>

<TabItem value="json">

```json
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "image": "${image_uri}",
            "resources": {
              "limits": {
                "cpu": 4,
                "memory": "12Gi",
                "nvidia.com/gpu": "{{gpus}}"
              },
              "volumeMounts": [
                {
                  "name": "nfs-storage",
                  "mountPath": "/root/.cache"
                }
              ]
            }
          }
        ],
        "restartPolicy": "Never",
        "volumes": [
          {
            "name": "nfs-storage",
            "persistentVolumeClaim": {
              "claimName": "nfs-pvc"
            }
          }
        ]
      }
    },
    "backoffLimit": 0
  }
}
```

</TabItem>

</Tabs>

Now, our NFS will be mounted at `/root/.cache` in the containers running our jobs. The mount path will require adjustment if your container runs as a user other than `root`. Huggingface’s libraries and W&B Artifacts both make use of `$HOME/.cache/` by default, so downloads should only happen once.

## Playing with stable diffusion

To test out our new system, we are going to experiment with stable diffusion’s inference parameters.
To run a simple stable diffusion inference job with a default prompt and sane
parameters, you can run:

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

The command above will submit the container image `wandb/job_stable_diffusion_inference:main` to your queue.
Once your agent picks up the job and schedules it for execution on your cluster,
it may take a while for the image to be pulled, depending on your connection.
You can follow the status of the job on the queue page on [wandb.ai/launch](http://wandb.ai/launch) (or \<your-wandb-url\>/launch for users on wandb server).

Once the run has finished, you should have a job artifact in the project you specified.
You can check your project's job page (`<project-url>/jobs`) to find the job artifact. Its default name should
be `job-wandb_job_stable_diffusion_inference` but you can change that to whatever you like on the job's page
by clicking the pencil icon next to the job name.

You can now use this job to run more stable diffusion inference on your cluster!
From the job page, we can click the **Launch** button in the top right hand corner
to configure a new inference job and submit it to our queue. The job configuration
page will be pre-populated with the parameters from the original run, but you can
change them to whatever you like by modifying their values in the **Overrides** section
of the launch drawer.

![Image of launch UI for stable diffusion inference job](/images/tutorials/minikube_gpu/sd_launch_drawer.png)
