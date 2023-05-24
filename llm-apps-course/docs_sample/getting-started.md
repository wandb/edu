---
description: Getting started guide for W&B Launch.
---
# Getting started

Follow this guide to get started using W&B Launch. This guide will walk you through the setup of the fundamental components of a launch workflow: a **job**, **launch queue**, and **launch queue**. 

* A **job** is a reusable blueprint for configuring and executing a step of your ML workflow. Jobs can be automatically captured from your workloads when your track those workloads with W&B. In this guide will create and then launch a job that trains a neural network.

* A **launch queue** is a place where you can submit your jobs for execution on a particular compute resource. For example, you might create separate launch queues for submitting jobs that should be run on specific GPU server, or a particular kubernetes cluster. The queue we will create in this guide will be used to submit jobs that will run on your machine via Docker.

* A **launch agent** is a long-running process that polls on one or more launch queues and executes the jobs that it pops from the queue. A launch agent can be started with the `wandb launch-agent` command and is capable on launching jobs onto a multitude of compute platforms, including docker, kubernetes, sagemaker, and more. In this example, you will run a launch agent that will pop jobs from your queue and execute them on its local host using Docker.

## Before you get started
Before you get started, ensure you [enable the W&B Launch UI](./intro.md) and install Docker on the machine where you will run your launch agent.

See the [Docker documentation](https://docs.docker.com/get-docker/) for more information on how to install Docker, and make sure the docker daemon is running on your machine before you proceed.

If you want the agent to make use of GPUs, you will also need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Create a job

Jobs are created automatically from any W&B run that has associated source code. For more details on how source code can be associated with a run, see [these docs](create-job.md).

Copy the following Python code to your machine in a file named `train.py`.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import wandb


class FashionCNN(nn.Module):
    """Simple CNN for Fashion MNIST."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def train_fmnist(config):
    # Pass config into wandb.init
    with wandb.init(project="launch-quickstart", config=config):
        
        # Log training code to W&B as an Artifact.
        wandb.run.log_code()

        # Training setup
        config = wandb.config
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = FashionCNN()
        model.to(device)
        train_dataset = FashionMNIST(
            "./data/", download=True, train=True, transform=transforms.ToTensor()
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, pin_memory=True
        )
        error = nn.CrossEntropyLoss()
        learning_rate = config.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # We can pass our network to wandb.watch and automatically log gradients,
        # weights, topology, and more...
        wandb.watch(model, log="all", log_graph=True)

        # Epoch loop
        iter = 0
        losses = []
        for epoch in range(config.epochs):

            # Iterate over batches of the data
            for _, (images, labels) in enumerate(train_loader):

                iter += 1
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = error(outputs, labels)
                losses.append(loss.item())

                if iter % 100 == 1:
                    wandb.log(
                        {
                            "train/loss": sum(losses) / len(losses),  # Log average loss
                            "train/losses": wandb.Histogram(losses),  # Log all losses
                            "train/epoch": epoch,
                        }
                    )
                    losses = []

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    config = dict(epochs=1, batch_size=32, learning_rate=0.0001)
    train_fmnist(config)
```

The script above initializes a simple neural network and then trains that network to distinguish types of clothing in images from the [fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). The training is tracked with `wandb` and the source code is logged as an Artifact with `wandb.run.log_code()`. This means that when we run this script W&B will automatically create our first job.


To install dependencies and run the script, execute the following commands in your terminal:

```bash
pip install wandb>=0.13.8 torch torchvision
python train.py
```

Let the script run to completion and then move on to the next step. Your console output should look roughly like:

```
wandb: Currently logged in as: user. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.14.0
wandb: Run data is saved locally in /home/user/wandb/run-20230323_120437-p89pnj2u
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run comic-firebrand-342
wandb: ⭐️ View project at https://wandb.ai/username/launch-quickstart
wandb: 🚀 View run at https://wandb.ai/username/launch-quickstart/runs/p89pnj2u
wandb: logging graph, to disable use `wandb.watch(log_graph=False)`
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb: train/epoch ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:  train/loss █▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb: train/epoch 0
wandb:  train/loss 0.33282
wandb: 
wandb: 🚀 View run comic-firebrand-342 at: https://wandb.ai/username/launch-quickstart/runs/p89pnj2u
wandb: Synced 5 W&B file(s), 1 media file(s), 3 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20230323_120437-p89pnj2u/logs
```

Navigate to your new **launch-quickstart** project in your W&B account and open the jobs tab from the nav on the left side of the screen.

![](/images/launch/jobs-tab.png)

The **Jobs** page displays a list of W&B Jobs that were created from previously executed W&B Runs. You should see a job named **job-source-launch-quickstart-main.py**. You can edit the name of the job from the jobs page if you would like to make the job a bit more memorable. Click on your job to open a detailed view of your job including the source code and dependencies for the job and a list of runs that have been launched from this job.

## Create a queue

Navigate to [wandb.ai/launch](https://wandb.ai/launch). Click **create queue** button in the top right of the screen to start creating a new launch queue.

When you click the button, a drawer will slide from the right side of your screen and present you with some options for your new queue:

* **Entity**: the owner of the queue, which can either be your W&B account or any W&B team you are a member of. For this demo, we recommend setting up a personal queue.
* **Queue name**: the name of the queue. Make this whatever you want!
* **Resource**: the execution platform for jobs in this queue. Check out the other options, but for this walkthrough we will use the default: **Docker container**.
* **Configuration**: json configuration that will passed to any launch agent that polls on this queue. This can be left blank, but some additional configuration does need to be provided in order to leverage GPU.

![](/images/launch/create-queue.gif)

:::info
Add the following resource configuration in order to use GPUs in jobs submitted to this queue:

```json
{
    "gpus": "all"
}
```

The `gpus` key of the resource configuration is used to pass values to the `--gpus` argument of `docker run`. This argument can be used to control which GPUs will be used for by a launch agent when it picks up runs from this queue. For more information, see the relevant [NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration).

For jobs that use tensorflow on GPU, you may also need to specify a custom base image for the container build that the agent will perform in order for your runs to properly utilize GPUs. This can be done by adding an image tag under the `builder.cuda.base_image` key to the resource configuration. For example:

```json
{
    "gpus": "all",
    "builder": {
        "cuda": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```
:::

## Add a job to your queue

Head back to the page for your job. It should look something like the image below:

![](/images/launch/launch-job.gif)

Click the **Launch** button in the top right to launch a new run from this job. A drawer will slide from the right side of your screen and present you with some options for your new run:

* **Job version**: the version of the job to launch. Jobs are versioned like any other W&B Artifact. Different versions of the same job will be created if you make modifications to the software dependencies or source code used to run the job. Since we only have one version, we will select the default **@latest** version.
* **Overrides**: new values for any of jobs inputs. These can be used to change the entrypoint command, arguments, or values in the `wandb.config` of your new run. Our run had 3 values in the `wandb.config`: `epochs`, `batch_size`, and `learning_rate`. We can override any of these values by specifying them in the overrides field. We can also paste values from other runs using this job by clicking the **Paste from...** button.
* **Queue**: the queue to launch the run on. We will select the queue we created in the previous step.
* **Resource config**: This is non-editable and shows the queue configuration, so we can see how our job will be run when we add it to this queue.

![](/images/launch/launch-job.gif)

Once you have configured your job as desired, click the **launch now** button at the bottom of the drawer to enqueue your launch job.

## Start a launch agent

To execute your job, you will need to start a launch agent polling on your launch queue.

1. From [wandb.ai/launch](https://wandb.ai/launch) navigate to the page for your launch queue.
2. Click the **Add an agent** button.
3. A modal will appear with a W&B CLI command. Copy this and paste this command into your terminal.

![](/images/launch/activate_starter_queue_agent.png)

In general, the command to start a launch agent is:

```bash
wandb launch-agent -e <entity-name> -q <queue-name>
```

Within your terminal, you will see the agent begin to poll for queues. The agent should pick up the job you enqueued earlier and begin to execute. First, the agent will build a container image from the job version you selected. Then, the agent will execute the job on its local host via `docker run`.

That’s it! Navigate to your Launch workspace or your terminal to see the status of your launch job. Jobs are executed in first in, first out order (FIFO). All jobs pushed to the queue will use the same resource type and resource arguments.
