# wandb docker

**Usage**

`wandb docker [OPTIONS] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**Summary**

Run your code in a docker container.

W&B docker lets you run your code in a docker image ensuring wandb is
configured. It adds the WANDB_DOCKER and WANDB_API_KEY environment variables
to your container and mounts the current directory in /app by default.  You
can pass additional args which will be added to `docker run` before the
image name is declared, we'll choose a default image for you if one isn't
passed:

```sh wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-
images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker
wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5" ```

By default, we override the entrypoint to check for the existence of wandb
and install it if not present.  If you pass the --jupyter flag we will
ensure jupyter is installed and start jupyter lab on port 8888.  If we
detect nvidia-docker on your system we will use the nvidia runtime.  If you
just want wandb to set environment variable to an existing docker run
command, see the wandb docker-run command.

**Options**

| **Option** | **Description** |
| :--- | :--- |
| --nvidia / --no-nvidia | Use the nvidia runtime, defaults to nvidia if   nvidia-docker is present |
| --digest | Output the image digest and exit |
| --jupyter / --no-jupyter | Run jupyter lab in the container |
| --dir | Which directory to mount the code in the container |
| --no-dir | Don't mount the current directory |
| --shell | The shell to start the container with |
| --port | The host port to bind jupyter on |
| --cmd | The command to run in the container |
| --no-tty | Run the command without a tty |

