# wandb docker-run

**Usage**

`wandb docker-run [OPTIONS] [DOCKER_RUN_ARGS]...`

**Summary**

Wrap `docker run` and adds WANDB_API_KEY and WANDB_DOCKER environment
variables.

This will also set the runtime to nvidia if the nvidia-docker executable is
present on the system and --runtime wasn't set.

See `docker run --help` for more details.

**Options**

| **Option** | **Description** |
| :--- | :--- |

