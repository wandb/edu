# wandb launch-agent

**Usage**

`wandb launch-agent [OPTIONS]`

**Summary**

Run a W&B launch agent.

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -q, --queue <queue(s)> | The name of a queue for the agent to watch. Multiple   -q flags supported. |
| -e, --entity | The entity to use. Defaults to current logged-in   user |
| -l, --log-file | Destination for internal agent logs. Use - for   stdout. By default all agents logs will go to debug.log in your wandb/ subdirectory or WANDB_DIR   if set. |
| -j, --max-jobs | The maximum number of launch jobs this agent can run   in parallel. Defaults to 1. Set to -1 for no upper limit |
| -c, --config | path to the agent config yaml to use |
| -v, --verbose | Display verbose output |

