# wandb agent

**Usage**

`wandb agent [OPTIONS] SWEEP_ID`

**Summary**

Run the W&B agent

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -p, --project | The name of the project where W&B runs created from the   sweep are sent to. If the project is not specified, the run is sent to a project labeled 'Uncategorized'. |
| -e, --entity | The username or team name where you want to send W&B   runs created by the sweep to. Ensure that the entity you specify already exists. If you don't specify an entity,   the run will be sent to your default entity, which is usually your username. |
| --count | The max number of runs for this agent. |

