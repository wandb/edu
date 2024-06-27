# wandb launch-sweep

**Usage**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**Summary**

Run a W&B launch sweep (Experimental).

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -q, --queue | The name of a queue to push the sweep to |
| -p, --project | Name of the project which the agent will watch. If   passed in, will override the project value passed in using a config file |
| -e, --entity | The entity to use. Defaults to current logged-in user |
| -r, --resume_id | Resume a launch sweep by passing an 8-char sweep id.   Queue required |
| --prior_run | ID of an existing run to add to this sweep |

