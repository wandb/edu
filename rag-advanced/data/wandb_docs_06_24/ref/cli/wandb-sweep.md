# wandb sweep

**Usage**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**Summary**

Initialize a hyperparameter sweep. Search for hyperparameters that optimizes
a cost function of a machine learning model by testing various combinations.

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -p, --project | The name of the project where W&B runs created from   the sweep are sent to. If the project is not specified, the run is sent to a project labeled   Uncategorized. |
| -e, --entity | The username or team name where you want to send W&B   runs created by the sweep to. Ensure that the entity you specify already exists. If you don't specify an   entity, the run will be sent to your default entity, which is usually your username. |
| --controller | Run local controller |
| --verbose | Display verbose output |
| --name | The name of the sweep. The sweep ID is used if no name   is specified. |
| --program | Set sweep program |
| --update | Update pending sweep |
| --stop | Finish a sweep to stop running new runs and let   currently running runs finish. |
| --cancel | Cancel a sweep to kill all running runs and stop   running new runs. |
| --pause | Pause a sweep to temporarily stop running new runs. |
| --resume | Resume a sweep to continue running new runs. |
| --prior_run | ID of an existing run to add to this sweep |

