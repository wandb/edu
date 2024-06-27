---
description: Set W&B environment variables.
displayed_sidebar: default
---

# Environment Variables

<head>
  <title>W&B Environment Variables</title>
</head>

When you're running a script in an automated environment, you can control **wandb** with environment variables set before the script runs or within the script.

```bash
# This is secret and shouldn't be checked into version control
WANDB_API_KEY=$YOUR_API_KEY
# Name and notes optional
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
```

```bash
# Only needed if you don't check in the wandb/settings file
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# If you don't want your script to sync to the cloud
os.environ["WANDB_MODE"] = "offline"
```

## Optional Environment Variables

Use these optional environment variables to do things like set up authentication on remote machines.

| Variable name               | Usage                                  |
| --------------------------- | ---------- |
| **WANDB\_ANONYMOUS**        | Set this to "allow", "never", or "must" to let users create anonymous runs with secret urls.                                                    |
| **WANDB\_API\_KEY**         | Sets the authentication key associated with your account. You can find your key on [your settings page](https://app.wandb.ai/settings). This must be set if `wandb login` hasn't been run on the remote machine.               |
| **WANDB\_BASE\_URL**        | If you're using [wandb/local](../hosting/intro.md) you should set this environment variable to `http://YOUR_IP:YOUR_PORT`        |
| **WANDB\_CACHE\_DIR**       | This defaults to \~/.cache/wandb, you can override this location with this environment variable                    |
| **WANDB\_CONFIG\_DIR**      | This defaults to \~/.config/wandb, you can override this location with this environment variable                             |
| **WANDB\_CONFIG\_PATHS**    | Comma separated list of yaml files to load into wandb.config. See [config](./config.md#file-based-configs).                                          |
| **WANDB\_CONSOLE**          | Set this to "off" to disable stdout / stderr logging. This defaults to "on" in environments that support it.                                          |
| **WANDB\_DIR**              | Set this to an absolute path to store all generated files here instead of the _wandb_ directory relative to your training script. _be sure this directory exists and the user your process runs as can write to it_                  |
| **WANDB\_DISABLE\_GIT**     | Prevent wandb from probing for a git repository and capturing the latest commit / diff.      |
| **WANDB\_DISABLE\_CODE**    | Set this to true to prevent wandb from saving notebooks or git diffs.  We'll still save the current commit if we're in a git repo.                   |
| **WANDB\_DOCKER**           | Set this to a docker image digest to enable restoring of runs. This is set automatically with the wandb docker command. You can obtain an image digest by running `wandb docker my/image/name:tag --digest`    |
| **WANDB\_ENTITY**           | The entity associated with your run. If you have run `wandb init` in the directory of your training script, it will create a directory named _wandb_ and will save a default entity which can be checked into source control. If you don't want to create that file or want to override the file you can use the environmental variable. |
| **WANDB\_ERROR\_REPORTING** | Set this to false to prevent wandb from logging fatal errors to its error tracking system.                             |
| **WANDB\_HOST**             | Set this to the hostname you want to see in the wandb interface if you don't want to use the system provided hostname                                |
| **WANDB\_IGNORE\_GLOBS**    | Set this to a comma separated list of file globs to ignore. These files will not be synced to the cloud.                              |
| **WANDB\_JOB\_NAME**        | Specify a name for any jobs created by `wandb`. For more information, see [create a job](../launch/create-launch-job.md)                                                                                                                                                                                                                        |
| **WANDB\_JOB\_TYPE**        | Specify the job type, like "training" or "evaluation" to indicate different types of runs. See [grouping](../runs/grouping.md) for more info.               |
| **WANDB\_MODE**             | If you set this to "offline" wandb will save your run metadata locally and not sync to the server. If you set this to "disabled" wandb will turn off completely.                  |
| **WANDB\_NAME**             | The human-readable name of your run. If not set it will be randomly generated for you                       |
| **WANDB\_NOTEBOOK\_NAME**   | If you're running in jupyter you can set the name of the notebook with this variable. We attempt to auto detect this.                    |
| **WANDB\_NOTES**            | Longer notes about your run. Markdown is allowed and you can edit this later in the UI.                                    |
| **WANDB\_PROJECT**          | The project associated with your run. This can also be set with `wandb init`, but the environmental variable will override the value.                               |
| **WANDB\_RESUME**           | By default this is set to _never_. If set to _auto_ wandb will automatically resume failed runs. If set to _must_ forces the run to exist on startup. If you want to always generate your own unique ids, set this to _allow_ and always set **WANDB\_RUN\_ID**.      |
| **WANDB\_RUN\_GROUP**       | Specify the experiment name to automatically group runs together. See [grouping](../runs/grouping.md) for more info.                                 |
| **WANDB\_RUN\_ID**          | Set this to a globally unique string (per project) corresponding to a single run of your script. It must be no longer than 64 characters. All non-word characters will be converted to dashes. This can be used to resume an existing run in cases of failure.      |
| **WANDB\_SILENT**           | Set this to **true** to silence wandb log statements. If this is set all logs will be written to **WANDB\_DIR**/debug.log               |
| **WANDB\_SHOW\_RUN**        | Set this to **true** to automatically open a browser with the run url if your operating system supports it.        |
| **WANDB\_TAGS**             | A comma separated list of tags to be applied to the run.                 |
| **WANDB\_USERNAME**         | The username of a member of your team associated with the run. This can be used along with a service account API key to enable attribution of automated runs to members of your team.               |
| **WANDB\_USER\_EMAIL**      | The email of a member of your team associated with the run. This can be used along with a service account API key to enable attribution of automated runs to members of your team.            |

## Singularity Environments

If you're running containers in [Singularity](https://singularity.lbl.gov/index.html) you can pass environment variables by pre-pending the above variables with **SINGULARITYENV\_**. More details about Singularity environment variables can be found [here](https://singularity.lbl.gov/docs-environment-metadata#environment).

## Running on AWS

If you're running batch jobs in AWS, it's easy to authenticate your machines with your W&B credentials. Get your API key from your [settings page](https://app.wandb.ai/settings), and set the WANDB\_API\_KEY environment variable in the [AWS batch job spec](https://docs.aws.amazon.com/batch/latest/userguide/job\_definition\_parameters.html#parameters).

## Common Questions

### Automated runs and service accounts

If you have automated tests or internal tools that launch runs logging to W&B, create a **Service Account** on your team settings page. This will allow you to use a service API key for your automated jobs. If you want to attribute service account jobs to a specific user, you can use the **WANDB\_USERNAME** or **WANDB\_USER\_EMAIL** environment variables.

![Create a service account on your team settings page for automated jobs](/images/track/common_questions_automate_runs.png)

This is useful for continuous integration and tools like TravisCI or CircleCI if you're setting up automated unit tests.

### Do environment variables overwrite the parameters passed to wandb.init()?

Arguments passed to `wandb.init` take precedence over the environment. You could call `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))` if you want to have a default other than the system default when the environment variable isn't set.

### Turn off logging

The command `wandb offline` sets an environment variable, `WANDB_MODE=offline` . This stops any data from syncing from your machine to the remote wandb server. If you have multiple projects, they will all stop syncing logged data to W&B servers.

To quiet the warning messages:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```

### Multiple wandb users on shared machines

If you're using a shared machine and another person is a wandb user, it's easy to make sure your runs are always logged to the proper account. Set the WANDB\_API\_KEY environment variable to authenticate. If you source it in your env, when you log in you'll have the right credentials, or you can set the environment variable from your script.

Run this command `export WANDB_API_KEY=X` where X is your API key. When you're logged in, you can find your API key at [wandb.ai/authorize](https://app.wandb.ai/authorize).
