# wandb launch

**Usage**

`wandb launch [OPTIONS]`

**Summary**

Launch or queue a W&B Job. See https://wandb.me/launch

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -u, --uri (str) | Local path or git repo uri to launch. If   provided this command will create a job from the specified uri. |
| -j, --job (str) | Name of the job to launch. If passed in,   launch does not require a uri. |
| --entry-point | Entry point within project. [default: main].   If the entry point is not found, attempts to run the project file with the specified name   as a script, using 'python' to run .py files and the default shell (specified by   environment variable $SHELL) to run .sh files. If passed in, will override the   entrypoint value passed in using a config file. |
| --build-context (str) | Path to the build context within the source   code. Defaults to the root of the source code. Compatible only with -u. |
| --name | Name of the run under which to launch the   run. If not specified, a random run name will be used to launch run. If passed in,   will override the name passed in using a config file. |
| -e, --entity (str) | Name of the target entity which the new run   will be sent to. Defaults to using the entity set by local wandb/settings folder.   If passed in, will override the entity value passed in using a config file. |
| -p, --project (str) | Name of the target project which the new run   will be sent to. Defaults to using the project name given by the source uri or for   github runs, the git repo name. If passed in, will override the project value passed   in using a config file. |
| -r, --resource | Execution resource to use for run. Supported   values: 'local-process', 'local-container', 'kubernetes', 'sagemaker', 'gcp-vertex'.   This is now a required parameter if pushing to a queue with no resource configuration.   If passed in, will override the resource value passed in using a config file. |
| -d, --docker-image | Specific docker image you'd like to use. In the form name:tag. If passed in, will   override the docker image value passed in using a config file. |
| -c, --config | Path to JSON file (must end in '.json') or   JSON string which will be passed as a launch config. Dictation how the launched run will   be configured. |
| -v, --set-var | Set template variable values for queues with   allow listing enabled, as key-value pairs e.g. `--set-var key1=value1 --set-var   key2=value2` |
| -q, --queue | Name of run queue to push to. If none,   launches single run directly. If supplied without an argument (`--queue`), defaults to   queue 'default'. Else, if name supplied, specified run queue must exist under the   project and entity supplied. |
| --async | Flag to run the job asynchronously. Defaults   to false, i.e. unless --async is set, wandb launch will wait for the job to finish. This   option is incompatible with --queue; asynchronous options when running with an   agent should be set on wandb launch-agent. |
| --resource-args | Path to JSON file (must end in '.json') or   JSON string which will be passed as resource args to the compute resource. The exact   content which should be provided is different for each execution backend. See   documentation for layout of this file. |
| --dockerfile | Path to the Dockerfile used to build the   job, relative to the job's root |
| --priority [critical|high|medium|low] | When --queue is passed, set the priority of the job. Launch jobs with higher priority   are served first.  The order, from highest to lowest priority, is: critical, high,   medium, low |

