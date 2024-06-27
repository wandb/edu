---
description: Resume a paused or exited W&B Run
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Resume Runs

<head>
  <title>Resume W&B Runs</title>
</head>

Specify how a run should behave in the event that that run stops or crashes. To resume or enable a run to automatically resume, you will need to specify the unique run ID associated with that run for the `id` parameter:
```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

:::tip
W&B encourages you to provide the name of the W&B Project where you want to store the run.
:::

Pass one of the following arguments to the `resume` parameter to determine how W&B should respond. In each case, W&B first checks if the run ID already exists. 

|Argument | Description | Run ID exists| Run ID does not exist | Use case |
| --- | --- | -- | --| -- |
| `"must"` | W&B must resume run specified by the run ID. | W&B resumes run with the same run ID. | W&B raises an error. | Resume a run that must use the same run ID.  |
| `"allow"`| Allow W&B to resume run if run ID exists. | W&B resumes run with the same run ID. | W&B initializes a new run with specified run ID. | Resume a run without overriding an existing run. |
| `"never"`| Never allow W&B to resume a run specified by run ID. | W&B raises an error. | W&B initializes a new run with specified run ID. | |

<!-- - `"must"`:  If the run ID exists, W&B resumes the run with that run ID. If the run ID does not exist, W&B does nothing. 
- `"allow"`:  If the run ID exists, W&B resumes the run with that run ID. If the run ID does not exist, W&B initializes a new run with the specified run ID.
- `"never"`: If the run ID exists, W&B does nothing. If the run ID does not exist, W&B starts a new run with the specified run ID.  -->

You can also specify `resume="auto"` to let W&B to automatically try to restart the run on your behalf. However, you will need to ensure that you restart your run from the same directory. See the [Enable runs to automatically resume](#enable-runs-to-automatically-resume) section for more information.

For all the examples below, replace values enclosed within `<>` with your own.

## Resume a run that must use the same run ID
Resume a run that uses the same run ID if it is stopped, crashes or fails. To do so, initialize a run and specify the following:

* Set the `resume` parameter to `"must"` (`resume="must"`) 
* Provide the run ID of the run that stopped or crashed

<!-- Set the `resume` parameter to `"must"` (`resume="must"`) when you initialize the run and provide the run ID of the run that stopped or crashed.  -->

The following code snippet shows how to accomplish this with the W&B Python SDK:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

:::caution
Unexpected results will occur if multiple processes use the same `id` concurrently. 


For more information on  how to manage multiple processes, see the [Log distributed training experiments](../track/log/distributed-training.md) 
:::

## Resume a run without overriding the existing run
Resume a run that stopped or crashed without overriding the existing run. This is especially helpful if your process doesn't exit successfully. The next time you start W&B, W&B will start logging from the last step.

Set the `resume` parameter to `"allow"` (`resume="allow"`) when you initialize a run with W&B. Provide the run ID of the run that stopped or crashed. The following code snippet shows how to accomplish this with the W&B Python SDK:

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```


## Enable runs to automatically resume 
The following code snippet shows how to enable runs to automatically resume with the Python SDK or with environment variables. 

<Tabs
  defaultValue="python"
  values={[
    {label: 'W&B Python SDK', value: 'python'},
    {label: 'Shell script', value: 'bash'},
  ]}>
  <TabItem value="python">

The following code snippet shows how to specify a W&B run ID with the Python SDK. 

Replace values enclosed within `<>` with your own:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

  </TabItem>
  <TabItem value="bash">

The following example shows how to specify the W&B `WANDB_RUN_ID` variable in a bash script: 

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
Within your terminal, you could run the shell script along with the W&B run ID. The following code snippet passes the run ID `akj172`: 

```bash
sh run_experiment.sh akj172 
```

  </TabItem>
</Tabs>

:::important
Automatic resuming only works if the process is restarted on top of the same filesystem as the failed process. 
:::

For example, suppose you execute a python script called `train.py` in a directory called called `Users/AwesomeEmployee/Desktop/ImageClassify/training/`. Within `train.py`, the script creates a run that enables automatic resuming. Suppose next that the training script is stopped.  To resume this run, you would need to restart your `train.py` script within `Users/AwesomeEmployee/Desktop/ImageClassify/training/` .


:::tip
If you can not share a filesystem, specify the `WANDB_RUN_ID` environment variable or pass the run ID with the W&B Python SDK. See the [Create a run](./intro.md#create-a-run) section in the "What are runs?" page for more information on run IDs.
:::





## Resume preemptible Sweeps runs
Automatically requeue interrupted [sweep](../sweeps/intro.md) runs. This is particularly useful if you run a sweep agent in a compute environment that is subject to preemption such as a SLURM job in a preemptible queue, an EC2 spot instance, or a Google Cloud preemptible VM.

Use the [`mark_preempting`](../../ref/python/run.md#markpreempting) function to enable W&B to automatically requeue interrupted sweep runs. For example, the following code snippet

```python
run = wandb.init()  # Initialize a run
run.mark_preempting()
```
The following table outlines how W&B handles runs based on the exit status of the a sweep run.

|Status| Behavior |
|------| ---------|
|Status code 0| Run is considered to have terminated successfully and it will not be requeued.  |
|Nonzero status| W&B automatically appends the run to a run queue associated with the sweep.|
|No status| Run is added to the sweep run queue. Sweep agents consume runs off the run queue until the queue is empty. Once the queue is empty, the sweep queue resumes generating new runs based on the sweep search algorithm.|



