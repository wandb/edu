---
description: Pause, resume, and cancel a W&B Sweep with the CLI.
displayed_sidebar: default
---

# Pause, resume, stop and cancel sweeps

<head>
    <title>Pause, resume, stop or cancel W&B Sweeps</title>
</head>

Pause, resume, and cancel a W&B Sweep with the CLI.  Pausing a W&B Sweep tells the W&B agent that new W&B Runs should not be executed until the Sweep is resumed. Resuming a Sweep tells the agent to continue executing new W&B Runs. Stopping a W&B Sweep tells the W&B Sweep agent to stop creating or executing new W&B Runs. Cancelling a W&B Sweep tells the Sweep agent to kill currently executing W&B Runs and stop executing new Runs.

In each case, provide the W&B Sweep ID that was generated when you initialized a W&B Sweep. Optionally open a new terminal window to execute the proceeding commands. A new terminal window makes it easier to execute a command if a W&B Sweep is printing output statements to your current terminal window.

Use the following guidance to pause, resume, and cancel sweeps.

### Pause sweeps

Pause a W&B Sweep so it temporarily stops executing new W&B Runs. Use the `wandb sweep --pause` command to pause a W&B Sweep. Provide the W&B Sweep ID that you want to pause.

```bash
wandb sweep --pause entity/project/sweep_ID
```

### Resume sweeps

Resume a paused W&B Sweep with the `wandb sweep --resume` command. Provide the W&B Sweep ID that you want to resume:

```bash
wandb sweep --resume entity/project/sweep_ID
```

### Stop sweeps

Finish a W&B sweep to stop executing newW&B Runs and let currently executing Runs finish.

```bash
wandb sweep --stop entity/project/sweep_ID
```

### Cancel sweeps

Cancel a sweep to kill all running runs and stop running new runs. Use the `wandb sweep --cancel` command to cancel a W&B Sweep. Provide the W&B Sweep ID that you want to cancel.

```bash
wandb sweep --cancel entity/project/sweep_ID
```

For a full list of CLI command options, see the [wandb sweep](../../ref/cli/wandb-sweep.md) CLI Reference Guide.

### Pause, resume, stop, and cancel a sweep across multiple agents

Pause, resume, stop, or cancel a W&B Sweep across multiple agents from a single terminal. For example, suppose you have have a multi-core machine. After you initialize a W&B Sweep, you open new terminal windows and copy the Sweep ID to each new terminal.

Within any terminal, use the the `wandb sweep` CLI command to pause, resume, stop, or cancel a W&B Sweep. For example, the proceeding code snippet demonstrates how to pause a W&B Sweep across multiple agents with the CLI:

```
wandb sweep --pause entity/project/sweep_ID
```

Specify the `--resume` flag along with the Sweep ID to resume the Sweep across your agents:

```
wandb sweep --resume entity/project/sweep_ID
```

For more information on how to parallelize W&B agents, see [Parallelize agents](./parallelize-agents.md).
