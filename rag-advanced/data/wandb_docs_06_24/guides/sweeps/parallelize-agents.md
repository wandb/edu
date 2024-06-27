---
description: Parallelize W&B Sweep agents on multi-core or multi-GPU machine.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Parallelize agents

<head>
  <title>Parallelize agents</title>
</head>

Parallelize your W&B Sweep agents on a multi-core or multi-GPU machine. Before you get started, ensure you have initialized your W&B Sweep. For more information on how to initialize a W&B Sweep, see [Initialize sweeps](./initialize-sweeps.md).

### Parallelize on a multi-CPU machine

Depending on your use case, explore the proceeding tabs to learn how to parallelize W&B Sweep agents using the CLI or within a Jupyter Notebook.


<Tabs
  defaultValue="cli_text"
  values={[
    {label: 'CLI', value: 'cli_text'},
    {label: 'Jupyter Notebook', value: 'jupyter'},
  ]}>
  <TabItem value="cli_text">

Use the [`wandb agent`](../../ref/cli/wandb-agent.md) command to parallelize your W&B Sweep agent across multiple CPUs with the terminal. Provide the sweep ID that was returned when you [initialized the sweep](./initialize-sweeps.md). 

1. Open more than one terminal window on your local machine.
2. Copy and paste the code snippet below and replace `sweep_id` with your sweep ID:


```bash
wandb agent sweep_id
```


  </TabItem>
  <TabItem value="jupyter">

Use the W&B Python SDK library to parallelize your W&B Sweep agent across multiple CPUs within Jupyter Notebooks. Ensure you have the sweep ID that was returned when you [initialized the sweep](./initialize-sweeps.md).  In addition, provide the name of the function the sweep will execute for the `function` parameter:

1. Open more than one Jupyter Notebook.
2. Copy and past the W&B Sweep ID on multiple Jupyter Notebooks to parallelize a W&B Sweep. For example, you can paste the following code snippet on multiple jupyter notebooks to paralleliz your sweep if you have the sweep ID stored in a variable called `sweep_id` and the name of the function is `function_name`: 


```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```

  </TabItem>
</Tabs>

### Parallelize on a multi-GPU machine

Follow the procedure outlined to parallelize your W&B Sweep agent across multiple GPUs with a terminal using CUDA Toolkit:

1. Open more than one terminal window on your local machine.
2. Specify the GPU instance to use with `CUDA_VISIBLE_DEVICES` when you start a W&B Sweep job ([`wandb agent`](../../ref/cli/wandb-agent.md)). Assign `CUDA_VISIBLE_DEVICES` an integer value corresponding to the GPU instance to use.

For example, suppose you have two NVIDIA GPUs on your local machine. Open a terminal window and set `CUDA_VISIBLE_DEVICES` to `0` (`CUDA_VISIBLE_DEVICES=0`). Replace `sweep_ID` in the proceeding example with the W&B Sweep ID that is returned when you initialized a W&B Sweep:

Terminal 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

Open a second terminal window. Set `CUDA_VISIBLE_DEVICES` to `1` (`CUDA_VISIBLE_DEVICES=1`). Paste the same W&B Sweep ID for the `sweep_ID` mentioned in the proceeding code snippet:

Terminal 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```
