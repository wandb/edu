---
description: Explore how to use W&B Tables with this 5 minute Quickstart.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Walkthrough

The following Quickstart demonstrates how to log data tables, visualize data, and query data.


Select the button below to try a PyTorch Quickstart example project on MNIST data. 

## 1. Log a table
Log a table with W&B. You can either construct a new table or pass a Pandas DataFrame.

<Tabs
  defaultValue="construct"
  values={[
    {label: 'Construct a table', value: 'construct'},
    {label: 'Pandas DataFrame', value: 'pandas'},
  ]}>
  <TabItem value="construct">

To construct and log a new Table, you will use:
- [`wandb.init()`](../../ref/python/init.md): Create a [run](../runs/intro.md) to track results.
- [`wandb.Table()`](../../ref/python/data-types/table.md): Create a new table object.
  - `columns`: Set the column names.
  - `data`: Set the contents of each row.
- [`run.log()`](../../ref/python/log.md): Log the table to save it to W&B.

Here's an example:
```python
import wandb

run = wandb.init(project="table-test")
# Create and log a new table.
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```
  </TabItem>
  <TabItem value="pandas">

Pass a Pandas DataFrame to `wandb.Table()` to create a new table.

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

run = wandb.init(project="df-table")
my_table = wandb.Table(dataframe=df)
wandb.log({"Table Name": my_table})
```

For more information on supported data types, see the [`wandb.Table`](../../ref/python/data-types/table.md) in the W&B API Reference Guide.

  </TabItem>
</Tabs>


## 2. Visualize tables in your project workspace

View the resulting table in your workspace. 

1. Navigate to your project in the W&B App.
2. Select the name of your run in your project workspace. A new panel is added for each unique table key. 

![](/images/data_vis/wandb_demo_logged_sample_table.png)

In this example, `my_table`, is logged under the key `"Table Name"`.

## 3. Compare across model versions

Log sample tables from multiple W&B Runs and compare results in the project workspace. In this [example workspace](https://wandb.ai/carey/table-test?workspace=user-carey), we show how to combine rows from multiple different versions in the same table.

![](/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif)

Use the table filter, sort, and grouping features to explore and evaluate model results.

![](/images/data_vis/wandb_demo_filter_on_a_table.png)
