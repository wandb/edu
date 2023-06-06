---
description: Explore how to use W&B Tables with this 5 minute Quickstart.
---

# Tables Quickstart

The following Quickstart demonstrates how to log data tables, visualize data, and query data.


Select the button below to try a PyTorch Quickstart example project on MNIST data. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/tables-quickstart)

## 1. Log a table

Follow the procedure outlined below to log a Table with W&B:
1. Initialize a W&B Run with [`wandb.init()`](../../ref/python/init.md). 
2. Create a [`wandb.Table()`](../../ref/python/data-types/table.md) object instance. Pass the name of the columns in your table along with the data for the `columns` and `data` parameters, respectively.  
3. Log the table with [`run.log()`](../../ref/python/log.md) as a key-value pair. Provide a name for your table for the key, and pass the object instance of `wandb.Table` as the value.

```python
run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

You can optionally pass in a Pandas DataFrame to `wandb.Table()` Class. For more information on supported data types, see the [`wandb.Table`](../../ref/python/data-types/table.md) in the W&B API Reference Guide.

## 2. Visualize tables in the workspace

View the resulting table in your workspace. Navigate to the W&B App and select the name of your Run in your Project workspace. A new panel is added for each unique table key. 

![](/images/data_vis/wandb_demo_logged_sample_table.png)

In this example, `my_table`, is logged under the key `"Table Name"`.

## 3. Compare across model versions

Log sample tables from multiple W&B Runs and compare results in the project workspace. In this [example workspace](https://wandb.ai/carey/table-test?workspace=user-carey), we show how to combine rows from multiple different versions in the same table.

![](/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif)

Use the table filter, sort, and grouping features to explore and evaluate model results.

![](/images/data_vis/wandb_demo_filter_on_a_table.png)
