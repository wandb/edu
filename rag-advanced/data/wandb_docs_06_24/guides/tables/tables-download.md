---
description: How to export data from tables.
displayed_sidebar: default
---

# Export Table Data
Like all W&B Artifacts, Tables can be converted into pandas dataframes for easy data exporting. 

## Convert `table` to `artifact`
First, you'll need to convert the table to an artifact. The easiest way to do this using `artifact.get(table, "table_name")`:

```python
# Create and log a new table.
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# Retrieve the created table using the artifact you created.
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## Convert `artifact` to Dataframe
Then, convert the table into a dataframe:

```python
# Following from the last code example:
df = table.get_dataframe()
```

## Export Data
Now you can export using any method dataframe supports:

```python
# Converting the table data to .csv
df.to_csv("example.csv", encoding="utf-8")
```

# Next Steps
- Check out the [reference documentation](../artifacts/construct-an-artifact.md) on `artifacts`.
- Go through our [Tables Walktrough](../tables/tables-walkthrough.md) guide.
- Check out the [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) reference docs.