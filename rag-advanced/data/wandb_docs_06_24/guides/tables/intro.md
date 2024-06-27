---
slug: /guides/tables
description: Iterate on datasets and understand model predictions
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Visualize your data

<CTAButtons productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb"/>

Use W&B Tables to visualize and query tabular data. For example:

* Compare how different models perform on the same test set
* Identify patterns in your data
* Look at sample model predictions visually
* Query to find commonly misclassified examples


![](/images/data_vis/tables_sample_predictions.png)
The above image shows a table with semantic segmentation and custom metrics. View this table here in this [sample project from the W&B ML Course](https://wandb.ai/av-team/mlops-course-001).

## How it works

A Table is a two-dimensional grid of data where each column has a single type of data. Tables support primitive and numeric types, as well as nested lists, dictionaries, and rich media types. 

## Log a Table

Log a table with a few lines of code:

- [`wandb.init()`](../../ref/python/init.md): Create a [run](../runs/intro.md) to track results.
- [`wandb.Table()`](../../ref/python/data-types/table.md): Create a new table object.
  - `columns`: Set the column names.
  - `data`: Set the contents of the table.
- [`run.log()`](../../ref/python/log.md): Log the table to save it to W&B.

```python showLineNumbers
import wandb

run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## How to get started
* [Quickstart](./tables-walkthrough.md): Learn to log data tables, visualize data, and query data.
* [Tables Gallery](./tables-gallery.md): See example use cases for Tables.

