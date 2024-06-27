---
slug: /guides/app/features/panels/query-panel
title: Query panels
description: >-
  Some features on this page are in beta, hidden behind a feature flag. Add
  `weave-plot` to your bio on your profile page to unlock all related features.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info
Looking for W&B Weave? W&B's suite of tools for Generative AI application building? Find the docs for weave here: [wandb.me/weave](https://wandb.github.io/weave/).
:::

Use query panels to query and interactively visualize your data.

![](/images/weave/pretty_panel.png)

<!-- :::tip
See [this report](http://wandb.me/keras-xla-benchmark) to see how this team used Weave Panels to visualize their benchmarks.
::: -->

## Create a query panel

Add a query to your workspace or within a report.

<Tabs
  defaultValue="workspace"
  values={[
    {label: 'Project workspace', value: 'workspace'},
    {label: 'W&B Report', value: 'report'},
  ]}>
  <TabItem value="report">

1. Navigate to your project's workspace. 
2. In the upper right hand corner, click `Add panel`.
3. From the dropdown, select `Query panel`.
![](/images/weave/add_weave_panel_workspace.png)


  </TabItem>
  <TabItem value="workspace">


1. Within your report, type and select `/Query panel`
![](/images/weave/add_weave_panel_report_1.png)

Alternatively, you can associated a query with a set of runs with:
1. Within your report, type and select `/Panel grid`.
2. Click the `Add panel` button.
3. From the dropdown, select `Query panel`.
<!-- ![](/images/weave/add_weave_panel_report_2.png) -->


  </TabItem>
</Tabs>
  

## Query components

### Expressions

Use query expressions to query your data stored in W&B such as runs, artifacts, models, tables, and more. 

#### Example: Query a table
Suppose you want to query a W&B Table. In your training code you log a table called `"cifar10_sample_table"`:

```python
import wandb
wandb.log({"cifar10_sample_table":<MY_TABLE>})
```

Within the query panel you can query your table with:
```python
runs.summary["cifar10_sample_table"]
```
![](/images/weave/basic_weave_expression.png)

Breaking this down:

* `runs` is a variable automatically injected in Query Panel Expressions when the Query Panel is in a Workspace. Its "value" is the list of runs which are visible for that particular Workspace. [Read about the different attributes available within a run here](../../../../track/public-api-guide.md#understanding-the-different-attributes).
* `summary` is an op which returns the Summary object for a Run. Note: ops are "mapped", meaning this op is applied to each Run in the list, resulting in a list of Summary objects.
* `["cifar10_sample_table"]` is a Pick op (denoted with brackets), with a parameter of "predictions". Since Summary objects act like dictionaries or maps, this operation "picks" the "predictions" field off of each Summary object.

To learn how to write your own queries interactively, see [this report](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr).

### Configurations

Select the gear icon on the upper left corner of the panel to expand the query configuration. This allows the user to configure the type of panel and the parameters for the result panel.

![](/images/weave/weave_panel_config.png)

### Result panels

Finally, the query result panel renders the result of the query expression, using the selected query panel, configured by the configuration to display the data in an interactive form. The following images shows a Table and a Plot of the same data.

![](/images/weave/result_panel_table.png)

![](/images/weave/result_panel_plot.png)

## Basic operations
The following common operations you can make within your query panels.
### Sort
Sort from the column options:
![](/images/weave/weave_sort.png)

### Filter
You can either filter directly in the query or using the filter button in the top left corner (second image)
![](/images/weave/weave_filter_1.png)
![](/images/weave/weave_filter_2.png)

### Map
Map operations iterate over lists and apply a function to each element in the data. You can do this directly with a panel query  or by inserting a new column from the column options.
![](/images/weave/weave_map.png)
![](/images/weave/weave_map.gif)

### Groupby
You can groupby using a query or from the column options.
![](/images/weave/weave_groupby.png)
![](/images/weave/weave_groupby.gif)

### Concat
The concat operation allows you to concatenate 2 tables and concatenate or join from the panel settings
![](/images/weave/weave_concat.gif)

### Join
It is also possible to join tables directly in the query. Consider the following query expression:
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
![](/images/weave/weave_join.png)

The table on the left is generated from:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
The table in the right is generated from:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
Where:
* `(row) => row["Label"]` are selectors for each table, determining which column to join on
* `"Table1"` and `"Table2"` are the names of each table when joined
* `true` and `false` are for left and right inner/outer join settings


## Runs object
Use query panels to access the `runs` object. Run objects store records of your experiments. You can find more details about it in [this section](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object) of the report but, as quick overview, `runs` object has available:
* `summary`: A dictionary of information that summarizes the run's results. This can be scalars like accuracy and loss, or large files. By default, `wandb.log()` sets the summary to the final value of a logged time series. You can set the contents of the summary directly. Think of the summary as the run's outputs.
* `history`: A list of dictionaries meant to store values that change while the model is training such as loss. The command `wandb.log()` appends to this object.
* `config`: A dictionary of the run's configuration information, such as the hyperparameters for a training run or the preprocessing methods for a run that creates a dataset Artifact. Think of these as the run's "inputs"
![](/images/weave/weave_runs_object.png)

## Access Artifacts

Artifacts are a core concept in W&B. They are a versioned, named collection of files and directories. Use Artifacts to track model weights, datasets, and any other file or directory. Artifacts are stored in W&B and can be downloaded or used in other runs. You can find more details and examples in [this section](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts) of the report. Artifacts are normally accessed from the `project` object:
* `project.artifactVersion()`: returns the specific artifact version for a given name and version within a project
* `project.artifact("")`: returns the artifact for a given name within a project. You can then use `.versions` to get a list of all versions of this artifact
* `project.artifactType()`: returns the `artifactType` for a given name within a project. You can then use `.artifacts` to get a list of all artifacts with this type
* `project.artifactTypes`: returns a list of all artifact types under the project
![](/images/weave/weave_artifacts.png)





