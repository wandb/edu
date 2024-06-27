---
description: >-
  Create a W&B Report with the App UI or programmatically with the Weights &
  Biases SDK.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a report

<head>
  <title>Create a W&B Report</title>
</head>

Create a report interactively with the W&B App UI or programmatically with the W&B Python SDK.

:::info
See this [Google Colab for an example](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb).
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Report tab', value: 'reporttab'},
    {label: 'W&B Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

1. Navigate to your project workspace in the W&B App.
2. Click **Create report** in the upper right corner of your workspace.

![](/images/reports/create_a_report_button.png)

3. A modal will appear. Select the charts you would like to start with. You can add or delete charts later from the report interface.

![](/images/reports/create_a_report_modal.png)

4. Select the **Filter run sets** option to prevent new runs from being added to your report. You can toggle this option on or off. Once you click **Create report,** a draft report will be available in the report tab to continue working on.


  </TabItem>
  <TabItem value="reporttab">

1. Navigate to your project workspace in the W&B App.
2. Select to the **Reports** tab (clipboard image) in your project.
3. Select the **Create Report** button on the report page. 

![](/images/reports/create_report_button.png)
  </TabItem>
  <TabItem value="sdk">

Create a report programmatically with the `wandb` library.

```python
import wandb
import wandb_workspaces.reports.v2 as wr
```

Create a report instance with the Report Class Public API ([`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports)). Specify a name for the project.

```python
report = wr.Report(project="report_standard")
```

Reports are not uploaded to the W&B server until you call the .`save()` method:

```python
report.save()
```

For information on how to edit a report interactively with the App UI or programmatically, see [Edit a report](https://docs.wandb.ai/guides/reports/edit-a-report).
  </TabItem>
</Tabs>
