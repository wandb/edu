---
description: A playground for exploring run data with interactive visualizations
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Workspaces

W&B workspace is your personal sandbox to customize charts and explore model results. A W&B workspace consists of *Tables* and *Panel sections*: 

* **Tables**: All runs logged to your project are listed in the project's table. Turn on and off runs, change colors, and expand the table to see notes, config, and summary metrics for each run.
* **Panel sections**: A section that contains one or more [panels](../features/panels/intro.md). Create new panels, organize them, and export to reports to save snapshots of your workspace.

![](/images/app_ui/workspace_table_and_panels.png)

## Workspace types
There are two main workspace categories: **Personal workspaces** and **Saved views**. 

* **Personal workspaces:**  A customizable workspace for in-depth analysis of models and data visualizations. Only the owner of the workspace can edit and save changes. Teammates can view a personal workspace but teammates can not make changes to someone else's personal workspace. 
* **Saved views:** Saved views are collaborative snapshots of a workspace. Anyone on your team can view, edit, and save changes to saved workspace views. Use saved workspace views for reviewing and discussing experiments, runs, and more.

The proceeding image shows multiple personal workspaces created by Cécile-parker's teammates. In this project, there are no saved views:
![](/images/app_ui/Menu_No_views.jpg)

## Saved workspace views
Improve team collaboration with tailored workspace views. Create Saved Views to organize your preferred setup of charts and data. 

### Create a new saved workspace view

1. Navigate to a personal workspace or a saved view.
2. Make edits to the workspace.
3. Click on the meatball menu (three horizontal dots) at the top right corner of your workspace. Click on **Save as a new view**.

New saved views appear in the workspace navigation menu.

![](/images/app_ui/Menu_Views.jpg)



### Update a saved workspace view 
Saved changes overwrite the previous state of the saved view. Unsaved changes are not retained. To update a saved workspace view in W&B:

1. Navigate to a saved view.
2. Make the desired changes to your charts and data within the workspace.
3. Click the **Save** button to confirm your changes. 

:::info
A confirmation dialog appears when you save your updates to a workspace view. If you prefer not to see this prompt in the future, select the option **Do not show this modal next time** before confirming your save.
:::

### Delete a saved workspace view
Remove saved views that are no longer needed.

1. Navigate to the saved view you want to remove.
2. Click on the hamburger menu (three horizontal lines) at the top right of the view.
3. Choose **Delete view**.
4. Confirm the deletion to remove the view from your workspace menu.

![](/images/app_ui/Deleting.gif)

### Share a workspace view
Share your customized workspace with your team by sharing the workspace URL directly. All users with access to the workspace project can see the saved Views of that workspace.

# Programmatic workspace

[`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) is a Python library for programmatically working with [Weights & Biases](https://wandb.ai/) workspaces and reports.

## Creating a workspace programmatically

You can define a workspace programmatically by defining the workspace's properties, such as its name, associated entity and project, and the sections it should contain.

### Features:

- **Programmatic workspace creation:**
  - Define and create workspaces with specific configurations.
  - Set panel layouts, colors, and section orders.
- **Workspace customization:**
  - Configure workspace settings like default x-axis, section order, and collapse states.
  - Add and customize panels within sections to organize workspace views.
- **Editing existing workspace `saved views`:**
  - Load and modify existing workspaces using a URL.
  - Save changes to existing workspaces or save as new views.
- **Run filtering and grouping:**
  - Filter, group, and sort runs programmatically using simple expressions.
  - Customize run appearance with settings like colors and visibility.
- **Cross-workspace integration:**
  - Copy views from one workspace to another for seamless integration and reuse.

## Example workspace API workflows
See [`wandb-workspace examples`](https://github.com/wandb/wandb-workspaces/tree/main/examples/workspaces) for more comprehensive examples using the [Weights & Biases](https://wandb.ai/) workspace API. 

### Define and save a workspace view programmatically
```python
wandb_workspaces.workspaces as ws

workspace = ws.Workspace(entity="your-entity", project="your-project", views=[...])
workspace.save()
```

### Edit an existing view
```python
existing_workspace = ws.Workspace.from_url("workspace-url")
existing_workspace.views[0] = ws.View(name="my-new-view", sections=[...])
existing_workspace.save()
```

### Copy a workspace `saved view` to another workspace

```python
old_workspace = ws.Workspace.from_url("old-workspace-url")
old_workspace_view = old_workspace.views[0]
new_workspace = ws.Workspace(entity="new-entity", project="new-project", views=[old_workspace_view])

new_workspace.save()
```