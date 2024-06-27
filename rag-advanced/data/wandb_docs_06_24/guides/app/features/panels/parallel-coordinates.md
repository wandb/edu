---
description: Compare results across machine learning experiments
displayed_sidebar: default
---

# Parallel Coordinates

Parallel coordinates charts summarize the relationship between large numbers of hyperparameters and model metrics at a glance.

![](/images/app_ui/parallel_coordinates.gif)

* **Axes**: Different hyperparameters from [`wandb.config`](../../../../guides/track/config.md) and metrics from [`wandb.log`](../../../../guides/track/log/intro.md).
* **Lines**: Each line represents a single run. Mouse over a line to see a tooltip with details about the run. All lines that match the current filters will be shown, but if you turn off the eye, lines will be grayed out.

## Panel Settings

Configure these features in the panel settings— click the edit button in the upper right corner of the panel.

* **Tooltip**: On hover, a legend shows up with info on each run
* **Titles**: Edit the axis titles to be more readable
* **Gradient**: Customize the gradient to be any color range you like
* **Log scale**: Each axis can be set to view on a log scale independently
* **Flip axis**: Switch the axis direction— this is useful when you have both accuracy and loss as columns

[See it live →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)
