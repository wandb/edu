---
description: Visualize metrics, customize axes, and compare categorical data as bars.
displayed_sidebar: default
---

# Bar Plot

A bar plot presents categorical data with rectangular bars which can be plotted vertically or horizontally. Bar plots show up by default with **wandb.log()** when all logged values are of length one.

![Plotting Box and horizontal Bar plots in W&B](/images/app_ui/bar_plot.png)

Customize with chart settings to limit max runs to show, group runs by any config and rename labels.

![](/images/app_ui/bar_plot_custom.png)

### Customize Bar Plots

You can also create **Box** or **Violin** Plots to combine many summary statistics into one chart type**.**

1. Group runs via runs table.
2. Click 'Add panel' in the workspace.
3. Add a standard 'Bar Chart' and select the metric to plot.
4. Under the 'Grouping' tab, pick 'box plot' or 'Violin', etc. to plot either of these styles.

![Customize Bar Plots](@site/static/images/app_ui/bar_plots.gif)
