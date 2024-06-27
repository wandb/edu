---
slug: /guides/app/features/panels/line-plot
description: Visualize metrics, customize axes, and compare multiple lines on the same plot
displayed_sidebar: default
---

# Line Plot

Line plots show up by default when you plot metrics over time with **wandb.log()**. Customize with chart settings to compare multiple lines on the same plot, calculate custom axes, and rename labels.

![](/images/app_ui/line_plot_example.png)

## Settings

**Data**

* **X axis**: Select default x-axes including Step and Relative Time, or select a custom x-axis. If you'd like to use a custom x-axis, make sure it's logged in the same call to `wandb.log()` that you use to log the y-axis.
  * **Relative Time (Wall)** is clock time since the process started, so if you started a run and resumed it a day later and logged something that would be plotted a 24hrs.
  * **Relative Time (Process)** is time inside the running process, so if you started a run and ran for 10 seconds and resumed a day later that point would be plotted at 10s
  * **Wall Time** is minutes elapsed since the start of the first run on the graph
  * **Step** increments by default each time `wandb.log()` is called, and is supposed to reflect the number of training steps you've logged from your model
* **Y axes**: Select y-axes from the logged values, including metrics and hyperparameters that change over time.
* **Min, max, and log scale**: Minimum, maximum, and log scale settings for x axis and y axis in line plots
* **Smoothing and exclude outliers**: Change the smoothing on the line plot or rescale to exclude outliers from the default plot min and max scale
* **Max runs to show**: Show more lines on the line plot at once by increasing this number, which defaults to 10 runs. You'll see the message "Showing first 10 runs" on the top of the chart if there are more than 10 runs available but the chart is constraining the number visible.
* **Chart type**: Change between a line plot, an area plot, and a percentage area plot

**X Axis Settings**
The x-axis can be set at the graph level, as well as globally for the project page or report page. Here's what the global settings look like:

![](/images/app_ui/x_axis_global_settings.png)

:::info
Pick **multiple y-axes** in the line plot settings to compare different metrics on the same chart, like accuracy and validation accuracy for example.
:::

**Grouping**

* Turn on grouping to see settings for visualizing averaged values.
* **Group key**: Select a column, and all the runs with the same value in that column will be grouped together.
* **Agg**: Aggregation— the value of the line on the graph. The options are mean, median, min, and max of the group.
* **Range**: Switch the behavior for the shaded area behind the grouped curve. None means there is no shaded area. Min/Max shows a shaded region that covers the whole range of points in the group. Std Dev shows the standard deviation of values in the group. Std Err shows the standard error as the shaded area.
* **Sampled runs**: If you have hundreds of runs selected, we default to only sampling the first 100. You can select to have all your runs included in the grouping calculation, but it might slow things down in the UI.

**Legend**

* **Title**: Add a custom title for line plot, which shows up at the top of the chart
* **X-Axis title**: Add a custom title for the x-axis of the line plot, which shows up in the lower right corner of the chart.
* **Y-Axis title**: Add a custom title for the y-axis of the line plot, which shows up in the upper left corner of the chart.
* **Legend**: Select field that you want to see in the legend of the plot for each line. You could, for example, show the name of the run and the learning rate.
* **Legend template**: Fully customizable, this powerful template allows you to specify exactly what text and variables you want to show up in the template at the top of the line plot as well as the legend that appears when you hover your mouse over the plot.

![Editing the line plot legend to show hyperparameters](/images/app_ui/legend.png)

**Expressions**

* **Y Axis Expressions**: Add calculated metrics to your graph. You can use any of the logged metrics as well as configuration values like hyperparameters to calculate custom lines.
* **X Axis Expressions**: Rescale the x-axis to use calculated values using custom expressions. Useful variables include\*\*\_step\*\* for the default x-axis, and the syntax for referencing summary values is `${summary:value}`

## Visualize average values on a plot

If you have several different experiments and you'd like to see the average of their values on a plot, you can use the Grouping feature in the table. Click "Group" above the run table and select "All" to show averaged values in your graphs.

Here is what the graph looks like before averaging:

![](/images/app_ui/demo_precision_lines.png)

Here I have grouped the lines to see the average value across runs.

![](/images/app_ui/demo_average_precision_lines.png)

## Visualize NaN value on a plot

You can also plot `NaN` values including PyTorch tensors on a line plot with `wandb.log`. For example:

```python
wandb.log({"test": [..., float("nan"), ...]})
```

![](/images/app_ui/visualize_nan.png)

## Compare two metrics on one chart

Click on a run to go to the run page. Here's an [example run](https://app.wandb.ai/stacey/estuary/runs/9qha4fuu?workspace=user-carey) from Stacey's Estuary project. The auto-generated charts show single metrics.


![](@site/static/images/app_ui/visualization_add.png)

Click **the plus sign** at the top right of the page, and select the **Line Plot**.

![](https://downloads.intercomcdn.com/i/o/142936481/d0648728180887c52ab46549/image.png)

In the **Y variables** field, select a few metrics you'd like to compare. They'll show up together on the line graph.

![](https://downloads.intercomcdn.com/i/o/146033909/899fc05e30795a1d7699dc82/Screen+Shot+2019-09-04+at+9.10.52+AM.png)

## Changing the color of the line plots

Sometimes the default color of runs is not helpful for comparison. To help overcome this, wandb provides two instances with which one can manually change the colors.

### From the run table

Each run is given a random color by default upon initialization.

![Random colors given to runs](/images/app_ui/line_plots_run_table_random_colors.png)

Upon clicking any of the colors, a color palette appears from which we can manually choose the color we want.

![The color palette](/images/app_ui/line_plots_run_table_color_palette.png)


### From the chart legend settings

One can also change the color of the runs from the chart legend settings.


![](/images/app_ui/plot_style_line_plot_legend.png)

## Visualize on different x axes

If you'd like to see the absolute time that an experiment has taken, or see what day an experiment ran, you can switch the x axis. Here's an example of switching from steps to relative time and then to wall time.

![](/images/app_ui/howto_use_relative_time_or_wall_time.gif)

## Area plots

In the line plot settings, in the advanced tab, click on different plot styles to get an area plot or a percentage area plot.

![](/images/app_ui/line_plots_area_plots.gif)

## Zoom

Click and drag a rectangle to zoom vertically and horizontally at the same time. This changes the x-axis and y-axis zoom.

![](/images/app_ui/line_plots_zoom.gif)

## Hide chart legend

Turn off the legend in the line plot with this simple toggle:

![](/images/app_ui/demo_hide_legend.gif)
