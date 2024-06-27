---
displayed_sidebar: default
---

# W&B Line Plots Point aggregation

Use point aggregation methods within W&B Line Plots to improve data visualization accuracy and performance. There are two types of point aggregation modes: [Random sampling](#random-sampling) and [Full fidelity](#full-fidelity).

:::note
Workspaces use **Random sampling** mode by default. Switching to Full fidelity applies the chart setting per user.
:::
## Random sampling
For performance reasons, when over 1500 points were chosen for a line plot metric, the point aggregation method returns 1500 randomly sampled points. Each metric is sampled separately. Only steps where the metric are actually logged are considered. Because random sampling samples non-deterministically, this method sometimes excluded important outliers or spikes.

### Example: Accessing run history

To access the complete history of metrics logged during a run, you can use the [W&B Run API](../../../../../ref/python/public-api/run.md). The following example demonstrates how to retrieve and process the loss values from a specific run:

```python
# Initialize the W&B API
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# Retrieve the history of the 'Loss' metric
history = run.scan_history(keys=["Loss"])

# Extract the loss values from the history
losses = [row["Loss"] for row in history]
```

## Full fidelity

The full fidelity point aggregation method replaces random sampling with an averaging approach that maintains the integrity of critical visual insights, such as outliers and spikes. Full fidelity mode guarantees the inclusion of minimum and maximum values within each bucket on your chart, allowing for high-detail zoom capabilities.

Some key benfits of full fidelity mode include:
* Accurate Data Representation: Ensures all critical outlier spikes are displayed.
* High-Density Visualization: Maintains full data resolution beyond the 1,500 point limit.
* Enhanced Zoom: Users can zoom into data points without losing detail due to downsampling.

### Enable Full fidelity mode
1. Navigate to your workspace settings or panel settings.
2. Select the Runs tab.
3. Under **Point aggregation method**, choose **Full fidelity**.



:::info Line Plot Grouping or Expressions
W&B downsamples points with buckets if you use Line Plot Grouping or Expressions on runs that have non-aligned x-axis values. The x-axis is divided into 200 segments, and points within each segment are averaged. These averages represent the metric values when grouping or combining metrics.
:::

:::caution Active feature development
Applying Grouping or Expressions will revert to Random sampling instead of Full fidelity. We are actively working on achieving full feature parity with the Run Plots settings for Full fidelity mode, including enabling Grouping and Custom Expressions, while also optimizing performance. For now, panels with grouping or expressions will use Random sampling. This feature is available early because it was highly requested and provided value to users, even though improvements are still ongoing. Please reachout to support@wandb.com if you have any issues. 
:::