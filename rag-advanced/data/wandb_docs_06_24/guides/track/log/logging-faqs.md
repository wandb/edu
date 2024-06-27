---
description: >-
  Answers to frequently asked questions about tracking data from machine
  learning experiments with W&B Experiments.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Logging FAQs

<head>
  <title>Frequently Asked Questions About Logging Data from Experiments</title>
</head>

### How can I organize my logged charts and media in the W&B UI?

We treat `/` as a separator for organizing logged panels in the W&B UI. By default, the component of the logged item's name before a `/` is used to define a group of panel called a "Panel Section".

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

In the [Workspace](../../app/pages/workspaces.md) settings, you can change whether panels are grouped by just the first component or by all components separated by `/`.

### How can I compare images or media across epochs or steps?

Each time you log images from a step, we save them to show in the UI. Expand the image panel, and use the step slider to look at images from different steps. This makes it easy to compare how a model's output changes during training.

### What if I want to log some metrics on batches and some metrics only on epochs?

If you'd like to log certain metrics in every batch and standardize plots, you can log x axis values that you want to plot with your metrics. Then in the custom plots, click edit and select a custom x-axis.

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```

### How do I log a list of values?

<!-- Logging lists directly is not supported. Instead, list-like collections of numerical data are converted to [histograms](../../../ref/python/data-types/histogram.md). To log all of the entries in a list, give a name to each entry in the list and use those names as keys in a dictionary, as below. -->

<Tabs
  defaultValue="dictionary"
  values={[
    {label: 'Using a dictionary', value: 'dictionary'},
    {label: 'As a histogram', value: 'histogram'},
  ]}>
  <TabItem value="dictionary">

```python
wandb.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
  </TabItem>
  <TabItem value="histogram">

```python
wandb.log({"losses": wandb.Histogram(losses)})  # converts losses to a histogram
```
  </TabItem>
</Tabs>

### How do I plot multiple lines on a plot with a legend?

Multi-line custom chart can be created by using `wandb.plot.line_series()`. You'll need to navigate to the [project page](../../app/pages/project-page.md) to see the line chart. To add a legend to the plot, pass the keys argument within `wandb.plot.line_series()`. For example:

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

You can find more information about Multi-line plots [here](../../track/log/plots.md#basic-charts) under the Multi-line tab.

### How do I add Plotly/Bokeh Charts into Tables?

Adding Plotly/Bokeh figures to Tables directly is not yet supported. Instead, write the figure to HTML and then add the HTML to the Table. Examples with interactive Plotly and Bokeh charts below.

<Tabs
  defaultValue="plotly"
  values={[
    {label: 'Using Plotly', value: 'plotly'},
    {label: 'Using Bokeh', value: 'bokeh'},
  ]}>
  <TabItem value="plotly">

```python
import wandb
import plotly.express as px

# Initialize a new run
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# Create a table
table = wandb.Table(columns=["plotly_figure"])

# Create path for Plotly figure
path_to_plotly_html = "./plotly_figure.html"

# Example Plotly figure
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Write Plotly figure to HTML
# Setting auto_play to False prevents animated Plotly
# charts from playing in the table automatically
fig.write_html(path_to_plotly_html, auto_play=False)

# Add Plotly figure as HTML file into Table
table.add_data(wandb.Html(path_to_plotly_html))

# Log Table
run.log({"test_table": table})
wandb.finish()
```

  </TabItem>
  <TabItem value="bokeh">

```python
from scipy.signal import spectrogram
import holoviews as hv
import panel as pn
from scipy.io import wavfile
import numpy as np
from bokeh.resources import INLINE

hv.extension("bokeh", logo=False)
import wandb


def save_audio_with_bokeh_plot_to_html(audio_path, html_file_name):
    sr, wav_data = wavfile.read(audio_path)
    duration = len(wav_data) / sr
    f, t, sxx = spectrogram(wav_data, sr)
    spec_gram = hv.Image((t, f, np.log10(sxx)), ["Time (s)", "Frequency (hz)"]).opts(
        width=500, height=150, labelled=[]
    )
    audio = pn.pane.Audio(wav_data, sample_rate=sr, name="Audio", throttle=500)
    slider = pn.widgets.FloatSlider(end=duration, visible=False)
    line = hv.VLine(0).opts(color="white")
    slider.jslink(audio, value="time", bidirectional=True)
    slider.jslink(line, value="glyph.location")
    combined = pn.Row(audio, spec_gram * line, slider).save(html_file_name)


html_file_name = "audio_with_plot.html"
audio_path = "hello.wav"
save_audio_with_bokeh_plot_to_html(audio_path, html_file_name)

wandb_html = wandb.Html(html_file_name)
run = wandb.init(project="audio_test")
my_table = wandb.Table(columns=["audio_with_plot"], data=[[wandb_html], [wandb_html]])
run.log({"audio_table": my_table})
run.finish()
```
  </TabItem>
</Tabs>

### Why is nothing showing up in my graphs?

If you're seeing "No visualization data logged yet" that means that we haven't gotten the first `wandb.log` call from your script yet. This could be because your run takes a long time to finish a step. If you're logging at the end of each epoch, you could log a few times per epoch to see data stream in more quickly.

### Why is the same metric appearing more than once?

If you're logging different types of data under the same key, we have to split them out in our database. This means you'll see multiple entries of the same metric name in a dropdown in the UI. The types we group by are `number`, `string`, `bool`, `other` (mostly arrays), and any `wandb` data type (`Histogram`, `Image`, etc). Send only one type to each key to avoid this behavior.

We store metrics in a case-insensitive fashion, so make sure you don't have two metrics with the same name like `"My-Metric"` and `"my-metric"`.

### How can I access the data logged to my runs directly and programmatically?

The history object is used to track metrics logged by `wandb.log`. Using [our API](../public-api-guide.md), you can access the history object via `run.history()`.

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```

### What happens when I log millions of steps to W&B? How is that rendered in the browser?

The more points you send us, the longer it will take to load your graphs in the UI. If you have more than 1000 points on a line, we sample down to 1000 points on the backend before we send your browser the data. This sampling is nondeterministic, so if you refresh the page you'll see a different set of sampled points.

**Guidelines**

We recommend that you try to log less than 10,000 points per metric. If you log more than 1 million points in a line, it will take us while to load the page. For more on strategies for reducing logging footprint without sacrificing accuracy, check out [this Colab](http://wandb.me/log-hf-colab). If you have more than 500 columns of config and summary metrics, we'll only show 500 in the table.

### What if I want to integrate W&B into my project, but I don't want to upload any images or media?

W&B can be used even for projects that only log scalars — you specify any files or data you'd like to upload explicitly. Here's [a quick example in PyTorch](http://wandb.me/pytorch-colab) that does not log images.

### What happens if I pass a class attribute into wandb.log()?

It is generally not recommended to pass class attributes into `wandb.log()` as the attribute may change before the network call is made. If you are storing metrics as the attribute of a class, it is recommended to deep copy the attribute to ensure the metric logged matches the value of the attribute at the time that `wandb.log()` was called.

### Why am I seeing fewer data points than I logged?

If you are visualizing your metrics against something other than `Step` on your X-Axis, you might see fewer data points than you expect. This is because we require the metrics to be plotted against one another to be logged at the same `Step` - that is how we keep your metrics synchronized, i.e., we only sample metrics that are logged at the same `Step` while interpolating in between samples.\
\
**Guidelines**\
****\
****We recommend you bundle your metrics into the same `log()` call. If your code looks like this:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

It would be better to log it as:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

Alternatively, you can manually control the step parameter and synchronize your metrics in your own code:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

If the value of `step` is the same in both the calls to `log()`, your metrics will be logged under the same step and be sampled together. Please note that step must be monotonically increasing in each call, otherwise the `step` value is ignored during your call to `log()`.
