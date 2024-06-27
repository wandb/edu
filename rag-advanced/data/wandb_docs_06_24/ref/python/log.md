# log

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1665-L1877' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Log a dictionary of data to the current run's history.

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

Use `wandb.log` to log data from runs, such as scalars, images, video,
histograms, plots, and tables.

See our [guides to logging](https://docs.wandb.ai/guides/track/log) for
live examples, code snippets, best practices, and more.

The most basic usage is `wandb.log({"train-loss": 0.5, "accuracy": 0.9})`.
This will save the loss and accuracy to the run's history and update
the summary values for these metrics.

Visualize logged data in the workspace at [wandb.ai](https://wandb.ai),
or locally on a [self-hosted instance](https://docs.wandb.ai/guides/hosting)
of the W&B app, or export data to visualize and explore locally, e.g. in
Jupyter notebooks, with [our API](https://docs.wandb.ai/guides/track/public-api-guide).

In the UI, summary values show up in the run table to compare single values across runs.
Summary values can also be set directly with `wandb.run.summary["key"] = value`.

Logged values don't have to be scalars. Logging any wandb object is supported.
For example `wandb.log({"example": wandb.Image("myimage.jpg")})` will log an
example image which will be displayed nicely in the W&B UI.
See the [reference documentation](https://docs.wandb.com/ref/python/data-types)
for all of the different supported types or check out our
[guides to logging](https://docs.wandb.ai/guides/track/log) for examples,
from 3D molecular structures and segmentation masks to PR curves and histograms.
`wandb.Table`s can be used to logged structured data. See our
[guide to logging tables](https://docs.wandb.ai/guides/data-vis/log-tables)
for details.

Logging nested metrics is encouraged and is supported in the W&B UI.
If you log with a nested dictionary like `wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`, the metrics will be organized into
`train` and `val` sections in the W&B UI.

wandb keeps track of a global step, which by default increments with each
call to `wandb.log`, so logging related metrics together is encouraged.
If it's inconvenient to log related metrics together
calling `wandb.log({"train-loss": 0.5}, commit=False)` and then
`wandb.log({"accuracy": 0.9})` is equivalent to calling
`wandb.log({"train-loss": 0.5, "accuracy": 0.9})`.

`wandb.log` is not intended to be called more than a few times per second.
If you want to log more frequently than that it's better to aggregate
the data on the client side or you may get degraded performance.

| Arguments |  |
| :--- | :--- |
|  `data` |  (dict, optional) A dict of serializable python objects i.e `str`, `ints`, `floats`, `Tensors`, `dicts`, or any of the `wandb.data_types`. |
|  `commit` |  (boolean, optional) Save the metrics dict to the wandb server and increment the step. If false `wandb.log` just updates the current metrics dict with the data argument and metrics won't be saved until `wandb.log` is called with `commit=True`. |
|  `step` |  (integer, optional) The global step in processing. This persists any non-committed earlier steps but defaults to not committing the specified step. |
|  `sync` |  (boolean, True) This argument is deprecated and currently doesn't change the behaviour of `wandb.log`. |

#### Examples:

For more and more detailed examples, see
[our guides to logging](https://docs.wandb.com/guides/track/log).

### Basic usage

<!--yeadoc-test:init-and-log-basic-->


```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### Incremental logging

<!--yeadoc-test:init-and-log-incremental-->


```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# Somewhere else when I'm ready to report this step:
run.log({"accuracy": 0.8})
```

### Histogram

<!--yeadoc-test:init-and-log-histogram-->


```python
import numpy as np
import wandb

# sample gradients at random from normal distribution
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### Image from numpy

<!--yeadoc-test:init-and-log-image-numpy-->


```python
import numpy as np
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
    image = wandb.Image(pixels, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### Image from PIL

<!--yeadoc-test:init-and-log-image-pillow-->


```python
import numpy as np
from PIL import Image as PILImage
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(pixels, mode="RGB")
    image = wandb.Image(pil_image, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### Video from numpy

<!--yeadoc-test:init-and-log-video-numpy-->


```python
import numpy as np
import wandb

run = wandb.init()
# axes are (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib Plot

<!--yeadoc-test:init-and-log-matplotlib-->


```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

run = wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y)  # plot y = x^2
run.log({"chart": fig})
```

### PR Curve

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3D Object

```python
import wandb

run = wandb.init()
run.log(
    {
        "generated_samples": [
            wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb")),
        ]
    }
)
```

| Raises |  |
| :--- | :--- |
|  `wandb.Error` |  if called before `wandb.init` |
|  `ValueError` |  if invalid data is passed |
