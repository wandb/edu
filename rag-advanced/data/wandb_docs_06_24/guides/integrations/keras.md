---
displayed_sidebar: default
---

# Keras

[**Try in a Colab Notebook here →**](http://wandb.me/intro-keras)

## The Weights & Biases Keras Callbacks

We have added three new callbacks for Keras and TensorFlow users, available from `wandb` v0.13.4. For the legacy `WandbCallback` scroll down.


**`WandbMetricsLogger`** : Use this callback for [Experiment Tracking](https://docs.wandb.ai/guides/track). It will log your training and validation metrics along with system metrics to Weights and Biases.

**`WandbModelCheckpoint`** : Use this callback to log your model checkpoints to Weight and Biases [Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning).

**`WandbEvalCallback`**: This base callback will log model predictions to Weights and Biases [Tables](https://docs.wandb.ai/guides/tables) for interactive visualization.

These new callbacks,

* Adhere to Keras design philosophy
* Reduce the cognitive load of using a single callback (`WandbCallback`) for everything,
* Make it easy for Keras users to modify the callback by subclassing it to support their niche use case.

## Experiment Tracking with `WandbMetricsLogger`

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbMetricLogger\_in\_your\_Keras\_workflow.ipynb)

`WandbMetricsLogger` automatically logs Keras' `logs` dictionary that callback methods such as `on_epoch_end`, `on_batch_end` etc, take as an argument.

Using this provides:

* train and validation metrics defined in `model.compile`
* system (CPU/GPU/TPU) metrics
* learning rate (both for a fixed value or a learning rate scheduler)

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Initialize a new W&B run
wandb.init(config={"bs": 12})

# Pass the WandbMetricsLogger to model.fit
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

**`WandbMetricsLogger` Reference**


| Parameter | Description | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | ("epoch", "batch", or int): if "epoch", logs metrics at the end of each epoch. If "batch", logs metrics at the end of each batch. If an int, logs metrics at the end of that many batches. Defaults to "epoch".                                 |
| `initial_global_step` | (int): Use this argument to correctly log the learning rate when you resume training from some initial_epoch, and a learning rate scheduler is used. This can be computed as step_size * initial_step. Defaults to 0. |

## Model Checkpointing using `WandbModelCheckpoint`

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbModelCheckpoint\_in\_your\_Keras\_workflow.ipynb)

Use `WandbModelCheckpoint` callback to save the Keras model (`SavedModel` format) or model weights periodically and uploads them to W&B as a `wandb.Artifact` for model versioning. 

This callback is subclassed from [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/ModelCheckpoint) ,thus the checkpointing logic is taken care of by the parent callback.

This callback provides the following features:

* Save the model that has achieved "best performance" based on the "monitor".
* Save the model at the end of every epoch regardless of the performance.
* Save the model at the end of the epoch or after a fixed number of training batches.
* Save only model weights, or save the whole model.
* Save the model either in SavedModel format or in `.h5` format.

This callback should be used in conjunction with `WandbMetricsLogger`.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Initialize a new W&B run
wandb.init(config={"bs": 12})

# Pass the WandbModelCheckpoint to model.fit
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        WandbModelCheckpoint("models"),
    ],
)
```

**`WandbModelCheckpoint` Reference**

| Parameter | Description | 
| ------------------------- |  ---- | 
| `filepath`   | (str): path to save the mode file.|  
| `monitor`                 | (str): The metric name to monitor.         |
| `verbose`                 | (int): Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action.   |
| `save_best_only`          | (bool): if `save_best_only=True`, it only saves when the model is considered the "best" and the latest best model according to the quantity monitored (`monitor`) will not be overwritten.     |
| `save_weights_only`       | (bool): if True, then only the model's weights will be saved.                                            |
| `mode`                    | ("auto", "min", or "max"): For val\_acc, this should be ‘max’, for val\_loss this should be ‘min’, etc.  |
| `save_weights_only`       | (bool): if True, then only the model's weights will be saved.                                            |
| `save_freq`               | ("epoch" or int): When using ‘epoch’, the callback saves the model after each epoch. When using an integer, the callback saves the model at end of this many batches. Note that when monitoring validation metrics such as `val_acc` or `val_loss`, `save_freq` must be set to "epoch" as those metrics are only available at the end of an epoch. |
| `options`                 | (str): Optional `tf.train.CheckpointOptions` object if `save_weights_only` is true or optional `tf.saved_model.SaveOptions` object if `save_weights_only` is false.    |
| `initial_value_threshold` | (float): Floating point initial "best" value of the metric to be monitored.       |

### How to log checkpoints after N epochs?

By default (`save_freq="epoch"`) the callback creates a checkpoint and uploads it as an artifact after each epoch. If we pass an integer to `save_freq` the checkpoint will be created after that many batches. To checkpoint after `N` epochs, compute the cardinality of the train dataloader and pass it to `save_freq`:

```
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### How to log checkpoints on a TPU Node architecture efficiently?

While checkpointing on TPUs you might encounter `UnimplementedError: File system scheme '[local]' not implemented` error message. This happens because the model directory (`filepath`) must use a cloud storage bucket path (`gs://bucket-name/...`), and this bucket must be accessible from the TPU server. We can however, use the local path for checkpointing which in turn is uploaded as an Artifacts.

```
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## Model Prediction Visualization using `WandbEvalCallback`

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

The `WandbEvalCallback` is an abstract base class to build Keras callbacks primarily for model prediction and, secondarily, dataset visualization.

This abstract callback is agnostic with respect to the dataset and the task. To use this, inherit from this base `WandbEvalCallback` callback class and implement the `add_ground_truth` and `add_model_prediction` methods.

The `WandbEvalCallback` is a utility class that provides helpful methods to:

* create data and prediction `wandb.Table` instances,
* log data and prediction Tables as `wandb.Artifact`
* logs the data table `on_train_begin`
* logs the prediction table `on_epoch_end`

For example, we have implemented `WandbClfEvalCallback` below for an image classification task. This example callback:

* logs the validation data (`data_table`) to W&B,
* performs inference and logs the prediction (`pred_table`) to W&B at the end of every epoch.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


# Implement your model prediction visualization callback
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validation_data, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.x = validation_data[0]
        self.y = validation_data[1]

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(zip(self.x, self.y)):
            self.data_table.add_data(idx, wandb.Image(image), label)

    def add_model_predictions(self, epoch, logs=None):
        preds = self.model.predict(self.x, verbose=0)
        preds = tf.argmax(preds, axis=-1)

        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            )


# ...

# Initialize a new W&B run
wandb.init(config={"hyper": "parameter"})

# Add the Callbacks to Model.fit
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        WandbClfEvalCallback(
            validation_data=(X_test, y_test),
            data_table_columns=["idx", "image", "label"],
            pred_table_columns=["epoch", "idx", "image", "label", "pred"],
        ),
    ],
)
```

:::info
💡 The Tables are logged to the W&B [Artifact page](https://docs.wandb.ai/ref/app/pages/project-page#artifacts-tab) by default and not the [Workspace](https://docs.wandb.ai/ref/app/pages/workspaces) page.
:::

**`WandbEvalCallback` Reference**

| Parameter            | Description                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) List of column names for the `data_table` |
| `pred_table_columns` | (list) List of column names for the `pred_table` |

### How the memory footprint is reduced?

We log the `data_table` to W&B when the `on_train_begin` method is invoked. Once it's uploaded as a W&B Artifact, we get a reference to this table which can be accessed using `data_table_ref` class variable. The `data_table_ref` is a 2D list that can be indexed like `self.data_table_ref[idx][n]`, where `idx` is the row number while `n` is the column number. Let's see the usage in the example below.

### Customize the callback further

You can override the `on_train_begin` or `on_epoch_end` methods to have more fine-grained control. If you want to log the samples after `N` batches, you can implement `on_train_batch_end` method.

:::info
💡 If you are implementing a callback for model prediction visualization by inheriting `WandbEvalCallback` and something needs to be clarified or fixed, please let us know by opening an [issue](https://github.com/wandb/wandb/issues).
:::

## WandbCallback [Legacy]

Use the W&B library [`WandbCallback`](https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback) Class to automatically save all the metrics and the loss values tracked in `model.fit`.

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # code to set up your model in Keras

# Pass the callback to model.fit
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

**Usage examples**

See this one minute, step-by-step video if this is your first time integrating W&B with Keras: [Get Started with Keras and Weights & Biases in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)

For a more detailed video, see [Integrate Weights & Biases with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab\_channel=Weights%26Biases). The notebook example used can be found here: [Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras\_pipeline\_with\_Weights\_and\_Biases.ipynb).

:::info
Try W&B and Keras integration example from the video above in a [colab notebook](http://wandb.me/keras-colab). Or see our [example repo](https://github.com/wandb/examples) for scripts, including a [Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) and the [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) it generates.
:::

The `WandbCallback` class supports a wide variety of logging configuration options: specifying a metric to monitor, tracking of weights and gradients, logging of predictions on training\_data and validation\_data, and more.

Check out [the reference documentation for the `keras.WandbCallback`](../../ref/python/integrations/keras/wandbcallback.md) for full details.

The `WandbCallback` 

* will automatically log history data from any metrics collected by keras: loss and anything passed into `keras_model.compile()`
* will set summary metrics for the run associated with the "best" training step, where "best" is defined by the `monitor` and `mode` attributes. This defaults to the epoch with the minimum `val_loss`. `WandbCallback` will by default save the model associated with the best `epoch`
* can optionally log gradient and parameter histogram
* can optionally save training and validation data for wandb to visualize.

**`WandbCallback` Reference**

| Arguments                  |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) name of metric to monitor. Defaults to `val_loss`.                                                                   |
| `mode`                     | (str) one of {`auto`, `min`, `max`}. `min` - save model when monitor is minimized `max` - save model when monitor is maximized `auto` - try to guess when to save the model (default).                                                                                                                                                |
| `save_model`               | True - save a model when monitor beats all previous epochs False - don't save models                                       |
| `save_graph`               | (boolean) if True save model graph to wandb (default to True).                                                           |
| `save_weights_only`        | (boolean) if True, then only the model's weights will be saved (`model.save_weights(filepath)`), else the full model is saved (`model.save(filepath)`).   |
| `log_weights`              | (boolean) if True save histograms of the model's layer's weights.                                                |
| `log_gradients`            | (boolean) if True log histograms of the training gradients                                                       |
| `training_data`            | (tuple) Same format `(X,y)` as passed to `model.fit`. This is needed for calculating gradients - this is mandatory if `log_gradients` is `True`.       |
| `validation_data`          | (tuple) Same format `(X,y)` as passed to `model.fit`. A set of data for wandb to visualize. If this is set, every epoch, wandb will make a small number of predictions and save the results for later visualization.          |
| `generator`                | (generator) a generator that returns validation data for wandb to visualize. This generator should return tuples `(X,y)`. Either `validate_data` or generator should be set for wandb to visualize specific data examples.     |
| `validation_steps`         | (int) if `validation_data` is a generator, how many steps to run the generator for the full validation set.       |
| `labels`                   | (list) If you are visualizing your data with wandb this list of labels will convert numeric output to understandable string if you are building a multiclass classifier. If you are making a binary classifier you can pass in a list of two labels \["label for false", "label for true"]. If `validate_data` and generator are both false, this won't do anything.    |
| `predictions`              | (int) the number of predictions to make for visualization each epoch, max is 100.    |
| `input_type`               | (string) type of the model input to help visualization. can be one of: (`image`, `images`, `segmentation_mask`).  |
| `output_type`              | (string) type of the model output to help visualziation. can be one of: (`image`, `images`, `segmentation_mask`).    |
| `log_evaluation`           | (boolean) if True, save a Table containing validation data and the model's predictions at each epoch. See `validation_indexes`, `validation_row_processor`, and `output_row_processor` for additional details.     |
| `class_colors`             | (\[float, float, float]) if the input or output is a segmentation mask, an array containing an rgb tuple (range 0-1) for each class.                  |
| `log_batch_frequency`      | (integer) if None, callback will log every epoch. If set to integer, callback will log training metrics every `log_batch_frequency` batches.          |
| `log_best_prefix`          | (string) if None, no extra summary metrics will be saved. If set to a string, the monitored metric and epoch will be prepended with this value and stored as summary metrics.   |
| `validation_indexes`       | (\[wandb.data\_types.\_TableLinkMixin]) an ordered list of index keys to associate with each validation example. If log\_evaluation is True and `validation_indexes` is provided, then a Table of validation data will not be created and instead each prediction will be associated with the row represented by the `TableLinkMixin`. The most common way to obtain such keys are is use `Table.get_index()` which will return a list of row keys.          |
| `validation_row_processor` | (Callable) a function to apply to the validation data, commonly used to visualize the data. The function will receive an `ndx` (int) and a `row` (dict). If your model has a single input, then `row["input"]` will be the input data for the row. Else, it will be keyed based on the name of the input slot. If your fit function takes a single target, then `row["target"]` will be the target data for the row. Else, it will be keyed based on the name of the output slots. For example, if your input data is a single ndarray, but you wish to visualize the data as an Image, then you can provide `lambda ndx, row: {"img": wandb.Image(row["input"])}` as the processor. Ignored if log\_evaluation is False or `validation_indexes` are present. |
| `output_row_processor`     | (Callable) same as `validation_row_processor`, but applied to the model's output. `row["output"]` will contain the results of the model output.          |
| `infer_missing_processors` | (bool) Determines if `validation_row_processor` and `output_row_processor` should be inferred if missing. Defaults to True. If `labels` are provided, we will attempt to infer classification-type processors where appropriate.      |
| `log_evaluation_frequency` | (int) Determines the frequency which evaluation results will be logged. Default 0 (only at the end of training). Set to 1 to log every epoch, 2 to log every other epoch, and so on. Has no effect when log\_evaluation is False.    |

## Frequently Asked Questions

### How do I use `Keras` multiprocessing with `wandb`?

If you're setting `use_multiprocessing=True` and seeing an error like:

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

then try this:

1. In the `Sequence` class construction, add: `wandb.init(group='...')`
2. In your main program, make sure you're using `if __name__ == "__main__":` and then put the rest of your script logic inside that.
