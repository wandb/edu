# WandbCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/keras.py#L291-L1080' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


`WandbCallback` automatically integrates keras with wandb.

```python
WandbCallback(
    monitor="val_loss", verbose=0, mode="auto", save_weights_only=(False),
    log_weights=(False), log_gradients=(False), save_model=(True),
    training_data=None, validation_data=None, labels=None, predictions=36,
    generator=None, input_type=None, output_type=None, log_evaluation=(False),
    validation_steps=None, class_colors=None, log_batch_frequency=None,
    log_best_prefix="best_", save_graph=(True), validation_indexes=None,
    validation_row_processor=None, prediction_row_processor=None,
    infer_missing_processors=(True), log_evaluation_frequency=0,
    compute_flops=(False), **kwargs
)
```

#### Example:

```python
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[WandbCallback()],
)
```

`WandbCallback` will automatically log history data from any
metrics collected by keras: loss and anything passed into `keras_model.compile()`.

`WandbCallback` will set summary metrics for the run associated with the "best" training
step, where "best" is defined by the `monitor` and `mode` attributes.  This defaults
to the epoch with the minimum `val_loss`. `WandbCallback` will by default save the model
associated with the best `epoch`.

`WandbCallback` can optionally log gradient and parameter histograms.

`WandbCallback` can optionally save training and validation data for wandb to visualize.

| Arguments |  |
| :--- | :--- |
|  `monitor` |  (str) name of metric to monitor. Defaults to `val_loss`. |
|  `mode` |  (str) one of {`auto`, `min`, `max`}. `min` - save model when monitor is minimized `max` - save model when monitor is maximized `auto` - try to guess when to save the model (default). |
|  `save_model` |  True - save a model when monitor beats all previous epochs False - don't save models |
|  `save_graph` |  (boolean) if True save model graph to wandb (default to True). |
|  `save_weights_only` |  (boolean) if True, then only the model's weights will be saved (`model.save_weights(filepath)`), else the full model is saved (`model.save(filepath)`). |
|  `log_weights` |  (boolean) if True save histograms of the model's layer's weights. |
|  `log_gradients` |  (boolean) if True log histograms of the training gradients |
|  `training_data` |  (tuple) Same format `(X,y)` as passed to `model.fit`. This is needed for calculating gradients - this is mandatory if `log_gradients` is `True`. |
|  `validation_data` |  (tuple) Same format `(X,y)` as passed to `model.fit`. A set of data for wandb to visualize. If this is set, every epoch, wandb will make a small number of predictions and save the results for later visualization. In case you are working with image data, please also set `input_type` and `output_type` in order to log correctly. |
|  `generator` |  (generator) a generator that returns validation data for wandb to visualize. This generator should return tuples `(X,y)`. Either `validate_data` or generator should be set for wandb to visualize specific data examples. In case you are working with image data, please also set `input_type` and `output_type` in order to log correctly. |
|  `validation_steps` |  (int) if `validation_data` is a generator, how many steps to run the generator for the full validation set. |
|  `labels` |  (list) If you are visualizing your data with wandb this list of labels will convert numeric output to understandable string if you are building a multiclass classifier. If you are making a binary classifier you can pass in a list of two labels ["label for false", "label for true"]. If `validate_data` and generator are both false, this won't do anything. |
|  `predictions` |  (int) the number of predictions to make for visualization each epoch, max is 100. |
|  `input_type` |  (string) type of the model input to help visualization. can be one of: (`image`, `images`, `segmentation_mask`, `auto`). |
|  `output_type` |  (string) type of the model output to help visualization. can be one of: (`image`, `images`, `segmentation_mask`, `label`). |
|  `log_evaluation` |  (boolean) if True, save a Table containing validation data and the model's predictions at each epoch. See `validation_indexes`, `validation_row_processor`, and `output_row_processor` for additional details. |
|  `class_colors` |  ([float, float, float]) if the input or output is a segmentation mask, an array containing an rgb tuple (range 0-1) for each class. |
|  `log_batch_frequency` |  (integer) if None, callback will log every epoch. If set to integer, callback will log training metrics every `log_batch_frequency` batches. |
|  `log_best_prefix` |  (string) if None, no extra summary metrics will be saved. If set to a string, the monitored metric and epoch will be prepended with this value and stored as summary metrics. |
|  `validation_indexes` |  ([wandb.data_types._TableLinkMixin]) an ordered list of index keys to associate with each validation example. If log_evaluation is True and `validation_indexes` is provided, then a Table of validation data will not be created and instead each prediction will be associated with the row represented by the `TableLinkMixin`. The most common way to obtain such keys are is use `Table.get_index()` which will return a list of row keys. |
|  `validation_row_processor` |  (Callable) a function to apply to the validation data, commonly used to visualize the data. The function will receive an `ndx` (int) and a `row` (dict). If your model has a single input, then `row["input"]` will be the input data for the row. Else, it will be keyed based on the name of the input slot. If your fit function takes a single target, then `row["target"]` will be the target data for the row. Else, it will be keyed based on the name of the output slots. For example, if your input data is a single ndarray, but you wish to visualize the data as an Image, then you can provide `lambda ndx, row: {"img": wandb.Image(row["input"])}` as the processor. Ignored if log_evaluation is False or `validation_indexes` are present. |
|  `output_row_processor` |  (Callable) same as `validation_row_processor`, but applied to the model's output. `row["output"]` will contain the results of the model output. |
|  `infer_missing_processors` |  (bool) Determines if `validation_row_processor` and `output_row_processor` should be inferred if missing. Defaults to True. If `labels` are provided, we will attempt to infer classification-type processors where appropriate. |
|  `log_evaluation_frequency` |  (int) Determines the frequency which evaluation results will be logged. Default 0 (only at the end of training). Set to 1 to log every epoch, 2 to log every other epoch, and so on. Has no effect when log_evaluation is False. |
|  `compute_flops` |  (bool) Compute the FLOPs of your Keras Sequential or Functional model in GigaFLOPs unit. |

## Methods

### `get_flops`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/keras.py#L1034-L1080)

```python
get_flops() -> float
```

Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model in inference mode.

It uses tf.compat.v1.profiler under the hood.

### `set_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/keras.py#L554-L563)

```python
set_model(
    model
)
```

### `set_params`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/keras.py#L551-L552)

```python
set_params(
    params
)
```
