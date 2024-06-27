# WandbEvalCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L10-L226' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Abstract base class to build Keras callbacks for model prediction visualization.

```python
WandbEvalCallback(
    data_table_columns: List[str],
    pred_table_columns: List[str],
    *args,
    **kwargs
) -> None
```

You can build callbacks for visualizing model predictions `on_epoch_end`
that can be passed to `model.fit()` for classification, object detection,
segmentation, etc. tasks.

To use this, inherit from this base callback class and implement the
`add_ground_truth` and `add_model_prediction` methods.

The base class will take care of the following:

- Initialize `data_table` for logging the ground truth and
  `pred_table` for predictions.
- The data uploaded to `data_table` is used as a reference for the
  `pred_table`. This is to reduce the memory footprint. The `data_table_ref`
  is a list that can be used to access the referenced data.
  Check out the example below to see how it's done.
- Log the tables to W&B as W&B Artifacts.
- Each new `pred_table` is logged as a new version with aliases.

#### Example:

```python
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(self, validation_data, data_table_columns, pred_table_columns):
        super().__init__(data_table_columns, pred_table_columns)

        self.x = validation_data[0]
        self.y = validation_data[1]

    def add_ground_truth(self):
        for idx, (image, label) in enumerate(zip(self.x, self.y)):
            self.data_table.add_data(idx, wandb.Image(image), label)

    def add_model_predictions(self, epoch):
        preds = self.model.predict(self.x, verbose=0)
        preds = tf.argmax(preds, axis=-1)

        data_table_ref = self.data_table_ref
        table_idxs = data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],
                data_table_ref.data[idx][1],
                data_table_ref.data[idx][2],
                pred,
            )


model.fit(
    x,
    y,
    epochs=2,
    validation_data=(x, y),
    callbacks=[
        WandbClfEvalCallback(
            validation_data=(x, y),
            data_table_columns=["idx", "image", "label"],
            pred_table_columns=["epoch", "idx", "image", "label", "pred"],
        )
    ],
)
```

To have more fine-grained control, you can override the `on_train_begin` and
`on_epoch_end` methods. If you want to log the samples after N batched, you
can implement `on_train_batch_end` method.

## Methods

### `add_ground_truth`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L117-L131)

```python
@abc.abstractmethod
add_ground_truth(
    logs: Optional[Dict[str, float]] = None
) -> None
```

Add ground truth data to `data_table`.

Use this method to write the logic for adding validation/training data to
`data_table` initialized using `init_data_table` method.

#### Example:

```python
for idx, data in enumerate(dataloader):
    self.data_table.add_data(idx, data)
```

This method is called once `on_train_begin` or equivalent hook.

### `add_model_predictions`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L133-L153)

```python
@abc.abstractmethod
add_model_predictions(
    epoch: int,
    logs: Optional[Dict[str, float]] = None
) -> None
```

Add a prediction from a model to `pred_table`.

Use this method to write the logic for adding model prediction for validation/
training data to `pred_table` initialized using `init_pred_table` method.

#### Example:

```python
# Assuming the dataloader is not shuffling the samples.
for idx, data in enumerate(dataloader):
    preds = model.predict(data)
    self.pred_table.add_data(
        self.data_table_ref.data[idx][0], self.data_table_ref.data[idx][1], preds
    )
```

This method is called `on_epoch_end` or equivalent hook.

### `init_data_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L155-L164)

```python
init_data_table(
    column_names: List[str]
) -> None
```

Initialize the W&B Tables for validation data.

Call this method `on_train_begin` or equivalent hook. This is followed by adding
data to the table row or column wise.

| Args |  |
| :--- | :--- |
|  `column_names` |  (list) Column names for W&B Tables. |

### `init_pred_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L166-L175)

```python
init_pred_table(
    column_names: List[str]
) -> None
```

Initialize the W&B Tables for model evaluation.

Call this method `on_epoch_end` or equivalent hook. This is followed by adding
data to the table row or column wise.

| Args |  |
| :--- | :--- |
|  `column_names` |  (list) Column names for W&B Tables. |

### `log_data_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L177-L203)

```python
log_data_table(
    name: str = "val",
    type: str = "dataset",
    table_name: str = "val_data"
) -> None
```

Log the `data_table` as W&B artifact and call `use_artifact` on it.

This lets the evaluation table use the reference of already uploaded data
(images, text, scalar, etc.) without re-uploading.

| Args |  |
| :--- | :--- |
|  `name` |  (str) A human-readable name for this artifact, which is how you can identify this artifact in the UI or reference it in use_artifact calls. (default is 'val') |
|  `type` |  (str) The type of the artifact, which is used to organize and differentiate artifacts. (default is 'dataset') |
|  `table_name` |  (str) The name of the table as will be displayed in the UI. (default is 'val_data'). |

### `log_pred_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L205-L226)

```python
log_pred_table(
    type: str = "evaluation",
    table_name: str = "eval_data",
    aliases: Optional[List[str]] = None
) -> None
```

Log the W&B Tables for model evaluation.

The table will be logged multiple times creating new version. Use this
to compare models at different intervals interactively.

| Args |  |
| :--- | :--- |
|  `type` |  (str) The type of the artifact, which is used to organize and differentiate artifacts. (default is 'evaluation') |
|  `table_name` |  (str) The name of the table as will be displayed in the UI. (default is 'eval_data') |
|  `aliases` |  (List[str]) List of aliases for the prediction table. |

### `set_model`

```python
set_model(
    model
)
```

### `set_params`

```python
set_params(
    params
)
```
