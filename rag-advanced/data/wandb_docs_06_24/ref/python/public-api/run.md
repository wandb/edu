# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L272-L901' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


A single run associated with an entity and project.

```python
Run(
    client: "RetryingClient",
    entity: str,
    project: str,
    run_id: str,
    attrs: Optional[Mapping] = None,
    include_sweeps: bool = (True)
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `create`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L371-L411)

```python
@classmethod
create(
    api, run_id=None, project=None, entity=None
)
```

Create a run for the given project.

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L529-L557)

```python
delete(
    delete_artifacts=(False)
)
```

Delete the given run from the wandb backend.

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `file`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L619-L629)

```python
file(
    name
)
```

Return the path of a file with a given name in the artifact.

| Arguments |  |
| :--- | :--- |
|  name (str): name of requested file. |

| Returns |  |
| :--- | :--- |
|  A `File` matching the name argument. |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L606-L617)

```python
files(
    names=None, per_page=50
)
```

Return a file path for each file named.

| Arguments |  |
| :--- | :--- |
|  names (list): names of the requested files, if empty returns all files per_page (int): number of results per page. |

| Returns |  |
| :--- | :--- |
|  A `Files` object, which is an iterator over `File` objects. |

### `history`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L655-L695)

```python
history(
    samples=500, keys=None, x_axis="_step", pandas=(True), stream="default"
)
```

Return sampled history metrics for a run.

This is simpler and faster if you are ok with the history records being sampled.

| Arguments |  |
| :--- | :--- |
|  `samples` |  (int, optional) The number of samples to return |
|  `pandas` |  (bool, optional) Return a pandas dataframe |
|  `keys` |  (list, optional) Only return metrics for specific keys |
|  `x_axis` |  (str, optional) Use this metric as the xAxis defaults to _step |
|  `stream` |  (str, optional) "default" for metrics, "system" for machine metrics |

| Returns |  |
| :--- | :--- |
|  `pandas.DataFrame` |  If pandas=True returns a `pandas.DataFrame` of history metrics. list of dicts: If pandas=False returns a list of dicts of history metrics. |

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L413-L477)

```python
load(
    force=(False)
)
```

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L794-L828)

```python
log_artifact(
    artifact, aliases=None
)
```

Declare an artifact as output of a run.

| Arguments |  |
| :--- | :--- |
|  artifact (`Artifact`): An artifact returned from `wandb.Api().artifact(name)` aliases (list, optional): Aliases to apply to this artifact |

| Returns |  |
| :--- | :--- |
|  A `Artifact` object. |

### `logged_artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L753-L755)

```python
logged_artifacts(
    per_page=100
)
```

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L559-L560)

```python
save()
```

### `scan_history`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L697-L751)

```python
scan_history(
    keys=None, page_size=1000, min_step=None, max_step=None
)
```

Returns an iterable collection of all history records for a run.

#### Example:

Export all the loss values for an example run

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

| Arguments |  |
| :--- | :--- |
|  keys ([str], optional): only fetch these keys, and only fetch rows that have all of keys defined. page_size (int, optional): size of pages to fetch from the api. min_step (int, optional): the minimum number of pages to scan at a time. max_step (int, optional): the maximum number of pages to scan at a time. |

| Returns |  |
| :--- | :--- |
|  An iterable collection over history records (dict). |

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L887-L895)

```python
to_html(
    height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this run.

### `update`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L502-L527)

```python
update()
```

Persist changes to the run object to the wandb backend.

### `upload_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L631-L653)

```python
upload_file(
    path, root="."
)
```

Upload a file.

| Arguments |  |
| :--- | :--- |
|  path (str): name of file to upload. root (str): the root path to save the file relative to. i.e. If you want to have the file saved in the run as "my_dir/file.txt" and you're currently in "my_dir" you would set root to "../". |

| Returns |  |
| :--- | :--- |
|  A `File` matching the name argument. |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L761-L792)

```python
use_artifact(
    artifact, use_as=None
)
```

Declare an artifact as an input to a run.

| Arguments |  |
| :--- | :--- |
|  artifact (`Artifact`): An artifact returned from `wandb.Api().artifact(name)` use_as (string, optional): A string identifying how the artifact is used in the script. Used to easily differentiate artifacts used in a run, when using the beta wandb launch feature's artifact swapping functionality. |

| Returns |  |
| :--- | :--- |
|  A `Artifact` object. |

### `used_artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L757-L759)

```python
used_artifacts(
    per_page=100
)
```

### `wait_until_finished`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L479-L500)

```python
wait_until_finished()
```
