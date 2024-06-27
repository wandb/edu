# Runs

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L61-L269' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


An iterable collection of runs associated with a project and optional filter.

```python
Runs(
    client: "RetryingClient",
    entity: str,
    project: str,
    filters: Optional[Dict[str, Any]] = None,
    order: Optional[str] = None,
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

This is generally used indirectly via the `Api`.runs method.

| Attributes |  |
| :--- | :--- |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L136-L168)

```python
convert_objects()
```

### `histories`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L170-L266)

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    format: Literal['default', 'pandas', 'polars'] = "default",
    stream: Literal['default', 'system'] = "default"
)
```

Return sampled history metrics for all runs that fit the filters conditions.

| Arguments |  |
| :--- | :--- |
|  `samples` |  (int, optional) The number of samples to return per run |
|  `keys` |  (list[str], optional) Only return metrics for specific keys |
|  `x_axis` |  (str, optional) Use this metric as the xAxis defaults to _step |
|  `format` |  (Literal, optional) Format to return data in, options are "default", "pandas", "polars" |
|  `stream` |  (Literal, optional) "default" for metrics, "system" for machine metrics |

| Returns |  |
| :--- | :--- |
|  `pandas.DataFrame` |  If format="pandas", returns a `pandas.DataFrame` of history metrics. |
|  `polars.DataFrame` |  If format="polars", returns a `polars.DataFrame` of history metrics. list of dicts: If format="default", returns a list of dicts containing history metrics with a run_id key. |

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L52-L53)

```python
update_variables()
```

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
