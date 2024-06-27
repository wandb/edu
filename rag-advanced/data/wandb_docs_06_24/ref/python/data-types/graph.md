# Graph

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1329-L1488' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Wandb class for graphs.

```python
Graph(
    format="keras"
)
```

This class is typically used for saving and displaying neural net models.  It
represents the graph as an array of nodes and edges.  The nodes can have
labels that can be visualized by wandb.

#### Examples:

Import a keras model:

```
Graph.from_keras(keras_model)
```

## Methods

### `add_edge`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1414-L1418)

```python
add_edge(
    from_node, to_node
)
```

### `add_node`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1402-L1412)

```python
add_node(
    node=None, **node_kwargs
)
```

### `from_keras`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1420-L1449)

```python
@classmethod
from_keras(
    model
)
```

### `pprint`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1396-L1400)

```python
pprint()
```

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1393-L1394)

```python
__getitem__(
    nid
)
```
