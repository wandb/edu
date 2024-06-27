# Files

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/files.py#L43-L105' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


An iterable collection of `File` objects.

```python
Files(
    client, run, names=None, per_page=50, upload=(False)
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/files.py#L98-L102)

```python
convert_objects()
```

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/files.py#L95-L96)

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
