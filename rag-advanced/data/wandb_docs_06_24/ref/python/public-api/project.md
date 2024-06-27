# Project

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/projects.py#L79-L154' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


A project is a namespace for runs.

```python
Project(
    client, entity, project, attrs
)
```

| Attributes |  |
| :--- | :--- |

## Methods

### `artifacts_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/projects.py#L112-L114)

```python
artifacts_types(
    per_page=50
)
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/projects.py#L116-L154)

```python
sweeps()
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/projects.py#L96-L104)

```python
to_html(
    height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project.
