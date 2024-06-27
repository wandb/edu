# WBTraceTree

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/trace_tree.py#L80-L119' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Media object for trace tree data.

```python
WBTraceTree(
    root_span: Span,
    model_dict: typing.Optional[dict] = None
)
```

| Arguments |  |
| :--- | :--- |
|  root_span (Span): The root span of the trace tree. model_dict (dict, optional): A dictionary containing the model dump. NOTE: model_dict is a completely-user-defined dict. The UI will render a JSON viewer for this dict, giving special treatment to dictionaries with a `_kind` key. This is because model vendors have such different serialization formats that we need to be flexible here. |
