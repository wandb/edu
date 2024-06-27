# watch

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_watch.py#L20-L106' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Hook into the torch model to collect gradients and the topology.

```python
watch(
    models,
    criterion=None,
    log: Optional[Literal['gradients', 'parameters', 'all']] = "gradients",
    log_freq: int = 1000,
    idx: Optional[int] = None,
    log_graph: bool = (False)
)
```

Should be extended to accept arbitrary ML models.

| Args |  |
| :--- | :--- |
|  `models` |  (torch.Module) The model to hook, can be a tuple |
|  `criterion` |  (torch.F) An optional loss value being optimized |
|  `log` |  (str) One of "gradients", "parameters", "all", or None |
|  `log_freq` |  (int) log gradients and parameters every N batches |
|  `idx` |  (int) an index to be used when calling wandb.watch on multiple models |
|  `log_graph` |  (boolean) log graph topology |

| Returns |  |
| :--- | :--- |
|  `wandb.Graph`: The graph object that will populate after the first backward pass |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If called before `wandb.init` or if any of models is not a torch.nn.Module. |
