# sweep

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_sweep.py#L31-L89' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Initialize a hyperparameter sweep.

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) -> str
```

Search for hyperparameters that optimizes a cost function
of a machine learning model by testing various combinations.

Make note the unique identifier, `sweep_id`, that is returned.
At a later step provide the `sweep_id` to a sweep agent.

| Args |  |
| :--- | :--- |
|  `sweep` |  The configuration of a hyperparameter search. (or configuration generator). See [Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) for information on how to define your sweep. If you provide a callable, ensure that the callable does not take arguments and that it returns a dictionary that conforms to the W&B sweep config spec. |
|  `entity` |  The username or team name where you want to send W&B runs created by the sweep to. Ensure that the entity you specify already exists. If you don't specify an entity, the run will be sent to your default entity, which is usually your username. |
|  `project` |  The name of the project where W&B runs created from the sweep are sent to. If the project is not specified, the run is sent to a project labeled 'Uncategorized'. |
|  `prior_runs` |  The run IDs of existing runs to add to this sweep. |

| Returns |  |
| :--- | :--- |
|  `sweep_id` |  str. A unique identifier for the sweep. |
