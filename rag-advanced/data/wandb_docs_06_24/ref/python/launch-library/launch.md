# launch

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/launch/_launch.py#L246-L328' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Launch a W&B launch experiment.

```python
launch(
    api: Api,
    job: Optional[str] = None,
    entry_point: Optional[List[str]] = None,
    version: Optional[str] = None,
    name: Optional[str] = None,
    resource: Optional[str] = None,
    resource_args: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    docker_image: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    synchronous: Optional[bool] = (True),
    run_id: Optional[str] = None,
    repository: Optional[str] = None
) -> AbstractRun
```

| Arguments |  |
| :--- | :--- |
|  `job` |  string reference to a wandb.Job eg: wandb/test/my-job:latest |
|  `api` |  An instance of a wandb Api from wandb.apis.internal. |
|  `entry_point` |  Entry point to run within the project. Defaults to using the entry point used in the original run for wandb URIs, or main.py for git repository URIs. |
|  `version` |  For Git-based projects, either a commit hash or a branch name. |
|  `name` |  Name run under which to launch the run. |
|  `resource` |  Execution backend for the run. |
|  `resource_args` |  Resource related arguments for launching runs onto a remote backend. Will be stored on the constructed launch config under `resource_args`. |
|  `project` |  Target project to send launched run to |
|  `entity` |  Target entity to send launched run to |
|  `config` |  A dictionary containing the configuration for the run. May also contain resource specific arguments under the key "resource_args". |
|  `synchronous` |  Whether to block while waiting for a run to complete. Defaults to True. Note that if `synchronous` is False and `backend` is "local-container", this method will return, but the current process will block when exiting until the local run completes. If the current process is interrupted, any asynchronous runs launched via this method will be terminated. If `synchronous` is True and the run fails, the current process will error out as well. |
|  `run_id` |  ID for the run (To ultimately replace the :name: field) |
|  `repository` |  string name of repository path for remote registry |

#### Example:

```python
from wandb.sdk.launch import launch

job = "wandb/jobs/Hello World:latest"
params = {"epochs": 5}
# Run W&B project and create a reproducible docker environment
# on a local host
api = wandb.apis.internal.Api()
launch(api, job, parameters=params)
```

| Returns |  |
| :--- | :--- |
|  an instance of`wandb.launch.SubmittedRun` exposing information (e.g. run ID) about the launched run. |

| Raises |  |
| :--- | :--- |
|  `wandb.exceptions.ExecutionError` If a run launched in blocking mode is unsuccessful. |
