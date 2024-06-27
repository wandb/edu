# launch_add

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/launch/_launch_add.py#L34-L131' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Enqueue a W&B launch experiment. With either a source uri, job or docker_image.

```python
launch_add(
    uri: Optional[str] = None,
    job: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    template_variables: Optional[Dict[str, Union[float, int, str]]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    queue_name: Optional[str] = None,
    resource: Optional[str] = None,
    entry_point: Optional[List[str]] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    docker_image: Optional[str] = None,
    project_queue: Optional[str] = None,
    resource_args: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    build: Optional[bool] = (False),
    repository: Optional[str] = None,
    sweep_id: Optional[str] = None,
    author: Optional[str] = None,
    priority: Optional[int] = None
) -> "public.QueuedRun"
```

| Arguments |  |
| :--- | :--- |
|  `uri` |  URI of experiment to run. A wandb run uri or a Git repository URI. |
|  `job` |  string reference to a wandb.Job eg: wandb/test/my-job:latest |
|  `config` |  A dictionary containing the configuration for the run. May also contain resource specific arguments under the key "resource_args" |
|  `template_variables` |  A dictionary containing values of template variables for a run queue. Expected format of {"VAR_NAME": VAR_VALUE} |
|  `project` |  Target project to send launched run to |
|  `entity` |  Target entity to send launched run to |
|  `queue` |  the name of the queue to enqueue the run to |
|  `priority` |  the priority level of the job, where 1 is the highest priority |
|  `resource` |  Execution backend for the run: W&B provides built-in support for "local-container" backend |
|  `entry_point` |  Entry point to run within the project. Defaults to using the entry point used in the original run for wandb URIs, or main.py for git repository URIs. |
|  `name` |  Name run under which to launch the run. |
|  `version` |  For Git-based projects, either a commit hash or a branch name. |
|  `docker_image` |  The name of the docker image to use for the run. |
|  `resource_args` |  Resource related arguments for launching runs onto a remote backend. Will be stored on the constructed launch config under `resource_args`. |
|  `run_id` |  optional string indicating the id of the launched run |
|  `build` |  optional flag defaulting to false, requires queue to be set if build, an image is created, creates a job artifact, pushes a reference to that job artifact to queue |
|  `repository` |  optional string to control the name of the remote repository, used when pushing images to a registry |
|  `project_queue` |  optional string to control the name of the project for the queue. Primarily used for back compatibility with project scoped queues |

#### Example:

```python
from wandb.sdk.launch import launch_add

project_uri = "https://github.com/wandb/examples"
params = {"alpha": 0.5, "l1_ratio": 0.01}
# Run W&B project and create a reproducible docker environment
# on a local host
api = wandb.apis.internal.Api()
launch_add(uri=project_uri, parameters=params)
```

| Returns |  |
| :--- | :--- |
|  an instance of`wandb.api.public.QueuedRun` which gives information about the queued run, or if `wait_until_started` or `wait_until_finished` are called, gives access to the underlying Run information. |

| Raises |  |
| :--- | :--- |
|  `wandb.exceptions.LaunchError` if unsuccessful |
