# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L95-L1179' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Used for querying the wandb server.

```python
Api(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) -> None
```

#### Examples:

Most common way to initialize

```
>>> wandb.Api()
```

| Arguments |  |
| :--- | :--- |
|  `overrides` |  (dict) You can set `base_url` if you are using a wandb server other than https://api.wandb.ai. You can also set defaults for `entity`, `project`, and `run`. |

| Attributes |  |
| :--- | :--- |

## Methods

### `artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1017-L1041)

```python
artifact(
    name, type=None
)
```

Return a single artifact by parsing path in the form `entity/project/name`.

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) An artifact name. May be prefixed with entity/project. Valid names can be in the following forms: name:version name:alias |
|  `type` |  (str, optional) The type of artifact to fetch. |

| Returns |  |
| :--- | :--- |
|  A `Artifact` object. |

### `artifact_collection`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L967-L983)

```python
artifact_collection(
    type_name: str,
    name: str
) -> "public.ArtifactCollection"
```

Return a single artifact collection by type and parsing path in the form `entity/project/name`.

| Arguments |  |
| :--- | :--- |
|  `type_name` |  (str) The type of artifact collection to fetch. |
|  `name` |  (str) An artifact collection name. May be prefixed with entity/project. |

| Returns |  |
| :--- | :--- |
|  An `ArtifactCollection` object. |

### `artifact_collection_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1162-L1179)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

Return whether an artifact collection exists within a specified project and entity.

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) An artifact collection name. May be prefixed with entity/project. If entity or project is not specified, it will be inferred from the override params if populated. Otherwise, entity will be pulled from the user settings and project will default to "uncategorized". |
|  `type` |  (str) The type of artifact collection |

| Returns |  |
| :--- | :--- |
|  True if the artifact collection exists, False otherwise. |

### `artifact_collections`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L947-L965)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: Optional[int] = 50
) -> "public.ArtifactCollections"
```

Return a collection of matching artifact collections.

| Arguments |  |
| :--- | :--- |
|  `project_name` |  (str) The name of the project to filter on. |
|  `type_name` |  (str) The name of the artifact type to filter on. |
|  `per_page` |  (int, optional) Sets the page size for query pagination. None will use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  An iterable `ArtifactCollections` object. |

### `artifact_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1140-L1160)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

Return whether an artifact version exists within a specified project and entity.

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) An artifact name. May be prefixed with entity/project. If entity or project is not specified, it will be inferred from the override params if populated. Otherwise, entity will be pulled from the user settings and project will default to "uncategorized". Valid names can be in the following forms: name:version name:alias |
|  `type` |  (str, optional) The type of artifact |

| Returns |  |
| :--- | :--- |
|  True if the artifact version exists, False otherwise. |

### `artifact_type`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L931-L945)

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) -> "public.ArtifactType"
```

Return the matching `ArtifactType`.

| Arguments |  |
| :--- | :--- |
|  `type_name` |  (str) The name of the artifact type to retrieve. |
|  `project` |  (str, optional) If given, a project name or path to filter on. |

| Returns |  |
| :--- | :--- |
|  An `ArtifactType` object. |

### `artifact_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L918-L929)

```python
artifact_types(
    project: Optional[str] = None
) -> "public.ArtifactTypes"
```

Return a collection of matching artifact types.

| Arguments |  |
| :--- | :--- |
|  `project` |  (str, optional) If given, a project name or path to filter on. |

| Returns |  |
| :--- | :--- |
|  An iterable `ArtifactTypes` object. |

### `artifact_versions`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L985-L995)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

Deprecated, use `artifacts(type_name, name)` instead.

### `artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L997-L1015)

```python
artifacts(
    type_name: str,
    name: str,
    per_page: Optional[int] = 50
) -> "public.Artifacts"
```

Return an `Artifacts` collection from the given parameters.

| Arguments |  |
| :--- | :--- |
|  `type_name` |  (str) The type of artifacts to fetch. |
|  `name` |  (str) An artifact collection name. May be prefixed with entity/project. |
|  `per_page` |  (int, optional) Sets the page size for query pagination. None will use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  An iterable `Artifacts` object. |

### `create_project`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L281-L288)

```python
create_project(
    name: str,
    entity: str
) -> None
```

Create a new project.

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) The name of the new project. |
|  `entity` |  (str) The entity of the new project. |

### `create_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L290-L310)

```python
create_run(
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) -> "public.Run"
```

Create a new run.

| Arguments |  |
| :--- | :--- |
|  `run_id` |  (str, optional) The ID to assign to the run, if given. The run ID is automatically generated by default, so in general, you do not need to specify this and should only do so at your own risk. |
|  `project` |  (str, optional) If given, the project of the new run. |
|  `entity` |  (str, optional) If given, the entity of the new run. |

| Returns |  |
| :--- | :--- |
|  The newly created `Run`. |

### `create_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L312-L422)

```python
create_run_queue(
    name: str,
    type: "public.RunQueueResourceType",
    entity: Optional[str] = None,
    prioritization_mode: Optional['public.RunQueuePrioritizationMode'] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) -> "public.RunQueue"
```

Create a new run queue (launch).

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) Name of the queue to create |
|  `type` |  (str) Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes", "sagemaker", or "gcp-vertex". |
|  `entity` |  (str) Optional name of the entity to create the queue. If None, will use the configured or default entity. |
|  `prioritization_mode` |  (str) Optional version of prioritization to use. Either "V0" or None |
|  `config` |  (dict) Optional default resource configuration to be used for the queue. Use handlebars (eg. "{{var}}") to specify template variables. |
|  `template_variables` |  (dict) A dictionary of template variable schemas to be used with the config. Expected format of: { "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value), "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] } } } |

| Returns |  |
| :--- | :--- |
|  The newly created `RunQueue` |

| Raises |  |
| :--- | :--- |
|  ValueError if any of the parameters are invalid wandb.Error on wandb API errors |

### `create_team`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L705-L715)

```python
create_team(
    team, admin_username=None
)
```

Create a new team.

| Arguments |  |
| :--- | :--- |
|  `team` |  (str) The name of the team |
|  `admin_username` |  (str) optional username of the admin user of the team, defaults to the current user. |

| Returns |  |
| :--- | :--- |
|  A `Team` object |

### `create_user`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L424-L434)

```python
create_user(
    email, admin=(False)
)
```

Create a new user.

| Arguments |  |
| :--- | :--- |
|  `email` |  (str) The email address of the user |
|  `admin` |  (bool) Whether this user should be a global instance admin |

| Returns |  |
| :--- | :--- |
|  A `User` object |

### `flush`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L501-L508)

```python
flush()
```

Flush the local cache.

The api object keeps a local cache of runs, so if the state of the run may
change while executing your script you must clear the local cache with
`api.flush()` to get the latest values associated with the run.

### `from_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L510-L564)

```python
from_path(
    path
)
```

Return a run, sweep, project or report from a path.

#### Examples:

```
project = api.from_path("my_project")
team_project = api.from_path("my_team/my_project")
run = api.from_path("my_team/my_project/runs/id")
sweep = api.from_path("my_team/my_project/sweeps/id")
report = api.from_path("my_team/my_project/reports/My-Report-Vm11dsdf")
```

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) The path to the project, run, sweep or report |

| Returns |  |
| :--- | :--- |
|  A `Project`, `Run`, `Sweep`, or `BetaReport` instance. |

| Raises |  |
| :--- | :--- |
|  wandb.Error if path is invalid or the object doesn't exist |

### `job`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1043-L1060)

```python
job(
    name: Optional[str],
    path: Optional[str] = None
) -> "public.Job"
```

Return a `Job` from the given parameters.

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) The job name. |
|  `path` |  (str, optional) If given, the root path in which to download the job artifact. |

| Returns |  |
| :--- | :--- |
|  A `Job` object. |

### `list_jobs`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L1062-L1138)

```python
list_jobs(
    entity: str,
    project: str
) -> List[Dict[str, Any]]
```

Return a list of jobs, if any, for the given entity and project.

| Arguments |  |
| :--- | :--- |
|  `entity` |  (str) The entity for the listed job(s). |
|  `project` |  (str) The project for the listed job(s). |

| Returns |  |
| :--- | :--- |
|  A list of matching jobs. |

### `project`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L657-L670)

```python
project(
    name: str,
    entity: Optional[str] = None
) -> "public.Project"
```

Return the `Project` with the given name (and entity, if given).

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) The project name. |
|  `entity` |  (str) Name of the entity requested. If None, will fall back to the default entity passed to `Api`. If no default entity, will raise a `ValueError`. |

| Returns |  |
| :--- | :--- |
|  A `Project` object. |

### `projects`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L631-L655)

```python
projects(
    entity: Optional[str] = None,
    per_page: Optional[int] = 200
) -> "public.Projects"
```

Get projects for a given entity.

| Arguments |  |
| :--- | :--- |
|  `entity` |  (str) Name of the entity requested. If None, will fall back to the default entity passed to `Api`. If no default entity, will raise a `ValueError`. |
|  `per_page` |  (int) Sets the page size for query pagination. None will use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A `Projects` object which is an iterable collection of `Project` objects. |

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L863-L884)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

Return a single queued run based on the path.

Parses paths of the form entity/project/queue_id/run_queue_item_id.

### `reports`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L672-L703)

```python
reports(
    path: str = "",
    name: Optional[str] = None,
    per_page: Optional[int] = 50
) -> "public.Reports"
```

Get reports for a given project path.

WARNING: This api is in beta and will likely change in a future release

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) path to project the report resides in, should be in the form: "entity/project" |
|  `name` |  (str, optional) optional name of the report requested. |
|  `per_page` |  (int) Sets the page size for query pagination. None will use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A `Reports` object which is an iterable collection of `BetaReport` objects. |

### `run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L846-L861)

```python
run(
    path=""
)
```

Return a single run by parsing path in the form entity/project/run_id.

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) path to run in the form `entity/project/run_id`. If `api.entity` is set, this can be in the form `project/run_id` and if `api.project` is set this can just be the run_id. |

| Returns |  |
| :--- | :--- |
|  A `Run` object. |

### `run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L886-L899)

```python
run_queue(
    entity, name
)
```

Return the named `RunQueue` for entity.

To create a new `RunQueue`, use `wandb.Api().create_run_queue(...)`.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L766-L844)

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

Return a set of runs from a project that match the filters provided.

You can filter by `config.*`, `summary_metrics.*`, `tags`, `state`, `entity`, `createdAt`, etc.

#### Examples:

Find runs in my_project where config.experiment_name has been set to "foo"

```
api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
```

Find runs in my_project where config.experiment_name has been set to "foo" or "bar"

```
api.runs(
    path="my_entity/my_project",
    filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
)
```

Find runs in my_project where config.experiment_name matches a regex (anchors are not supported)

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}}
)
```

Find runs in my_project where the run name matches a regex (anchors are not supported)

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}}
)
```

Find runs in my_project sorted by ascending loss

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) path to project, should be in the form: "entity/project" |
|  `filters` |  (dict) queries for specific runs using the MongoDB query language. You can filter by run properties such as config.key, summary_metrics.key, state, entity, createdAt, etc. For example: {"config.experiment_name": "foo"} would find runs with a config entry of experiment name set to "foo" You can compose operations to make more complicated queries, see Reference for the language is at https://docs.mongodb.com/manual/reference/operator/query |
|  `order` |  (str) Order can be `created_at`, `heartbeat_at`, `config.*.value`, or `summary_metrics.*`. If you prepend order with a + order is ascending. If you prepend order with a - order is descending (default). The default order is run.created_at from newest to oldest. |
|  `per_page` |  (int) Sets the page size for query pagination. |
|  `include_sweeps` |  (bool) Whether to include the sweep runs in the results. |

| Returns |  |
| :--- | :--- |
|  A `Runs` object, which is an iterable collection of `Run` objects. |

### `sweep`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L901-L916)

```python
sweep(
    path=""
)
```

Return a sweep by parsing path in the form `entity/project/sweep_id`.

| Arguments |  |
| :--- | :--- |
|  `path` |  (str, optional) path to sweep in the form entity/project/sweep_id. If `api.entity` is set, this can be in the form project/sweep_id and if `api.project` is set this can just be the sweep_id. |

| Returns |  |
| :--- | :--- |
|  A `Sweep` object. |

### `sync_tensorboard`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L436-L458)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

Sync a local directory containing tfevent files to wandb.

### `team`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L717-L726)

```python
team(
    team: str
) -> "public.Team"
```

Return the matching `Team` with the given name.

| Arguments |  |
| :--- | :--- |
|  `team` |  (str) The name of the team. |

| Returns |  |
| :--- | :--- |
|  A `Team` object. |

### `user`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L728-L748)

```python
user(
    username_or_email: str
) -> Optional['public.User']
```

Return a user from a username or email address.

Note: This function only works for Local Admins, if you are trying to get your own user object, please use `api.viewer`.

| Arguments |  |
| :--- | :--- |
|  `username_or_email` |  (str) The username or email address of the user |

| Returns |  |
| :--- | :--- |
|  A `User` object or None if a user couldn't be found |

### `users`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/api.py#L750-L764)

```python
users(
    username_or_email: str
) -> List['public.User']
```

Return all users from a partial username or email address query.

Note: This function only works for Local Admins, if you are trying to get your own user object, please use `api.viewer`.

| Arguments |  |
| :--- | :--- |
|  `username_or_email` |  (str) The prefix or suffix of the user you want to find |

| Returns |  |
| :--- | :--- |
|  An array of `User` objects |

| Class Variables |  |
| :--- | :--- |
|  `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
|  `DEFAULT_ENTITY_QUERY`<a id="DEFAULT_ENTITY_QUERY"></a> |   |
|  `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
|  `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |
