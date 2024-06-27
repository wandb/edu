# init

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_init.py#L924-L1186' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Start a new run to track and log to W&B.

```python
init(
    job_type: Optional[str] = None,
    dir: Optional[StrPath] = None,
    config: Union[Dict, str, None] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    reinit: Optional[bool] = None,
    tags: Optional[Sequence] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    magic: Optional[Union[dict, str, bool]] = None,
    config_exclude_keys: Optional[List[str]] = None,
    config_include_keys: Optional[List[str]] = None,
    anonymous: Optional[str] = None,
    mode: Optional[str] = None,
    allow_val_change: Optional[bool] = None,
    resume: Optional[Union[bool, str]] = None,
    force: Optional[bool] = None,
    tensorboard: Optional[bool] = None,
    sync_tensorboard: Optional[bool] = None,
    monitor_gym: Optional[bool] = None,
    save_code: Optional[bool] = None,
    id: Optional[str] = None,
    fork_from: Optional[str] = None,
    resume_from: Optional[str] = None,
    settings: Union[Settings, Dict[str, Any], None] = None
) -> Union[Run, RunDisabled]
```

In an ML training pipeline, you could add `wandb.init()`
to the beginning of your training script as well as your evaluation
script, and each piece would be tracked as a run in W&B.

`wandb.init()` spawns a new background process to log data to a run, and it
also syncs data to wandb.ai by default, so you can see live visualizations.

Call `wandb.init()` to start a run before logging data with `wandb.log()`:

<!--yeadoc-test:init-method-log-->


```python
import wandb

wandb.init()
# ... calculate metrics, generate media
wandb.log({"accuracy": 0.9})
```

`wandb.init()` returns a run object, and you can also access the run object
via `wandb.run`:

<!--yeadoc-test:init-and-assert-global-->


```python
import wandb

run = wandb.init()

assert run is wandb.run
```

At the end of your script, we will automatically call `wandb.finish` to
finalize and cleanup the run. However, if you call `wandb.init` from a
child process, you must explicitly call `wandb.finish` at the end of the
child process.

For more on using `wandb.init()`, including detailed examples, check out our
[guide and FAQs](https://docs.wandb.ai/guides/track/launch).

| Arguments |  |
| :--- | :--- |
|  `project` |  (str, optional) The name of the project where you're sending the new run. If the project is not specified, the run is put in an "Uncategorized" project. |
|  `entity` |  (str, optional) An entity is a username or team name where you're sending runs. This entity must exist before you can send runs there, so make sure to create your account or team in the UI before starting to log runs. If you don't specify an entity, the run will be sent to your default entity. Change your default entity in [your settings](https://wandb.ai/settings) under "default location to create new projects". |
|  `config` |  (dict, argparse, absl.flags, str, optional) This sets `wandb.config`, a dictionary-like object for saving inputs to your job, like hyperparameters for a model or settings for a data preprocessing job. The config will show up in a table in the UI that you can use to group, filter, and sort runs. Keys should not contain `.` in their names, and values should be under 10 MB. If dict, argparse or absl.flags: will load the key value pairs into the `wandb.config` object. If str: will look for a yaml file by that name, and load config from that file into the `wandb.config` object. |
|  `save_code` |  (bool, optional) Turn this on to save the main script or notebook to W&B. This is valuable for improving experiment reproducibility and to diff code across experiments in the UI. By default this is off, but you can flip the default behavior to on in [your settings page](https://wandb.ai/settings). |
|  `group` |  (str, optional) Specify a group to organize individual runs into a larger experiment. For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate a model against different test sets. Group gives you a way to organize runs together into a larger whole, and you can toggle this on and off in the UI. For more details, see our [guide to grouping runs](https://docs.wandb.com/guides/runs/grouping). |
|  `job_type` |  (str, optional) Specify the type of run, which is useful when you're grouping runs together into larger experiments using group. For example, you might have multiple jobs in a group, with job types like train and eval. Setting this makes it easy to filter and group similar runs together in the UI so you can compare apples to apples. |
|  `tags` |  (list, optional) A list of strings, which will populate the list of tags on this run in the UI. Tags are useful for organizing runs together, or applying temporary labels like "baseline" or "production". It's easy to add and remove tags in the UI, or filter down to just runs with a specific tag. If you are resuming a run, its tags will be overwritten by the tags you pass to `wandb.init()`. If you want to add tags to a resumed run without overwriting its existing tags, use `run.tags += ["new_tag"]` after `wandb.init()`. |
|  `name` |  (str, optional) A short display name for this run, which is how you'll identify this run in the UI. By default, we generate a random two-word name that lets you easily cross-reference runs from the table to charts. Keeping these run names short makes the chart legends and tables easier to read. If you're looking for a place to save your hyperparameters, we recommend saving those in config. |
|  `notes` |  (str, optional) A longer description of the run, like a `-m` commit message in git. This helps you remember what you were doing when you ran this run. |
|  `dir` |  (str or pathlib.Path, optional) An absolute path to a directory where metadata will be stored. When you call `download()` on an artifact, this is the directory where downloaded files will be saved. By default, this is the `./wandb` directory. |
|  `resume` |  (bool, str, optional) Sets the resuming behavior. Options: `"allow"`, `"must"`, `"never"`, `"auto"` or `None`. Defaults to `None`. Cases: - `None` (default): If the new run has the same ID as a previous run, this run overwrites that data. - `"auto"` (or `True`): if the previous run on this machine crashed, automatically resume it. Otherwise, start a new run. - `"allow"`: if id is set with `init(id="UNIQUE_ID")` or `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run, wandb will automatically resume the run with that id. Otherwise, wandb will start a new run. - `"never"`: if id is set with `init(id="UNIQUE_ID")` or `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run, wandb will crash. - `"must"`: if id is set with `init(id="UNIQUE_ID")` or `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run, wandb will automatically resume the run with the id. Otherwise, wandb will crash. See [our guide to resuming runs](https://docs.wandb.com/guides/runs/resuming) for more. |
|  `reinit` |  (bool, optional) Allow multiple `wandb.init()` calls in the same process. (default: `False`) |
|  `magic` |  (bool, dict, or str, optional) The bool controls whether we try to auto-instrument your script, capturing basic details of your run without you having to add more wandb code. (default: `False`) You can also pass a dict, json string, or yaml filename. |
|  `config_exclude_keys` |  (list, optional) string keys to exclude from `wandb.config`. |
|  `config_include_keys` |  (list, optional) string keys to include in `wandb.config`. |
|  `anonymous` |  (str, optional) Controls anonymous data logging. Options: - `"never"` (default): requires you to link your W&B account before tracking the run, so you don't accidentally create an anonymous run. - `"allow"`: lets a logged-in user track runs with their account, but lets someone who is running the script without a W&B account see the charts in the UI. - `"must"`: sends the run to an anonymous account instead of to a signed-up user account. |
|  `mode` |  (str, optional) Can be `"online"`, `"offline"` or `"disabled"`. Defaults to online. |
|  `allow_val_change` |  (bool, optional) Whether to allow config values to change after setting the keys once. By default, we throw an exception if a config value is overwritten. If you want to track something like a varying learning rate at multiple times during training, use `wandb.log()` instead. (default: `False` in scripts, `True` in Jupyter) |
|  `force` |  (bool, optional) If `True`, this crashes the script if a user isn't logged in to W&B. If `False`, this will let the script run in offline mode if a user isn't logged in to W&B. (default: `False`) |
|  `sync_tensorboard` |  (bool, optional) Synchronize wandb logs from tensorboard or tensorboardX and save the relevant events file. (default: `False`) |
|  `monitor_gym` |  (bool, optional) Automatically log videos of environment when using OpenAI Gym. (default: `False`) See [our guide to this integration](https://docs.wandb.com/guides/integrations/openai-gym). |
|  `id` |  (str, optional) A unique ID for this run, used for resuming. It must be unique in the project, and if you delete a run you can't reuse the ID. Use the `name` field for a short descriptive name, or `config` for saving hyperparameters to compare across runs. The ID cannot contain the following special characters: `/\#?%:`. See [our guide to resuming runs](https://docs.wandb.com/guides/runs/resuming). |
|  `fork_from` |  (str, optional) A string with the format {run_id}?_step={step} describing a moment in a previous run to fork a new run from. Creates a new run that picks up logging history from the specified run at the specified moment. The target run must be in the current project. Example: `fork_from="my-run-id?_step=1234"`. |

#### Examples:

### Set where the run is logged

You can change where the run is logged, just like changing
the organization, repository, and branch in git:

```python
import wandb

user = "geoff"
project = "capsules"
display_name = "experiment-2021-10-31"

wandb.init(entity=user, project=project, name=display_name)
```

### Add metadata about the run to the config

Pass a dictionary-style object as the `config` keyword argument to add
metadata, like hyperparameters, to your run.

<!--yeadoc-test:init-set-config-->


```python
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
```

| Raises |  |
| :--- | :--- |
|  `Error` |  if some unknown or internal error happened during the run initialization. |
|  `AuthenticationError` |  if the user failed to provide valid credentials. |
|  `CommError` |  if there was a problem communicating with the WandB server. |
|  `UsageError` |  if the user provided invalid arguments. |
|  `KeyboardInterrupt` |  if user interrupts the run. |

| Returns |  |
| :--- | :--- |
|  A `Run` object. |
