# Import & Export API

<!-- Insert buttons and diff -->


<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


## Classes

[`class Api`](./api.md): Used for querying the wandb server.

[`class File`](./file.md): File is a class associated with a file saved by wandb.

[`class Files`](./files.md): An iterable collection of `File` objects.

[`class Job`](./job.md)

[`class Project`](./project.md): A project is a namespace for runs.

[`class Projects`](./projects.md): An iterable collection of `Project` objects.

[`class QueuedRun`](./queuedrun.md): A single queued run associated with an entity and project. Call `run = queued_run.wait_until_running()` or `run = queued_run.wait_until_finished()` to access the run.

[`class Run`](./run.md): A single run associated with an entity and project.

[`class RunQueue`](./runqueue.md)

[`class Runs`](./runs.md): An iterable collection of runs associated with a project and optional filter.

[`class Sweep`](./sweep.md): A set of runs associated with a sweep.
