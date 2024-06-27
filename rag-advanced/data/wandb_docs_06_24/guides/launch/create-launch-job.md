---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Create a Launch job
<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

Launch jobs are blueprints for reproducing W&B runs. Jobs are W&B Artifacts that capture the source code, dependencies, and inputs required to execute a workload. 

Create and run jobs with the `wandb launch` command.

:::info
To create a job without submitting it for execution, use the `wandb job create` command. See the [command reference docs](../../ref/cli/wandb-job/wandb-job-create.md) for more information.
:::


## Git jobs

You can create a Git-based job where code and other tracked assets are cloned from a certain commit, branch, or tag in a remote git repository with W&B Launch. Use the `--uri` or `-u` flag to specify the URI containing the code, along with optionally a `--build-context` flag to specify a subdirectory.

Run a "hello world" job from a git repository with the following command:

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

The command does the following:
1. Clones the [W&B Launch jobs repository](https://github.com/wandb/launch-jobs) to a temporary directory.
2. Creates a job named **hello-world-git** in the **hello** project. The job is associated with the commit at the head of the default branch of the repository.
3. Builds a container image from the `jobs/hello_world` directory and the `Dockerfile.wandb`.
4. Starts the container and runs `python job.py`.

To build a job from a specific branch or commit hash, append the `-g`, `--git-hash` argument. For a full list of arguments, run `wandb launch --help`.

### Remote URL format

The git remote associated with a Launch job can be either an HTTPS or an SSH URL. The URL type determines the protocol used to fetch job source code. 

| Remote URL Type| URL Format | Requirements for access and authentication |
| ----------| ------------------- | ------------------------------------------ |
| https      | `https://github.com/organization/repository.git`  | username and password to authenticate with the git remote |
| ssh        | `git@github.com:organization/repository.git` | ssh key to authenticate with the git remote |

Note that the exact URL format varies by hosting provider. Jobs created with `wandb launch --uri` will use the transfer protocol specified in the provided `--uri`.


## Code artifact jobs

Jobs can be created from any source code stored in a W&B Artifact. Use a local directory with the `--uri` or `-u` argument to create a new code artifact and job.

To get started, create an empty directory and add a Python script named `main.py` with the following content:

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

Add a file `requirements.txt` with the following content:

```txt
wandb>=0.17.1
```

Log the directory as a code artifact and launch a job with the following command:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

The preceding command does the following:
1. Logs the current directory as a code artifact named `hello-world-code`.
2. Creates a job named `hello-world-code` in the `launch-quickstart` project.
3. Builds a container image from the current directory and Launch's default Dockerfile. The default Dockerfile will install the `requirements.txt` file and set the entry point to `python main.py`.

## Image jobs

Alternatively, you can build jobs off of pre-made Docker images. This is useful when you already have an established build system for your ML code, or when you don't expect to adjust the code or requirements for the job but do want to experiment with hyperparameters or different infrastructure scales.

The image is pulled from a Docker registry and run with the specified entry point, or the default entry point if none is specified. Pass a full image tag to the `--docker-image` option to create and run a job from a Docker image.

To run a simple job from a pre-made image, use the following command:

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```


## Automatic job creation

W&B will automatically create and track a job for any run with tracked source code, even if that run was not created with Launch. Runs are considered to have tracked source code if any of the three following conditions are met:
- The run has an associated git remote and commit hash
- The run logged a code artifact (see [`Run.log_code`](../../ref/python/run.md#log_code) for more information)
- The run was executed in a Docker container with the `WANDB_DOCKER` environment variable set to an image tag

The Git remote URL is inferred from the local git repository if your Launch job is created automatically by a W&B run. 

### Launch job names

By default, W&B automatically generates a job name for you. The name is generated depending on how the job is created (GitHub, code artifact, or Docker image). Alternatively, you can define a Launch job's name with environment variables or with the W&B Python SDK.

The following table describes the job naming convention used by default based on job source:

| Source        | Naming convention                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |

Name your job with a W&B environment variable or with the W&B Python SDK

<Tabs
defaultValue="env_var"
values={[
{label: 'Environment variable', value: 'env_var'},
{label: 'W&B Python SDK', value: 'python_sdk'},
]}>
<TabItem value="env_var">

Set the `WANDB_JOB_NAME` environment variable to your preferred job name. For example:

```bash
WANDB_JOB_NAME=awesome-job-name
```

  </TabItem>
  <TabItem value="python_sdk">

Define the name of your job with `wandb.Settings`. Then pass this object when you initialize W&B with `wandb.init`. For example:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

  </TabItem>
</Tabs>

:::note
For docker image jobs, the version alias is automatically added as an alias to the job.
:::

## Containerization

Jobs are executed in a container. Image jobs use a pre-built Docker image, while Git and code artifact jobs require a container build step.

Job containerization can be customized with arguments to `wandb launch` and files within the job source code.

### Build context

The term build context refers to the tree of files and directories that are sent to the Docker daemon to build a container image. By default, Launch uses the root of the job source code as the build context. To specify a subdirectory as the build context, use the `--build-context` argument of `wandb launch` when creating and launching a job.

:::tip
The `--build-context` argument is particularly useful for working with Git jobs that refer to a monorepo with multiple projects. By specifying a subdirectory as the build context, you can build a container image for a specific project within the monorepo.

See the [example above](#git-jobs) for a demonstration of how to use the `--build-context` argument with the official W&B Launch jobs repository.
:::

### Dockerfile

The Dockerfile is a text file that contains instructions for building a Docker image. By default, Launch uses a default Dockerfile that installs the `requirements.txt` file. To use a custom Dockerfile, specify the path to the file with the `--dockerfile` argument of `wandb launch`.

The Dockerfile path is specified relative to the build context. For example, if the build context is `jobs/hello_world`, and the Dockerfile is located in the `jobs/hello_world` directory, the `--dockerfile` argument should be set to `Dockerfile.wandb`. See the [example above](#git-jobs) for a demonstration of how to use the `--dockerfile` argument with the official W&B Launch jobs repository.

### Requirements file

If no custom Dockerfile is provided, Launch will look in the build context for Python dependencies to install. If a `requirements.txt` file is found at the root of the build context, Launch will install the dependencies listed in the file. Otherwise, if a `pyproject.toml` file is found, Launch will install dependencies from the `project.dependencies` section. 