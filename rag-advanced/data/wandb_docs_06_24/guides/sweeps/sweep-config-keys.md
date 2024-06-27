---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sweep configuration options

A sweep configuration consists of nested key-value pairs. Use top-level keys within your sweep configuration to define qualities of your sweep search such as the parameters to search through ([`parameter`](./sweep-config-keys.md#parameters) key), the methodology to search the parameter space ([`method`](./sweep-config-keys.md#method) key), and more. 

The proceeding table lists top-level sweep configuration keys and a brief description. See the respective sections for more information about each key. 


| Top-level keys    | Description                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `program`         | (required) Training script to run.                                                                                            |
| `entity`          | Specify the entity for this sweep.                                                                                            |
| `project`         | Specify the project for this sweep.                                                                                           |
| `description`     | Text description of the sweep.                                                                                                |
| `name`            | The name of the sweep, displayed in the W&B UI.                                                                               |
| [`method`](#method) | (required) Specify the [search strategy](./define-sweep-configuration.md#configuration-keys).                               |
| [`metric`](#metric) | Specify the metric to optimize (only used by certain search strategies and stopping criteria).                              |
| [`parameters`](#parameters) | (required) Specify [parameters](define-sweep-configuration.md#parameters) bounds to search.                         |
| [`early_terminate`](#early_terminate) | Specify any [early stopping criteria](./define-sweep-configuration.md#early_terminate).                                 |
| [`command`](#command)         | Specify [command structure ](./define-sweep-configuration.md#command)for invoking and passing arguments to the training script. |
| `run_cap` | Specify a maximum number of runs in a sweep.                                                                                          |

See the [Sweep configuration](./sweep-config-keys.md) structure for more information on how to structure your sweep configuration.

<!-- ## `program`

##  `entity`

## `project`

## `description`

## `name` -->

## `metric`

Use the `metric` top-level sweep configuration key to specify the name, the goal, and the target metric to optimize.

|Key | Description |
| -------- | --------------------------------------------------------- |
| `name`   | Name of the metric to optimize.                           |
| `goal`   | Either `minimize` or `maximize` (Default is `minimize`).  |
| `target` | Goal value for the metric you are optimizing. The sweep does not create new runs when if or when a run reaches a target value that you specify. Active agents that have a run executing (when the run reaches the target) wait until the run completes before the agent stops creating new runs. |




## `parameters`
In your YAML file or Python script, specify `parameters` as a top level key. Within the `parameters` key, provide the name of a hyperparameter you want to optimize. Common hyperparameters include: learning rate, batch size, epochs, optimizers, and more. For each hyperparameter you define in your sweep configuration, specify one or more search constraints. 

The proceeding table shows supported hyperparameter search constraints. Based on your hyperparameter and use case, use one of the search constraints below to tell your sweep agent where (in the case of a distribution) or what (`value`, `values`, and so forth) to search or use.


| Search constraint | Description   |
| --------------- | ------------------------------------------------------------------------------ |
| `values`        | Specifies all valid values for this hyperparameter. Compatible with `grid`.    |
| `value`         | Specifies the single valid value for this hyperparameter. Compatible with `grid`.  |
| `distribution`  | Specify a probability [distribution](#distribution-options-for-random-and-bayesian-search). See the note following this table for information on default values. |
| `probabilities` | Specify the probability of selecting each element of `values` when using `random`.  |
| `min`, `max`    | (`int`or `float`) Maximum and minimum values. If `int`, for `int_uniform` -distributed hyperparameters. If `float`, for `uniform` -distributed hyperparameters. |
| `mu`            | (`float`) Mean parameter for `normal` - or `lognormal` -distributed hyperparameters. |
| `sigma`         | (`float`) Standard deviation parameter for `normal` - or `lognormal` -distributed hyperparameters. |
| `q`             | (`float`) Quantization step size for quantized hyperparameters.     |
| `parameters`    | Nest other parameters inside a root level parameter.    |


:::info
W&B sets the following distributions based on the following conditions if a [distribution](#distribution-options-for-random-and-bayesian-search) is not specified:
* `categorical` if you specify `values`
* `int_uniform` if you specify `max` and `min` as integers
* `uniform` if you specify `max` and `min` as floats
* `constant` if you provide a set to `value`
:::

## `method`
Specify the hyperparameter search strategy with the `method` key. There are three hyperparameter search strategies to choose from: grid, random, and Bayesian search. 
#### Grid search
Iterate over every combination of hyperparameter values. Grid search makes uninformed decisions on the set of hyperparameter values to use on each iteration. Grid search can be computationally costly.     

Grid search executes forever if it is searching within in a continuous search space.

#### Random search
Choose a random, uninformed, set of hyperparameter values on each iteration based on a distribution. Random search runs forever unless you stop the process from the command line, within your python script, or [the W&B App UI](./sweeps-ui.md).

Specify the distribution space with the metric key if you choose random (`method: random`) search.

#### Bayesian search
In contrast to [random](#random-search) and [grid](#grid-search) search, Bayesian models make informed decisions. Bayesian optimization uses a probabilistic model to decide which values to use through an iterative process of testing values on a surrogate function before evaluating the objective function. Bayesian search works well for small numbers of continuous parameters but scales poorly. For more information about Bayesian search, see the [Bayesian Optimization Primer paper](https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf).

<!-- There are different Bayesian optimization methods. W&B uses a Gaussian process to model the relationship between hyperparameters and the model metric. For more information, see this paper. [LINK] -->

Bayesian search runs forever unless you stop the process from the command line, within your python script, or [the W&B App UI](./sweeps-ui.md). 

### Distribution options for random and Bayesian search
Within the `parameter` key, nest the name of the hyperparameter. Next, specify the `distribution` key and specify a distribution for the value.

The proceeding tables lists distributions W&B supports.

| Value for `distribution` key  | Description            |
| ------------------------ | ------------------------------------ |
| `constant`               | Constant distribution. Must specify the constant value (`value`) to use.                    |
| `categorical`            | Categorical distribution. Must specify all valid values (`values`) for this hyperparameter. |
| `int_uniform`            | Discrete uniform distribution on integers. Must specify `max` and `min` as integers.     |
| `uniform`                | Continuous uniform distribution. Must specify `max` and `min` as floats.      |
| `q_uniform`              | Quantized uniform distribution. Returns `round(X / q) * q` where X is uniform. `q` defaults to `1`.|
| `log_uniform`            | Log-uniform distribution. Returns a value `X` between `exp(min)` and `exp(max)`such that the natural logarithm is uniformly distributed between `min` and `max`.   |
| `log_uniform_values`     | Log-uniform distribution. Returns a value `X` between `min` and `max` such that `log(`X`)` is uniformly distributed between `log(min)` and `log(max)`.     |
| `q_log_uniform`          | Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform`. `q` defaults to `1`. |
| `q_log_uniform_values`   | Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform_values`. `q` defaults to `1`.  |
| `inv_log_uniform`        | Inverse log uniform distribution. Returns `X`, where  `log(1/X)` is uniformly distributed between `min` and `max`. |
| `inv_log_uniform_values` | Inverse log uniform distribution. Returns `X`, where  `log(1/X)` is uniformly distributed between `log(1/max)` and `log(1/min)`.    |
| `normal`                 | Normal distribution. Return value is normally distributed with mean `mu` (default `0`) and standard deviation `sigma` (default `1`).|
| `q_normal`               | Quantized normal distribution. Returns `round(X / q) * q` where `X` is `normal`. Q defaults to 1.  |
| `log_normal`             | Log normal distribution. Returns a value `X` such that the natural logarithm `log(X)` is normally distributed with mean `mu` (default `0`) and standard deviation `sigma` (default `1`). |
| `q_log_normal`  | Quantized log normal distribution. Returns `round(X / q) * q` where `X` is `log_normal`. `q` defaults to `1`. |



## `early_terminate`

Use early termination (`early_terminate`) to stop poorly performing runs. If early termination occurs, W&B stops the current run before it creates a new run with a new set of hyperparameter values.

:::note
You must specify a stopping algorithm if you use `early_terminate`. Nest the `type` key within `early_terminate` within your sweep configuration.
:::


### Stopping algorithm

:::info
W&B currently supports [Hyperband](https://arxiv.org/abs/1603.06560) stopping algorithm. 
:::

[Hyperband](https://arxiv.org/abs/1603.06560) hyperparameter optimization evaluates if a program should stop or if it should to continue at one or more pre-set iteration counts, called *brackets*.

When a W&B run reaches a bracket, the sweep compares that run's metric to all previously reported metric values. The sweep terminates the run if the run's metric value is too high (when the goal is minimization) or if the run's metric is too low (when the goal is maximization).

Brackets are based on the number of logged iterations. The number of brackets corresponds to the number of times you log the metric you are optimizing. The iterations can correspond to steps, epochs, or something in between. The numerical value of the step counter is not used in bracket calculations.

:::info
Specify either `min_iter` or `max_iter` to create a bracket schedule.
:::


| Key        | Description                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | Specify the iteration for the first bracket                    |
| `max_iter` | Specify the maximum number of iterations.                      |
| `s`        | Specify the total number of brackets (required for `max_iter`) |
| `eta`      | Specify the bracket multiplier schedule (default: `3`).        |
| `strict`   | Enable 'strict' mode that prunes runs aggressively, more closely following the original Hyperband paper. Defaults to false. |



:::info
Hyperband checks which [W&B runs](../../ref/python/run.md) to end once every few minutes. The end run timestamp might differ from the specified brackets if your run or iteration are short.
:::

## `command` 

<!-- Agents created with [`wandb agent`](../../ref/cli/wandb-agent.md) receive a command in the following format by default: -->

Modify the format and contents with nested values within the `command` key. You can directly include fixed components such as filenames.

:::info
On Unix systems, `/usr/bin/env` ensures that the OS chooses the correct Python interpreter based on the environment.
:::

W&B supports the following macros for variable components of the command:

| Command macro              | Description                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | `/usr/bin/env` on Unix systems, omitted on Windows.                                                                                                                   |
| `${interpreter}`           | Expands to `python`.                                                                                                                                                  |
| `${program}`               | Training script filename specified by the sweep configuration `program` key.                                                                                          |
| `${args}`                  | Hyperparameters and their values in the form `--param1=value1 --param2=value2`.                                                                                       |
| `${args_no_boolean_flags}` | Hyperparameters and their values in the form `--param1=value1` except boolean parameters are in the form `--boolean_flag_param` when `True` and omitted when `False`. |
| `${args_no_hyphens}`       | Hyperparameters and their values in the form `param1=value1 param2=value2`.                                                                                           |
| `${args_json}`             | Hyperparameters and their values encoded as JSON.                                                                                                                     |
| `${args_json_file}`        | The path to a file containing the hyperparameters and their values encoded as JSON.                                                                                   |
| `${envvar}`                | A way to pass environment variables. `${envvar:MYENVVAR}` __ expands to the value of MYENVVAR environment variable. __                                               |

