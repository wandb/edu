---
slug: /guides/integrations/hydra
description: How to integrate W&B with Hydra.
displayed_sidebar: default
---

# Hydra

> [Hydra](https://hydra.cc) is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

You can continue to use Hydra for configuration management while taking advantage of the power of W&B.

## Track metrics

Track your metrics as normal with `wandb.init` and `wandb.log` . Here, `wandb.entity` and `wandb.project` are defined within a hydra configuration file.

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
```

## Track Hyperparameters

Hydra uses [omegaconf](https://omegaconf.readthedocs.io/en/2.1\_branch/) as the default way to interface with configuration dictionaries. `OmegaConf`'s dictionary are not a subclass of primitive dictionaries so directly passing Hydra's `Config` to `wandb.config` leads to unexpected results on the dashboard. It's necessary to convert `omegaconf.DictConfig` to the primitive `dict` type before passing to `wandb.config`.

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    wandb.log({"loss": loss})
    model = Model(**wandb.config.model.configs)
```

### Troubleshooting Multiprocessing

If your process hangs when started, this may be caused by [this known issue](../../track/log/distributed-training.md). To solve this, try to changing wandb's multiprocessing protocol either by adding an extra settings parameter to \`wandb.init\` as:

```
wandb.init(settings=wandb.Settings(start_method="thread"))
```

or by setting a global environment variable from your shell:

```
$ export WANDB_START_METHOD=thread
```

## Optimize Hyperparameters

[W&B Sweeps](../../sweeps/intro.md) is a highly scalable hyperparameter search platform, which provides interesting insights and visualization about W&B experiments with minimal requirements code real-estate. Sweeps integrates seamlessly with Hydra projects with no-coding requirements. The only thing needed is a configuration file describing the various parameters to sweep over as normal.

A simple example `sweep.yaml` file would be:

```yaml
program: main.py
method: bayes
metric:
  goal: maximize
  name: test/accuracy
parameters:
  dataset:
    values: [mnist, cifar10]

command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}
```

Invoke the sweep with:

`wandb sweep sweep.yaml`\
``\
``Once you call this, W&B automatically creates a sweep inside your project and returns a `wandb agent` command for you to run on each machine you want to run your sweep.

#### Passing parameters not present in Hydra defaults <a href="#pitfall-3-sweep-passing-parameters-not-present-in-defaults" id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra supports passing extra parameters through the command line which aren't present in the default configuration file, by using a `+` before command. For example, you can pass an extra parameter with some value by simply calling:

```
$ python program.py +experiment=some_experiment
```

You cannot sweep over such `+` configurations similar to what one does while configuring [Hydra Experiments](https://hydra.cc/docs/patterns/configuring\_experiments/). To work around this, you can initialize the experiment parameter with a default empty file and use W&B Sweep to override those empty configs on each call. For more information, read [**this W&B Report**](http://wandb.me/hydra)**.**
