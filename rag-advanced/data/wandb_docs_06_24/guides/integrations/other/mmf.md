---
slug: /guides/integrations/mmf
description: How to integrate W&B with Meta AI's MMF.
displayed_sidebar: default
---

# MMF

The `WandbLogger` class in [Meta AI's MMF](https://github.com/facebookresearch/mmf) library will enable Weights & Biases to log the training/validation metrics, system (GPU and CPU) metrics, model checkpoints and configuration parameters.

### Current features

The following features are currently supported by the `WandbLogger` in MMF:

* Training & Validation metrics
* Learning Rate over time
* Model Checkpoint saving to W&B Artifacts
* GPU and CPU system metrics
* Training configuration parameters

### Config parameters

The following options are available in MMF config to enable and customize the wandb logging:

```
training:
    wandb:
        enabled: true
        
        # An entity is a username or team name where you're sending runs.
        # By default it will log the run to your user account.
        entity: null
        
        # Project name to be used while logging the experiment with wandb
        project: mmf
        
        # Experiment/ run name to be used while logging the experiment
        # under the project with wandb. The default experiment name
        # is: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # Turn on model checkpointing, saving checkpoints to W&B Artifacts
        log_model_checkpoint: true
        
        # Additional argument values that you want to pass to wandb.init(). 
        # Check out the documentation at https://docs.wandb.ai/ref/python/init
        # to see what arguments are available, such as:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # To change the path to the directory where wandb metadata would be 
    # stored (Default: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```
