---
slug: /guides/integrations/stable-baselines-3
description: How to integrate W&B with Stable Baseline 3.
displayed_sidebar: default
---

# Stable Baselines 3

[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) \(SB3\) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. W&B's SB3 integration will: 

* Record metrics such as losses and episodic returns
* Upload videos of agents playing the games
* Save the trained model
* Log model's hyperparameters
* Log model gradient histograms

[Here is an example](https://wandb.ai/wandb/sb3/runs/1jyr6z10) of a  SB3 training run with W&B

## Log your SB3 Experiments in 2 lines of code

```python
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

![](@site/static/images/integrations/stable_baselines_demo.gif)

## WandbCallback Arguments

| Argument | Usage |
| :--- | :--- |
| `verbose` | The verbosity of sb3 output |
| `model_save_path` | Path to the folder where the model will be saved, The default value is \`None\` so the model is not logged |
| `model_save_freq` | Frequency to save the model |
| `gradient_save_freq` | Frequency to log gradient. The default value is 0 so the gradients are not logged |

## Basic Example

The W&B SB3 integration uses the logs output from TensorBoard to log your metrics 

```python
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env


env = DummyVecEnv([make_env])
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
```
