---
slug: /guides/integrations/w-and-b-for-julia
description: How to integrate W&B with Julia.
displayed_sidebar: default
---

# W&B for Julia

For those running machine learning experiments in the Julia programming language, a community contributor has created an unofficial set of Julia bindings called [wandb.jl](https://github.com/avik-pal/Wandb.jl) that you can use.

## Example

You can find examples [in the documentation](https://github.com/avik-pal/Wandb.jl/tree/main/docs/src/examples) on the wandb.jl repository. Their "Getting Started" example is here:

```julia
using Wandb, Dates, Logging

# Start a new run, tracking hyperparameters in config
lg = WandbLogger(project = "Wandb.jl",
                 name = "wandbjl-demo-$(now())",
                 config = Dict("learning_rate" => 0.01,
                               "dropout" => 0.2,
                               "architecture" => "CNN",
                               "dataset" => "CIFAR-100"))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)

# Simulating the training or evaluation loop
for x ∈ 1:50
    acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand() + get_config(lg, "dropout"))
    # Log metrics from your script to W&B
    @info "metrics" accuracy=acc loss=loss
end

# Finish the run
close(lg)
```
