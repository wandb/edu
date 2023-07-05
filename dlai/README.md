[![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/capecape/dlai_diffusion)

# DLAI with W&B 😎

We instrument various notebooks from the generative AI course with W&B to track metrics, hyperparameters, and artifacts.

- [00_intro](00_intro.ipynb) In this notebooks we learn about using wegiths and biases! we train a simple classifier on the Sprites datasets and log the results to W&B.
- [01_diffusion_training](01_diffusion_training.ipynb) In this notebook we train a diffusion model to generate images from the Sprites dataset. We log the training metrics to W&B. We sample from the model and log the images to W&B.
- [02_diffusion_sampling](02_diffusion_sampling.ipynb) In this notebook we sample from the trained model and log the images to W&B. We compare different sampling methods and log the results.
- [03 LLM evaluation and debugging](03_llm_eval.ipynb) In this notebook we generate character names using LLMs and use W&B autologgers and Tracer to evaluate and debug our generations.
- [04 WIP]() We are planning to add a CPU-based LLM finetuning notebook with a small LLM finetuned for generating names

The W&B dashboard: https://wandb.ai/capecape/dlai_diffusion