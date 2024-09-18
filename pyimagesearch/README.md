<p align="center">
  <img src="https://raw.githubusercontent.com/wandb/wandb/508982e50e82c54cbf0dd464a9959fee0e1740ad/.github/wb-logo-lightbg.png#gh-light-mode-only" width="600" alt="Weights & Biases"/>
  <img src="https://raw.githubusercontent.com/wandb/wandb/508982e50e82c54cbf0dd464a9959fee0e1740ad/.github/wb-logo-darkbg.png#gh-dark-mode-only" width="600" alt="Weights & Biases"/>
</p>

# W&B + PyImageSearch MLOps Course

This repo contains the code for the course: https://pyimagesearch.mykajabi.com/offers/LQSsX59C


1. Download the dataset.
```shell
$ git clone -qq https://github.com/softwaremill/lemon-dataset.git
$ unzip -q lemon-dataset/data/lemon-dataset.zip
```

1.5 Open the `params.py` file and change the wandb parameters to your own project.
```python
PROJECT_NAME = "wandb_course"
ENTITY = "pyimagesearch"  # just set this to your username
```

2. Run `python prepare_data.py` to prepare the dataset as an artifact and upload it to W&B.

3. Run `python eda.py` to do Exploratory Data Analysis on the dataset. We also upload the wandb Table from our analysis to W&B.

4. Train the model using `python train.py`

5. Evaluate the model using `python eval.py`
