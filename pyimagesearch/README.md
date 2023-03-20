# W&B + PyImageSearch MLOps Course

1. Download the dataset.
```shell
$ git clone -qq https://github.com/softwaremill/lemon-dataset.git
$ unzip -q lemon-dataset/data/lemon-dataset.zip
```

2. Run `python prepare_data.py` to prepare the dataset as an artifact and upload it to W&B.

3. Run `python eda.py` to do Exploratory Data Analysis on the dataset. We also upload the wandb Table from our analysis to W&B.

4. Train the model using `python train.py`

5. Evaluate the model using `python eval.py`