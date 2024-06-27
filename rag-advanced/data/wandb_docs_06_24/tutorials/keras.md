# Keras

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb)

Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />


This colab notebook introduces the `WandbMetricsLogger` callback. Use this callback for [Experiment Tracking](https://docs.wandb.ai/guides/track). It will log your training and validation metrics along with system metrics to Weights and Biases.


# 🌴 Setup and Installation

First, let us install the latest version of Weights and Biases. We will then authenticate this colab instance to use W&B.


```python
!pip install -qq -U wandb
```


```python
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# Weights and Biases related imports
import wandb
from wandb.integration.keras import WandbMetricsLogger
```

If this is your first time using W&B or you are not logged in, the link that appears after running `wandb.login()` will take you to sign-up/login page. Signing up for a [free account](https://wandb.ai/signup) is as easy as a few clicks.


```python
wandb.login()
```

# 🌳 Hyperparameters

Use of proper config system is a recommended best practice for reproducible machine learning. We can track the hyperparameters for every experiment using W&B. In this colab we will be using simple Python `dict` as our config system.


```python
configs = dict(
    num_classes = 10,
    shuffle_buffer = 1024,
    batch_size = 64,
    image_size = 28,
    image_channels = 1,
    earlystopping_patience = 3,
    learning_rate = 1e-3,
    epochs = 10
)
```

# 🍁 Dataset

In this colab, we will be using [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) dataset from TensorFlow Dataset catalog. We aim to build a simple image classification pipeline using TensorFlow/Keras.


```python
train_ds, valid_ds = tfds.load('fashion_mnist', split=['train', 'test'])
```


```python
AUTOTUNE = tf.data.AUTOTUNE


def parse_data(example):
    # Get image
    image = example["image"]
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Get label
    label = example["label"]
    label = tf.one_hot(label, depth=configs["num_classes"])

    return image, label


def get_dataloader(ds, configs, dataloader_type="train"):
    dataloader = ds.map(parse_data, num_parallel_calls=AUTOTUNE)

    if dataloader_type=="train":
        dataloader = dataloader.shuffle(configs["shuffle_buffer"])
      
    dataloader = (
        dataloader
        .batch(configs["batch_size"])
        .prefetch(AUTOTUNE)
    )

    return dataloader
```


```python
trainloader = get_dataloader(train_ds, configs)
validloader = get_dataloader(valid_ds, configs, dataloader_type="valid")
```

# 🎄 Model


```python
def get_model(configs):
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
    backbone.trainable = False

    inputs = layers.Input(shape=(configs["image_size"], configs["image_size"], configs["image_channels"]))
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3,3), padding="same")(resize)
    preprocess_input = tf.keras.applications.mobilenet.preprocess_input(neck)
    x = backbone(preprocess_input)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(configs["num_classes"], activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)
```


```python
tf.keras.backend.clear_session()
model = get_model(configs)
model.summary()
```

# 🌿 Compile Model


```python
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')]
)
```

# 🌻 Train


```python
# Initialize a W&B run
run = wandb.init(
    project = "intro-keras",
    config = configs
)

# Train your model
model.fit(
    trainloader,
    epochs = configs["epochs"],
    validation_data = validloader,
    callbacks = [WandbMetricsLogger(log_freq=10)] # Notice the use of WandbMetricsLogger here
)

# Close the W&B run
run.finish()
```
