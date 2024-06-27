---
description: >-
  W&B's Embedding Projector allows users to plot multi-dimensional embeddings on
  a 2D plane using common dimension reduction algorithms like PCA, UMAP, and
  t-SNE.
displayed_sidebar: default
---

# Embedding Projector

![](/images/weave/embedding_projector.png)

[Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) are used to represent objects (people, images, posts, words, etc...) with a list of numbers - sometimes referred to as a _vector_. In machine learning and data science use cases, embeddings can be generated using a variety of approaches across a range of applications. This page assumes the reader is familiar with embeddings and is interested in visually analyzing them inside of W&B.

## Embedding Examples

You can jump right into a [Live Interactive Demo Report](https://wandb.ai/timssweeney/toy\_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq) or run the code in this report from the [Example Colab](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm\_).

### Hello World

W&B allows you to log embeddings using the `wandb.Table` class. Consider the following example of 3 embeddings, each consisting of 5 dimensions:

```python
import wandb

wandb.init(project="embedding_tutorial")
embeddings = [
    # D1   D2   D3   D4   D5
    [0.2, 0.4, 0.1, 0.7, 0.5],  # embedding 1
    [0.3, 0.1, 0.9, 0.2, 0.7],  # embedding 2
    [0.4, 0.5, 0.2, 0.2, 0.1],  # embedding 3
]
wandb.log(
    {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
)
wandb.finish()
```

After running the above code, the W&B dashboard will have a new Table containing your data. You can select `2D Projection` from the upper right panel selector to plot the embeddings in 2 dimensions. Smart default will be automatically selected, which can be easily overridden in the configuration menu accessed by clicking the gear icon. In this example, we automatically use all 5 available numeric dimensions.

![](/images/app_ui/weave_hello_world.png)

### Digits MNIST

While the above example shows the basic mechanics of logging embeddings, typically you are working with many more dimensions and samples. Let's consider the MNIST Digits dataset ([UCI ML hand-written digits dataset](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)) made available via [SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load\_digits.html). This dataset has 1797 records, each with 64 dimensions. The problem is a 10 class classification use case. We can convert the input data to an image for visualization as well.

```python
import wandb
from sklearn.datasets import load_digits

wandb.init(project="embedding_tutorial")

# Load the dataset
ds = load_digits(as_frame=True)
df = ds.data

# Create a "target" column
df["target"] = ds.target.astype(str)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

# Create an "image" column
df["image"] = df.apply(
    lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df})
wandb.finish()
```

After running the above code, again we are presented with a Table in the UI. By selecting `2D Projection` we can configure the definition of the embedding, coloring, algorithm (PCA, UMAP, t-SNE), algorithm parameters, and even overlay (in this case we show the image when hovering over a point). In this particular case, these are all "smart defaults" and you should see something very similar with a single click on `2D Projection`. ([Click here to interact](https://wandb.ai/timssweeney/embedding\_tutorial/runs/k6guxhum?workspace=user-timssweeney) with this example).

![](/images/weave/embedding_projector.png)

## Logging Options

You can log embeddings in a number of different formats:

1. **Single Embedding Column:** Often your data is already in a "matrix"-like format. In this case, you can create a single embedding column - where the data type of the cell values can be `list[int]`, `list[float]`, or `np.ndarray`.
2. **Multiple Numeric Columns:** In the above two examples, we use this approach and create a column for each dimension. We currently accept python `int` or `float` for the cells.

![Single Embedding Column](/images/weave/logging_options.png)
![Many Numeric Columns](/images/weave/logging_option_image_right.png)

Furthermore, just like all tables, you have many options regarding how to construct the table:

1. Directly from a **dataframe** using `wandb.Table(dataframe=df)`
2. Directly from a **list of data** using `wandb.Table(data=[...], columns=[...])`
3. Build the table **incrementally row-by-row** (great if you have a loop in your code). Add rows to your table using `table.add_data(...)`
4. Add an **embedding column** to your table (great if you have a list of predictions in the form of embeddings): `table.add_col("col_name", ...)`
5. Add a **computed column** (great if you have a function or model you want to map over your table): `table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## Plotting Options

After selecting `2D Projection`, you can click the gear icon to edit the rendering settings. In addition to selecting the intended columns (see above), you can select an algorithm of interest (along with the desired parameters). Below you can see the parameters for UMAP and t-SNE respectively.

![](/images/weave/plotting_options_left.png) 
![](/images/weave/plotting_options_right.png)

:::info
Note: we currently downsample to a random subset of 1000 rows and 50 dimensions for all three algorithms.
:::
