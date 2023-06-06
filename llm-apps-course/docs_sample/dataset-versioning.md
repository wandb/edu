---
description: Guide to using Artifacts for dataset versioning
---

# Dataset Versioning

<head>
  <title>Version datasets with W&B Artifacts.</title>
</head>

W&B Artifacts help you save and organize machine learning datasets throughout a project's lifecycle.

### Common use cases

1. [**Version data seamlessly**](dataset-versioning.md#25c79f05-174e-4d35-abda-e5c238b8d6d6), without interrupting your workflow
2. [**Prepackage data splits**](dataset-versioning.md#7ccfb650-1f87-458c-a4e2-538138660292), like training, validation, and test sets
3. [**Iteratively refine datasets**](dataset-versioning.md#cee1428d-3b7a-4e1b-956b-e83170e7038f), without desynchronizing the team
4. [**Juggle multiple datasets**](dataset-versioning.md#4ba93c33-dd39-468b-8b3e-96c938bbd024), as in fine-tuning and domain adaptation
5. [**Visualize & share your data workflow**](dataset-versioning.md#57023a52-2c00-4b24-8e17-b193b40e216b), keeping all your work in one place

### Flexible tracking and hosting

Beyond these common scenarios, you can use core Artifact features to upload, version, alias, compare, and download data, supporting any custom dataset workflow on local or remote file systems, via S3, GCP, or https.

## Core Artifacts features

W&B Artifacts support dataset versioning through these basic features:

1. **Upload**: Start tracking and versioning any data (files or directories) with `run.log_artifact()`. You can also track datasets in a remote filesystem (e.g. cloud storage in S3 or GCP) by [reference](https://docs.wandb.ai/guides/artifacts/track-external-files), using a link or URI instead of the raw contents.
2. **Version**: Define an artifact by giving it a type (`"raw_data"`, `"preprocessed_data"`, `"balanced_data"`) and a name (`"imagenet_cats_10K"`). When you log the same name again, W&B automatically creates a new version of the artifact with the latest contents.
3. **Alias**: Set an alias like `"best"` or `"production"` to highlight the important versions in a lineage of artifacts.
4. **Compare**: Select any two versions to browse the contents side-by-side. We're also working on a tool for dataset visualization, [learn more here.](https://docs.wandb.ai/datasets-and-predictions)
5. **Download**: Obtain a local copy of the artifact or verify the contents by reference.

For more detail on these features, check out [Artifacts how it works](https://docs.wandb.ai/guides/artifacts#how-it-works).

## Version data seamlessly

The immediate value of W&B Artifacts is automatically versioning your data: tracking the contents of individual files and directories, in which you may add, remove, replace, or edit items. All these operations are traceable and reversible, reducing the cognitive load of handling data correctly. Once you create and upload an artifact, adding and logging new files will create a new version of that artifact. Behind the scenes, we'll diff the contents (deeply, via checksum) so that only the changes are uploaded — kind of like [git](https://www.atlassian.com/git/tutorials/what-is-git). You can view all the versions and the individual files in your browser, diff contents across versions, and download any specific version by index or alias (by default, `"latest"` is the alias of the most recent version). To keep data transfer lean and fast, wandb caches files.

Sample code:

```python
run = wandb.init(project="my_project")
my_data = wandb.Artifact("new_dataset", type="raw_data")
my_data.add_dir("path/to/my/data")
run.log_artifact(my_data)
```

In [this example](https://wandb.ai/stacey/mendeleev/artifacts/balanced\_data/inat\_80-10-10\_5K/ab79f01e007113280018), I have three datasets of 1K, 5K, and 10K items, and I can see and compare across the file names in subfolders (by data split or by class label).

![](/images/data_model_versioning/version_data_seamlessly.png)

## Prepackage data splits

As you iterate on your models and training schemes, you may want different slices of your data, varying the

* **number of items**: a smaller dataset to start as proof of concept/to iterate quickly, or several datasets of increasing size to see how much the model benefits from more data
* **train/val/test assignment and proportion**: a train/test split or a train/val/test split, with different proportions of items
* **per-class balance**: equalize label representation (N images for each of K classes) or follow the existing, unbalanced distribution of the data

or other factors specific to your task.

You can save and independently version all of these as artifacts and download them reliably by name across different machines, environments, team members, etc—without having to write down or remember where you last saved which version.

The Artifacts system avoids duplicating files wherever possible, so any files used across multiple versions will only be stored once!

Sample code:

```python
run = wandb.init(project="my_project")
my_data = wandb.Artifact("new_dataset", type="raw_data")

for dir in ["train", "val", "test"]:
	my_data.add_dir(dir)`

run.log_artifact(my_data)
```

View the [file contents →](https://wandb.ai/stacey/mendeleev/artifacts/balanced\_data/inat\_80-10-10\_5K/ab79f01e007113280018/files)

![](/images/data_model_versioning/version_data_seamlessly_2.png)

## Iteratively refine your data 

As you browse through your training data or add new batches of examples, you may notice issues like

* incorrect ground truth labels
* hard negatives or commonly misclassified examples
* problematic class imbalances

To clean up and refine your data, you might modify incorrect labels, add or remove files to address imbalances, or group hard negatives into a special test split. With Artifacts, once you've finished a batch of changes, you can then call `run.log_artifact()` to push the new version to the cloud. This will automatically update the artifact with a new version to reflect your changes, while preserving the lineage and history of previous changes.

You can tag artifact versions with custom aliases, take notes on what changed, store metadata alongside each version, and see which experiment runs use a particular version. That way, your entire team can be sure they're working with the `latest` or a `stable` version of the data, to taste.

We're also working to make this refinement process easier and more visual — check out our beta of Datasets & Predictions [here →](https://docs.wandb.ai/datasets-and-predictions)

### Versioning is automatic 

If an artifact changes, just re-run the same artifact creation script. In this case, imagine the `nature-data` directory contains two lists of photo ids, `animal-ids.txt` and `plant-ids.txt`. We edit `animals-ids.txt` to remove mislabeled examples. This script will capture the new version neatly — we'll checksum the artifact, identify that something changed, and track the new version. If nothing changes, we don't reupload any data (i.e. in this case, we don't reupload `plant-ids.txt`) or create a new version.

```python
run = wandb.init(job_type="dataset-creation")
artifact = wandb.Artifact('nature-dataset', type='dataset')
artifact.add_dir("nature-data")

# Edit the list of ids in one of the file to remove the mislabeled examples
# Let's say nature-photos contains "animal-ids.txt", which changes
# and "plant-ids", which does not

# Log that artifact, and we identify the changed file
run.log_artifact(artifact)
# Now you have a new version of the artifact, tracked in W&B
```

Give your datasets custom names and annotate them with notes or key-value pair metadata

![](/images/data_model_versioning/version_is_automatic.png)

## Juggle multiple datasets

Your task may require a more complex curriculum: perhaps pre-training on a subset of classes from [ImageNet](http://www.image-net.org) and fine-tuning on a custom dataset, say [iNaturalist](https://github.com/visipedia/inat\_comp/tree/master/2021) or your own photo collection. In domain adaptation, transfer learning, metalearning, and related tasks, you can save a different artifact for each data type or source to keep your experiments organized and more easily reproducible.

[Explore the graph interactively →](https://wandb.ai/stacey/mendeleev/artifacts/balanced\_data/inat\_80-10-10\_5K/ab79f01e007113280018/graph)

![](/images/data_model_versioning/juggle_multiple_datasets_1.png)

Multiple versions of balanced datasets of different sizes: 1K, 5K, and 10K and the corresponding artifact graph, showing training and inference runs on that data.

![](/images/data_model_versioning/juggle_multiple_datasets_2.png)

Several versions of raw data with 50 and 500 items total, from which a data\_split job creates two separate artifacts for "train" and "val" data.

## Visualize & easily share your data workflow

Artifacts let you see and formalize the flow of data through your model development scripts, whether for preprocessing, training, testing, analysis, or any other job type:

* choose **meaningful organizational types** for your artifacts and jobs: for data, this could be `train`, `val`, or `test`; for scripts this could be `preprocess`, `train`, `evaluate`, etc. You may also want to log other data as artifacts: a model's predictions on fixed validation data, samples of generated output, evaluation metrics, etc.
* **explore the artifact graph**: interact with all the connections between your code and data (input artifact(s) → script or job → output artifact(s)). Click "explode" on the compute graph to see all the versions for each artifact or all the runs of each script by job type. Click on individual nodes to see further details (file contents or code, annotations/metadata, config, timestamp, parent/child nodes, etc).
* **iterate confidently**: all your experimental script runs and data will be saved and versioned automatically, so you can focus on the core modeling task and not worry about where and when you saved which version of your dataset or code
* **share & replicate easily**: once you've integrated artifacts, you and your teammates can smoothly rerun the same workflow and pull from identical datasets (defaulting to the latest/best version), even to train in a different context/on different hardware

[Interactive example →](https://wandb.ai/wandb/arttest/artifacts/model/iv3\_trained/5334ab69740f9dda4fed/graph)

Simple compute graph example

![](/images/data_model_versioning/compute_graph_example.png)

More complex compute graph with predictions and evaluation results logged as artifacts

![](/images/data_model_versioning/compute_graph_example_advanced.png)

Simplified version of the compute graph, node grouped by artifact/job type ("Explode" off)

![](/images/data_model_versioning/compute_graph_simple.png)

Full details of each node: versions by artifact type and scripts runs by job type ("Explode" on)

![](/images/data_model_versioning/compute_graph_deatiled.png)

Details for a particular version of ResNet18: which training run produced it and which further runs loaded it for inference. These are deeply linked in each project so you can navigate the full graph.

![](/images/data_model_versioning/compute_graph_resnet_example.png)
