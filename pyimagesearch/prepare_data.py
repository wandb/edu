import os
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import params

import wandb

# start with a new wandb run
run = wandb.init(
    project=params.PROJECT_NAME, entity=params.ENTITY, job_type="data_prep"
)

# create a new artifact to store and version our data
path = Path(params.RAW_DATA_FOLDER)
data_at = wandb.Artifact(params.DATA_AT, type="dataset")
data_at.add_dir(path / "images", name="images")
data_at.add_file(params.ANNOTATIONS_FILE, name="annotations/instances_default.json")

# read annotations data and convert it to dataframes
data = json.load(open(params.ANNOTATIONS_FILE))
annotations = pd.DataFrame.from_dict(data["annotations"])
images = pd.DataFrame.from_dict(data["images"])

# wrangle data to give us a binary classification target and fruit ids based on our EDA
df = (
    annotations[["image_id", "category_id"]]
    .groupby("image_id")["category_id"]
    .apply(lambda x: list(set(x)))
    .reset_index()
)
df["mold"] = df["category_id"].apply(lambda x: 4 in x)
df = pd.merge(df, images[["id", "file_name"]], left_on="image_id", right_on="id")
del df["id"]
df["fruit_id"] = df["file_name"].apply(lambda x: x.split("/")[1].split("_")[0])

# TRAIN / VALIDATION / TEST SPLIT
df["fold"] = -1
X = df.index.values
y = df.mold.values  # stratify by our target column
groups = df.fruit_id.values  # group individual fruit to avoid leakage

cv = StratifiedGroupKFold(n_splits=10, random_state=42, shuffle=True)
for i, (train_idxs, test_idxs) in enumerate(cv.split(X, y, groups)):
    df["fold"].iloc[test_idxs] = i

df["stage"] = df["fold"].apply(
    lambda x: "test" if x == 0 else ("valid" if x == 1 else "train")
)
df.to_csv("data_split.csv", index=False)

# add csv containing processed data split into the artifact
data_at.add_file("data_split.csv")

# log artifact to W&B and finish the run
run.log_artifact(data_at)
run.finish()
