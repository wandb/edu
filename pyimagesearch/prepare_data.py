"""
Made with love by tcapelle
@wandbcode{pis_course}
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import params
import wandb


def prepare_dataset():
    """
    This function prepares the dataset for training and evaluation.
    It splits the data into train, validation and test sets and logs the data split to W&B.
    """
    # start with a new wandb run
    run = wandb.init(
        project=params.PROJECT_NAME, entity=params.ENTITY, job_type="data_prep"
    )

    # get the path to the raw data folder
    raw_data_folder = Path(params.RAW_DATA_FOLDER)
    images_folder = raw_data_folder / params.IMAGES_FOLDER
    annotations_file = raw_data_folder / params.ANNOTATIONS_FILE

    # create a new artifact to store the dataset
    dataset_artifact = wandb.Artifact(params.ARTIFACT_NAME, type="dataset")
    dataset_artifact.add_dir(images_folder, name=params.IMAGES_FOLDER)
    dataset_artifact.add_file(annotations_file, name=params.ANNOTATIONS_FILE)

    # read annotations data and convert it to dataframes
    data = json.load(open(annotations_file, mode="r", encoding="utf-8"))
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
    for i, (_, test_idxs) in enumerate(cv.split(X, y, groups)):
        df["fold"].iloc[test_idxs] = i

    df["stage"] = df["fold"].apply(
        lambda x: "test" if x == 0 else ("valid" if x == 1 else "train")
    )
    df.to_csv("data_split.csv", index=False)

    # add csv containing processed data split into the artifact
    dataset_artifact.add_file("data_split.csv")

    # log artifact to W&B and finish the run
    run.log_artifact(dataset_artifact)
    run.finish()


if __name__=="__main__":
    prepare_dataset()