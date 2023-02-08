import os
from typing import List

import numpy as np
import pandas as pd
import skimage.io as io
from pycocotools.coco import COCO

import params
import wandb


def get_annotations(image: dict, coco_obj: COCO, categoryIds: List[dict]):
    # get the annotation ids and annotations corresponding to the give image
    annotationIds = coco_obj.getAnnIds(
        imgIds=image["id"], catIds=categoryIds, iscrowd=None
    )
    annotations = coco_obj.loadAnns(annotationIds)
    return annotations


def get_label(annotation: dict, categories: List[dict]):
    labels = [
        category["name"]
        for category in categories
        if category["id"] == annotation["category_id"]
    ]
    return labels[0]


def make_wandb_image(
    image: dict, coco_obj: COCO, categoryIds: List, categories: List[dict]
):
    """Helper function to create an annotated wandb.Image"""
    image_path = os.path.join(params.RAW_DATA_FOLDER, image["file_name"])
    image_array = io.imread(image_path)

    annotations = get_annotations(
        image=image, coco_obj=coco_obj, categoryIds=categoryIds
    )
    truth_box_data = [
        {
            "position": {
                "minX": annotation["bbox"][0],
                "minY": annotation["bbox"][1],
                "maxX": annotation["bbox"][0] + annotation["bbox"][2],
                "maxY": annotation["bbox"][1] + annotation["bbox"][3],
            },
            "class_id": annotation["category_id"],
            "box_caption": get_label(annotation=annotation, categories=categories),
            "domain": "pixel",
        }
        for annotation in annotations
    ]

    masks = [
        coco_obj.annToMask(annotation) * annotation["category_id"]
        for annotation in annotations
    ]
    mask = np.stack(masks).max(axis=0)  # arbitrary way to select a label...

    # We will log the images with the corresponding bounding boxes and masks
    # to a wandb.Table to visualize and do further EDA on the W&B workspace
    return wandb.Image(
        image_array,
        classes=categories,
        boxes={"ground_truth": {"box_data": truth_box_data}},
        masks={"ground_truth": {"mask_data": mask}},
    )


def is_mold(image: dict, coco_obj: COCO, categoryIds: List[dict]):
    """A function to filter mold catergory"""
    annotations = get_annotations(
        image=image, coco_obj=coco_obj, categoryIds=categoryIds
    )

    # 4 is id of mold category
    return 4 in [annotation["category_id"] for annotation in annotations]


def make_row(image: dict, coco_obj: COCO, categoryIds: List, categories: List[dict]):
    """Refactor each dictionary into a table row"""
    # get the filename and split the file name into file componenets
    file_name = image["file_name"]
    file_components = file_name.split("/")[-1].split(".")[0].split("_")

    # build image for the table
    wandb_image = make_wandb_image(
        image=image,
        coco_obj=coco_obj,
        categoryIds=categoryIds,
        categories=categories,
    )

    # whether the lemon has mold or not
    has_mold = is_mold(image=image, coco_obj=coco_obj, categoryIds=categoryIds)

    # build the row of the table and return it
    row = [
        wandb_image,
        *file_components,
        file_name,
        has_mold,
    ]
    return row


if __name__ == "__main__":
    # we will read the COCO object using the pycoco library,
    # this is a standard format for object detection/segmentation.
    coco_obj = COCO(params.ANNOTATIONS_FILE)

    # get category ids and categories
    categoryIds = coco_obj.getCatIds()
    categories = coco_obj.loadCats(categoryIds)

    # get image ids and images
    imageIds = coco_obj.getImgIds()
    images = coco_obj.loadImgs(imageIds)

    # upload 5 image to wandb
    images = images[0:5]

    # let's log the dataset as a Table, it takes around 5 minutes depending on your connection.
    columns = ["images", "ids", "n1", "n2", "n3", "n4", "file_name", "is_mold"]
    with wandb.init(
        project=params.PROJECT_NAME, entity=params.ENTITY, job_type="EDA"
    ) as run:
        df = pd.DataFrame(
            data=[
                make_row(
                    image=image,
                    coco_obj=coco_obj,
                    categoryIds=categoryIds,
                    categories=categories,
                )
                for image in images
            ],
            columns=columns,
        )
        run.log({"table_coco_sample": wandb.Table(dataframe=df)})
