import os
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as io
from pycocotools.coco import COCO
from fastprogress import progress_bar

import params
import wandb


# we define some helper functions to load the annotations and labels
def get_anns(img):
    annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
    return coco.loadAnns(annIds)


def get_label(ann):
    return [cat["name"] for cat in cats if cat["id"] == ann["category_id"]][0]


# We will log the images with the corresponding bounding boxes and masks
# to a wandb.Table to visualize and do further EDA on the W&B workspace
def make_wandb_image(img, dataset_folder):
    "Helper function to create an annotated wandb.Image"
    pth = os.path.join(dataset_folder, img["file_name"])
    img_array = io.imread(pth)
    anns = get_anns(img)

    truth_box_data = [
        {
            "position": {
                "minX": ann["bbox"][0],
                "minY": ann["bbox"][1],
                "maxX": ann["bbox"][0] + ann["bbox"][2],
                "maxY": ann["bbox"][1] + ann["bbox"][3],
            },
            "class_id": ann["category_id"],
            "box_caption": get_label(ann),
            "domain": "pixel",
        }
        for ann in anns
    ]

    masks = [coco.annToMask(ann) * ann["category_id"] for ann in anns]
    mask = np.stack(masks).max(axis=0)  # arbitrary way to select a label...
    return wandb.Image(
        img_array,
        classes=cats,
        boxes={"ground_truth": {"box_data": truth_box_data}},
        masks={"ground_truth": {"mask_data": mask}},
    )


# we are interested in mold, so let's create a function to filter this catergory
def is_mold(img):
    anns = get_anns(img)
    # 4 is id of mold category
    return 4 in [x["category_id"] for x in anns]


# The filename contains the different info about each image separated by an underscore.
# We will log this information in separate columns
def make_row(img, dataset_folder):
    "Refactor of each table row"
    fname = img["file_name"].split("/")[1].split(".")[0]
    return [
        make_wandb_image(img, dataset_folder),
        *fname.split("_"),
        img["file_name"],
        is_mold(img),
    ]


if __name__ == "__main__":
    dataset_folder = Path(params.RAW_DATA_FOLDER)

    # we will read the COCO object using the pycoco library,
    # this is a standard format for object detection/segmentation.
    coco = COCO(os.path.join(dataset_folder, params.ANNOTATIONS_FILE))

    # get categories
    cats = coco.loadCats(coco.getCatIds())
    catIds = coco.getCatIds()

    # get image ids
    imgIds = coco.getImgIds()
    imgs = coco.loadImgs(imgIds)
    
    with wandb.init(
        project=params.PROJECT_NAME, entity=params.ENTITY, job_type="EDA"
        ) as run:
        # Let's log the dataset as a Table, it takes around 5 minutes depending on your connection.
        # imgs = imgs[0:5]  # uncomment to log a sample only
        df = pd.DataFrame(
            data=[make_row(img, dataset_folder) for img in progress_bar(imgs)],
            columns="imgs,ids,n1,n2,n3,n4,file_name,is_mold".split(","),
        )
        run.log({"table_coco_sample": wandb.Table(dataframe=df)})
