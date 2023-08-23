# W&B parameters
PROJECT_NAME = "wandb_course"
ENTITY = "pyimagesearch"
# ENTITY = None


# parameters for the dataset
RAW_DATA_FOLDER = "lemon-dataset"
IMAGES_FOLDER = "images"
ANNOTATIONS_FILE = "annotations/instances_default.json"
ARTIFACT_NAME = "lemon_data"
DATA_AT = f"{ENTITY}/{PROJECT_NAME}/{ARTIFACT_NAME}:v0"
