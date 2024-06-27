# BoundingBoxes2D

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L16-L293' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Format images with 2D bounding box overlays for logging to W&B.

```python
BoundingBoxes2D(
    val: dict,
    key: str
) -> None
```

| Arguments |  |
| :--- | :--- |
|  `val` |  (dictionary) A dictionary of the following form: box_data: (list of dictionaries) One dictionary for each bounding box, containing: position: (dictionary) the position and size of the bounding box, in one of two formats Note that boxes need not all use the same format. {"minX", "minY", "maxX", "maxY"}: (dictionary) A set of coordinates defining the upper and lower bounds of the box (the bottom left and top right corners) {"middle", "width", "height"}: (dictionary) A set of coordinates defining the center and dimensions of the box, with "middle" as a list [x, y] for the center point and "width" and "height" as numbers domain: (string) One of two options for the bounding box coordinate domain null: By default, or if no argument is passed, the coordinate domain is assumed to be relative to the original image, expressing this box as a fraction or percentage of the original image. This means all coordinates and dimensions passed into the "position" argument are floating point numbers between 0 and 1. "pixel": (string literal) The coordinate domain is set to the pixel space. This means all coordinates and dimensions passed into "position" are integers within the bounds of the image dimensions. class_id: (integer) The class label id for this box scores: (dictionary of string to number, optional) A mapping of named fields to numerical values (float or int), can be used for filtering boxes in the UI based on a range of values for the corresponding field box_caption: (string, optional) A string to be displayed as the label text above this box in the UI, often composed of the class label, class name, and/or scores class_labels: (dictionary, optional) A map of integer class labels to their readable class names |
|  `key` |  (string) The readable name or id for this set of bounding boxes (e.g. predictions, ground_truth) |

#### Examples:

### Log bounding boxes for a single image

<!--yeadoc-test:boundingbox-2d-->


```python
import numpy as np
import wandb

wandb.init()
image = np.random.randint(low=0, high=256, size=(200, 300, 3))

class_labels = {0: "person", 1: "car", 2: "road", 3: "building"}

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # one box expressed in the default relative/fractional domain
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 1,
                    "box_caption": class_labels[1],
                    "scores": {"acc": 0.2, "loss": 1.2},
                },
                {
                    # another box expressed in the pixel domain
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                },
                # Log as many boxes an as needed
            ],
            "class_labels": class_labels,
        }
    },
)

wandb.log({"driving_scene": img})
```

### Log a bounding box overlay to a Table

<!--yeadoc-test:bb2d-image-with-labels-->


```python
import numpy as np
import wandb

wandb.init()
image = np.random.randint(low=0, high=256, size=(200, 300, 3))

class_labels = {0: "person", 1: "car", 2: "road", 3: "building"}

class_set = wandb.Classes(
    [
        {"name": "person", "id": 0},
        {"name": "car", "id": 1},
        {"name": "road", "id": 2},
        {"name": "building", "id": 3},
    ]
)

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # one box expressed in the default relative/fractional domain
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 1,
                    "box_caption": class_labels[1],
                    "scores": {"acc": 0.2, "loss": 1.2},
                },
                {
                    # another box expressed in the pixel domain
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                },
                # Log as many boxes an as needed
            ],
            "class_labels": class_labels,
        }
    },
    classes=class_set,
)

table = wandb.Table(columns=["image"])
table.add_data(img)
wandb.log({"driving_scene": table})
```

## Methods

### `type_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L215-L217)

```python
@classmethod
type_name() -> str
```

### `validate`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/helper_types/bounding_boxes_2d.py#L219-L276)

```python
validate(
    val: dict
) -> bool
```
