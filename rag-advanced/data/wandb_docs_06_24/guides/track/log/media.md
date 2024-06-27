---
description: Log rich media, from 3D point clouds and molecules to HTML and histograms
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Log Media & Objects

We support images, video, audio, and more. Log rich media to explore your results and visually compare your runs, models, and datasets. Read on for examples and how-to guides.

:::info
Looking for reference docs for our media types? You want [this page](../../../ref/python/data-types/README.md).
:::

<!-- {% embed url="https://www.youtube.com/watch?v=96MxRvx15Ts" %} -->

:::info
You can see working code to log all of these media objects in [this Colab Notebook](http://wandb.me/media-colab), check out what the results look like on wandb.ai [here](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA), and follow along with a video tutorial, linked above.
:::

## Images

Log images to track inputs, outputs, filter weights, activations, and more!

![Inputs and outputs of an autoencoder network performing in-painting.](/images/track/log_images.png)

Images can be logged directly from NumPy arrays, as PIL images, or from the filesystem.

:::info
It's recommended to log fewer than 50 images per step to prevent logging from becoming a bottleneck during training and image loading from becoming a bottleneck when viewing results.
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: 'Logging Arrays as Images', value: 'arrays'},
    {label: 'Logging PIL Images', value: 'pil_images'},
    {label: 'Logging Images from Files', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

Provide arrays directly when constructing images manually, e.g. using [`make_grid` from `torchvision`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make\_grid).

Arrays are converted to png using [Pillow](https://pillow.readthedocs.io/en/stable/index.html).

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

We assume the image is gray scale if the last dimension is 1, RGB if it's 3, and RGBA if it's 4. If the array contains floats, we convert them to integers between `0` and `255`. If you want to normalize your images differently, you can specify the [`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes) manually or just supply a [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html), as described in the "Logging PIL Images" tab of this panel.
  </TabItem>
  <TabItem value="pil_images">

For full control over the conversion of arrays to images, construct the [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) yourself and provide it directly.

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```
  </TabItem>
  <TabItem value="images_files">
For even more control, create images however you like, save them to disk, and provide a filepath.

```python
im = PIL.fromarray(...)
rgb_im = im.convert("RGB")
rgb_im.save("myimage.jpg")

wandb.log({"example": wandb.Image("myimage.jpg")})
```
  </TabItem>
</Tabs>

## Image Overlays

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'Segmentation Masks', value: 'segmentation_masks'},
    {label: 'Bounding Boxes', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

Log semantic segmentation masks and interact with them (altering opacity, viewing changes over time, and more) via the W&B UI.

![Interactive mask viewing in the W&B UI.](/images/track/semantic_segmentation.gif)

To log an overlay, you'll need to provide a dictionary with the following keys and values to the `masks` keyword argument of `wandb.Image`:

* one of two keys representing the image mask:
  * `"mask_data"`: a 2D NumPy array containing an integer class label for each pixel
  * `"path"`: (string) a path to a saved image mask file
* `"class_labels"`: (optional) a dictionary mapping the integer class labels in the image mask to their readable class names

To log multiple masks, log a mask dictionary with multiple keys, as in the code snippet below.

[See a live example →](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[Sample code →](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix\_P4J)

```python
mask_data = np.array([[1, 2, 2, ..., 2, 2, 1], ...])

class_labels = {1: "tree", 2: "car", 3: "road"}

mask_img = wandb.Image(
    image,
    masks={
        "predictions": {"mask_data": mask_data, "class_labels": class_labels},
        "ground_truth": {
            # ...
        },
        # ...
    },
)
```
  </TabItem>
  <TabItem value="bounding_boxes">
Log bounding boxes with images, and use filters and toggles to dynamically visualize different sets of boxes in the UI.

![](@site/static/images/track/bb-docs.jpeg)

[See a live example →](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

To log a bounding box, you'll need to provide a dictionary with the following keys and values to the boxes keyword argument of `wandb.Image`:

* `box_data`: a list of dictionaries, one for each box. The box dictionary format is described below.
  * `position`: a dictionary representing the position and size of the box in one of two formats, as described below. Boxes need not all use the same format.
    * _Option 1:_ `{"minX", "maxX", "minY", "maxY"}`. Provide a set of coordinates defining the upper and lower bounds of each box dimension.
    * _Option 2:_ `{"middle", "width", "height"}`. Provide a set of coordinates specifying the `middle` coordinates as `[x,y]`, and `width` and `height` as scalars.
  * `class_id`: an integer representing the class identity of the box. See `class_labels` key below.
  * `scores`: a dictionary of string labels and numeric values for scores. Can be used for filtering boxes in the UI.
  * `domain`: specify the units/format of the box coordinates. **Set this to "pixel"** if the box coordinates are expressed in pixel space (i.e. as integers within the bounds of the image dimensions). By default, the domain is assumed to be a fraction/percentage of the image (a floating point number between 0 and 1).
  * `box_caption`: (optional) a string to be displayed as the label text on this box
* `class_labels`: (optional) A dictionary mapping `class_id`s to strings. By default we will generate class labels `class_0`, `class_1`, etc.

Check out this example:

```python
class_id_to_label = {
    1: "car",
    2: "road",
    3: "building",
    # ...
}

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # one box expressed in the default relative/fractional domain
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # another box expressed in the pixel domain
                    # (for illustration purposes only, all boxes are likely
                    # to be in the same domain/format)
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # Log as many boxes an as needed
                }
            ],
            "class_labels": class_id_to_label,
        },
        # Log each meaningful group of boxes with a unique key name
        "ground_truth": {
            # ...
        },
    },
)

wandb.log({"driving_scene": img})
```
  </TabItem>
</Tabs>

## Image Overlays in Tables

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'Segmentation Masks', value: 'segmentation_masks'},
    {label: 'Bounding Boxes', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

![Interactive Segmentation Masks in Tables](/images/track/Segmentation_Masks.gif)

To log Segmentation Masks in tables, you will need to provide a `wandb.Image` object for each row in the table.

An example is provided in the Code snippet below:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, label in zip(ids, images, labels):
    mask_img = wandb.Image(
        img,
        masks={
            "prediction": {"mask_data": label, "class_labels": class_labels}
            # ...
        },
    )

    table.add_data(id, img)

wandb.log({"Table": table})
```
  </TabItem>
  <TabItem value="bounding_boxes">


![Interactive Bounding Boxes in Tables](/images/track/Bounding_Boxes.gif)

To log Images with Bounding Boxes in tables, you will need to provide a `wandb.Image` object for each row in the table.

An example is provided in the code snippet below:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, boxes in zip(ids, images, boxes_set):
    box_img = wandb.Image(
        img,
        boxes={
            "prediction": {
                "box_data": [
                    {
                        "position": {
                            "minX": box["minX"],
                            "minY": box["minY"],
                            "maxX": box["maxX"],
                            "maxY": box["maxY"],
                        },
                        "class_id": box["class_id"],
                        "box_caption": box["caption"],
                        "domain": "pixel",
                    }
                    for box in boxes
                ],
                "class_labels": class_labels,
            }
        },
    )
```
  </TabItem>
</Tabs>

## Histograms

<Tabs
  defaultValue="histogram_logging"
  values={[
    {label: 'Basic Histogram Logging', value: 'histogram_logging'},
    {label: 'Flexible Histogram Logging', value: 'flexible_histogram'},
    {label: 'Histograms in Summary', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">
  
If a sequence of numbers (e.g. list, array, tensor) is provided as the first argument, we will construct the histogram automatically by calling `np.histogram`. Note that all arrays/tensors are flattened. You can use the optional `num_bins` keyword argument to override the default of `64` bins. The maximum number of bins supported is `512`.

In the UI, histograms are plotted with the training step on the x-axis, the metric value on the y-axis, and the count represented by color, to ease comparison of histograms logged throughout training. See the "Histograms in Summary" tab of this panel for details on logging one-off histograms.

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![Gradients for the discriminator in a GAN.](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

If you want more control, call `np.histogram` and pass the returned tuple to the `np_histogram` keyword argument.

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # if only in summary, only visible on overview tab
    {"final_logits": wandb.Histogram(logits)}
)
```
  </TabItem>
</Tabs>

If histograms are in your summary they will appear on the Overview tab of the [Run Page](../../app/pages/run-page.md). If they are in your history, we plot a heatmap of bins over time on the Charts tab.

## 3D Visualizations

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3D Object', value: '3d_object'},
    {label: 'Point Clouds', value: 'point_clouds'},
    {label: 'Molecules', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

Log files in the formats `'obj', 'gltf', 'glb', 'babylon', 'stl', 'pts.json'`, and we will render them in the UI when your run finishes.

```python
wandb.log(
    {
        "generated_samples": [
            wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb")),
        ]
    }
)
```

![Ground truth and prediction of a headphones point cloud](/images/track/ground_truth_prediction_of_3d_point_clouds.png)

[See a live example →](https://app.wandb.ai/nbaryd/SparseConvNet-examples\_3d\_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)
  </TabItem>
  <TabItem value="point_clouds">

Log 3D point clouds and Lidar scenes with bounding boxes. Pass in a NumPy array containing coordinates and colors for the points to render. In the UI, we truncate to 300,000 points.

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

Three different shapes of NumPy arrays are supported for flexible color schemes.

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c is a category` in the range `[1, 14]` (Useful for segmentation)
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b` are values in the range `[0,255]`for red, green, and blue color channels.

Here's an example of logging code below:

* `points`is a NumPy array with the same format as the simple point cloud renderer shown above.
* `boxes` is a NumPy array of python dictionaries with three attributes:
  * `corners`- a list of eight corners
  * `label`- a string representing the label to be rendered on the box (Optional)
  * `color`- rgb values representing the color of the box
* `type` is a string representing the scene type to render. Currently the only supported value is `lidar/beta`

```python
# Log points and boxes in W&B
point_scene = wandb.Object3D(
    {
        "type": "lidar/beta",
        "points": np.array(  # add points, as in a point cloud
            [[0.4, 1, 1.3], [1, 1, 1], [1.2, 1, 1.2]]
        ),
        "boxes": np.array(  # draw 3d boxes
            [
                {
                    "corners": [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ],
                    "label": "Box",
                    "color": [123, 321, 111],
                },
                {
                    "corners": [
                        [0, 0, 0],
                        [0, 2, 0],
                        [0, 0, 2],
                        [2, 0, 0],
                        [2, 2, 0],
                        [0, 2, 2],
                        [2, 0, 2],
                        [2, 2, 2],
                    ],
                    "label": "Box-2",
                    "color": [111, 321, 0],
                },
            ]
        ),
        "vectors": np.array(  # add 3d vectors
            [{"start": [0, 0, 0], "end": [0.1, 0.2, 0.5]}]
        ),
    }
)
wandb.log({"point_scene": point_scene})
```
  </TabItem>
  <TabItem value="molecules">

```python
wandb.log({"protein": wandb.Molecule("6lu7.pdb")})
```

Log molecular data in any of 10 file types:`pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, or `mmtf.`

W&B also supports logging molecular data from SMILES strings, [`rdkit`](https://www.rdkit.org/docs/index.html) `mol` files, and `rdkit.Chem.rdchem.Mol` objects.

```python
resveratrol = rdkit.Chem.MolFromSmiles("Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O")

wandb.log(
    {
        "resveratrol": wandb.Molecule.from_rdkit(resveratrol),
        "green fluorescent protein": wandb.Molecule.from_rdkit("2b3p.mol"),
        "acetaminophen": wandb.Molecule.from_smiles("CC(=O)Nc1ccc(O)cc1"),
    }
)
```

When your run finishes, you'll be able to interact with 3D visualizations of your molecules in the UI.

[See a live example using AlphaFold →](http://wandb.me/alphafold-workspace)

![](@site/static/images/track/docs-molecule.png)
  </TabItem>
</Tabs>

## Other Media

W&B also supports logging of a variety of other media types.

<Tabs
  defaultValue="audio"
  values={[
    {label: 'Audio', value: 'audio'},
    {label: 'Video', value: 'video'},
    {label: 'Text', value: 'text'},
    {label: 'HTML', value: 'html'},
  ]}>
  <TabItem value="audio">

```python
wandb.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

The maximum number of audio clips that can be logged per step is 100.

  </TabItem>
  <TabItem value="video">

```python
wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

If a numpy array is supplied we assume the dimensions are, in order: time, channels, width, height. By default we create a 4 fps gif image ([`ffmpeg`](https://www.ffmpeg.org) and the [`moviepy`](https://pypi.org/project/moviepy/) python library are required when passing numpy objects). Supported formats are `"gif"`, `"mp4"`, `"webm"`, and `"ogg"`. If you pass a string to `wandb.Video` we assert the file exists and is a supported format before uploading to wandb. Passing a `BytesIO` object will create a temporary file with the specified format as the extension.

On the W&B [Run](../../app/pages/run-page.md) and [Project](../../app/pages/project-page.md) Pages, you will see your videos in the Media section.

  </TabItem>
  <TabItem value="text">

Use `wandb.Table` to log text in tables to show up in the UI. By default, the column headers are `["Input", "Output", "Expected"]`. To ensure optimal UI performance, the default maximum number of rows is set to 10,000. However, users can explicitly override the maximum with `wandb.Table.MAX_ROWS = {DESIRED_MAX}`.

```python
columns = ["Text", "Predicted Sentiment", "True Sentiment"]
# Method 1
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
table = wandb.Table(data=data, columns=columns)
wandb.log({"examples": table})

# Method 2
table = wandb.Table(columns=columns)
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
wandb.log({"examples": table})
```

You can also pass a pandas `DataFrame` object.

```python
table = wandb.Table(dataframe=my_dataframe)
```
  </TabItem>
  <TabItem value="html">

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

Custom html can be logged at any key, and this exposes an HTML panel on the run page. By default we inject default styles, you can disable default styles by passing `inject=False`.

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

  </TabItem>
</Tabs>

## Frequently Asked Questions

### How can I compare images or media across epochs or steps?

Each time you log images from a step, we save them to show in the UI. Expand the image panel, and use the step slider to look at images from different steps. This makes it easy to compare how a model's output changes during training.

### What if I want to integrate W&B into my project, but I don't want to upload any images or media?

W&B can be used even for projects that only log scalars — you specify any files or data you'd like to upload explicitly. Here's [a quick example in PyTorch](http://wandb.me/pytorch-colab) that does not log images.

### How do I log a PNG?

[`wandb.Image`](../../../ref/python/data-types/image.md) converts `numpy` arrays or instances of `PILImage` to PNGs by default.

```python
wandb.log({"example": wandb.Image(...)})
# Or multiple images
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### How do I log a video?

Videos are logged using the [`wandb.Video`](../../../ref/python/data-types/video.md) data type:

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

Now you can view videos in the media browser. Go to your project workspace, run workspace, or report and click "Add visualization" to add a rich media panel.

### How do I navigate and zoom in point clouds?

You can hold control and use the mouse to move around inside the space.

### How do I log a 2D view of a molecule?

You can log a 2D view of a molecule using the [`wandb.Image`](../../../ref/python/data-types/image.md) data type and [`rdkit`](https://www.rdkit.org/docs/index.html):

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

wandb.log({"acetic_acid": wandb.Image(pil_image)})
```
