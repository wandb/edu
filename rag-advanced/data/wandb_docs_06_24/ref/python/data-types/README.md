# Data Types

<!-- Insert buttons and diff -->


<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


This module defines data types for logging rich, interactive visualizations to W&B.

Data types include common media types, like images, audio, and videos,
flexible containers for information, like tables and HTML, and more.

For more on logging media, see [our guide](https://docs.wandb.com/guides/track/log/media)

For more on logging structured data for interactive dataset and model analysis,
see [our guide to W&B Tables](https://docs.wandb.com/guides/data-vis).

All of these special data types are subclasses of WBValue. All the data types
serialize to JSON, since that is what wandb uses to save the objects locally
and upload them to the W&B server.

## Classes

[`class Audio`](./audio.md): Wandb class for audio clips.

[`class BoundingBoxes2D`](./boundingboxes2d.md): Format images with 2D bounding box overlays for logging to W&B.

[`class Graph`](./graph.md): Wandb class for graphs.

[`class Histogram`](./histogram.md): wandb class for histograms.

[`class Html`](./html.md): Wandb class for arbitrary html.

[`class Image`](./image.md): Format images for logging to W&B.

[`class ImageMask`](./imagemask.md): Format image masks or overlays for logging to W&B.

[`class Molecule`](./molecule.md): Wandb class for 3D Molecular data.

[`class Object3D`](./object3d.md): Wandb class for 3D point clouds.

[`class Plotly`](./plotly.md): Wandb class for plotly plots.

[`class Table`](./table.md): The Table class used to display and analyze tabular data.

[`class Video`](./video.md): Format a video for logging to W&B.

[`class WBTraceTree`](./wbtracetree.md): Media object for trace tree data.
