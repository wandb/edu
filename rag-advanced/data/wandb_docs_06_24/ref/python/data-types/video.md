# Video

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/video.py#L48-L239' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Format a video for logging to W&B.

```python
Video(
    data_or_path: Union['np.ndarray', str, 'TextIO', 'BytesIO'],
    caption: Optional[str] = None,
    fps: int = 4,
    format: Optional[str] = None
)
```

| Arguments |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) Video can be initialized with a path to a file or an io object. The format must be "gif", "mp4", "webm" or "ogg". The format must be specified with the format argument. Video can be initialized with a numpy tensor. The numpy tensor must be either 4 dimensional or 5 dimensional. Channels should be (time, channel, height, width) or (batch, time, channel, height width) |
|  `caption` |  (string) caption associated with the video for display |
|  `fps` |  (int) frames per second for video. Default is 4. |
|  `format` |  (string) format of video, necessary if initializing with path or io object. |

#### Examples:

### Log a numpy array as a video

<!--yeadoc-test:log-video-numpy-->


```python
import numpy as np
import wandb

wandb.init()
# axes are (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=4)})
```

## Methods

### `encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/video.py#L130-L167)

```python
encode() -> None
```

| Class Variables |  |
| :--- | :--- |
|  `EXTS`<a id="EXTS"></a> |   |
