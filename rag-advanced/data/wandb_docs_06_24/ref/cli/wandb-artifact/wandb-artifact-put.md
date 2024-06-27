# wandb artifact put

**Usage**

`wandb artifact put [OPTIONS] PATH`

**Summary**

Upload an artifact to wandb

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -n, --name | The name of the artifact to push:   project/artifact_name |
| -d, --description | A description of this artifact |
| -t, --type | The type of the artifact |
| -a, --alias | An alias to apply to this artifact |
| --id | The run you want to upload to. |
| --resume | Resume the last run from your current   directory. |
| --skip_cache | Skip caching while uploading artifact files. |
| --policy [mutable|immutable] | Set the storage policy while uploading   artifact files. |

