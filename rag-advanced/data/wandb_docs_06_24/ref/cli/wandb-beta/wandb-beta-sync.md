# wandb beta sync

**Usage**

`wandb beta sync [OPTIONS] WANDB_DIR`

**Summary**

Upload a training run to W&B

**Options**

| **Option** | **Description** |
| :--- | :--- |
| --id | The run you want to upload to. |
| -p, --project | The project you want to upload to. |
| -e, --entity | The entity to scope to. |
| --skip-console | Skip console logs |
| --append | Append run |
| -i, --include | Glob to include. Can be used multiple times. |
| -e, --exclude | Glob to exclude. Can be used multiple times. |
| --mark-synced / --no-mark-synced | Mark runs as synced |
| --skip-synced / --no-skip-synced | Skip synced runs |
| --dry-run | Perform a dry run without uploading   anything. |

