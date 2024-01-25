# this script will save predictions from baseline model to a wandb artifact

import wandb
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

from mini_llm.utils import parse_args

WANDB_PROJECT = "tinyllama"
WANDB_ENTITY = "reviewco"
WANDB_MODEL = "reviewco/model-registry/Small-Instruct-LLM"
TABLE_NAME = "sample_predictions"
ALIAS_BASELINE = "baseline"
ARTIFACT_NAME = "baseline_predictions"

config = SimpleNamespace(
    wandb_model = WANDB_MODEL,
    alias_baseline = ALIAS_BASELINE,
    table_name = TABLE_NAME,
    out_dir="./output",
)

if __name__ == "__main__":
    parse_args(config)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # create a run to have lineage
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="save_artifact", config=config)
    config = wandb.config
    # create a new wandb artifact if it doeasn't exist
    baseline_artifact = wandb.Artifact(ARTIFACT_NAME, type="predictions")

    # download the table from the baseline model
    table_artifact = wandb.use_artifact(f"{WANDB_MODEL}:{config.alias_baseline}", type="model")
    # Get the producer run ID from the artifact
    producer_run_id = table_artifact.logged_by().id
    # Retrieve the specific table ('sample_predictions') from the run
    table_artifact = wandb.use_artifact(f"run-{producer_run_id}-{config.table_name}:v0")
    # Download the table artifact
    table = table_artifact.get(config.table_name)

    # add the table to the artifact
    baseline_artifact.add(table, config.table_name)
    # save the artifact
    wandb.log_artifact(baseline_artifact)

