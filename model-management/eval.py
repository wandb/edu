import wandb
import json
from pathlib import Path
from types import SimpleNamespace
from tqdm.auto import tqdm
import pandas as pd
import asyncio
import openai
import instructor
from pydantic import BaseModel, Field
from typing_extensions import Literal
import numpy as np

from mini_llm.utils import parse_args

WANDB_PROJECT = "tinyllama"
WANDB_ENTITY = "reviewco"
WANDB_MODEL = "reviewco/model-registry/Small-Instruct-LLM"
TABLE_NAME = "sample_predictions"
BASELINE_ARTIFACT = "reviewco/tinyllama/baseline_predictions:latest"
ALIAS_EVAL = "candidate"

class Choice(BaseModel):
    choice: int = Field(description="-1 for first choice, 1 for second choice, 0 for equal.")
    reason: str = Field(description="Reason why the choice was made")


config = SimpleNamespace(
    openai_model = "gpt-4",
    temperature=1,
    system_prompt = ("You will be presented with a choice of two possible responses for an instruction"
                     "You have to pick the best one and give a reason why.\n"
                     "The reponse should follow the instructions and use the provided context if there is some.\n"
                     "If the first response is better, the choice is -1. If the second response is better, the choice is 1.\n"
                     "If both answers are equivalent, the choice is 0"),
    model_names=["baseline", "candidate"],
    num_samples=-1,
    out_dir="./output",
    wandb_model = WANDB_MODEL,
    table_name = TABLE_NAME,
    baseline_artifact = BASELINE_ARTIFACT,
    alias_eval = ALIAS_EVAL,
)

def download_table_from_model(alias, table_name="sample_predictions"):                  
    # Fetch the artifact (model) using the model URL and alias
    artifact = wandb.use_artifact(f"{WANDB_MODEL}:{alias}", type="model")
    # Get the producer run ID from the artifact
    producer_run_id = artifact.logged_by().id
    # Retrieve the specific table ('sample_predictions') from the run
    table_artifact = wandb.use_artifact(f"run-{producer_run_id}-{table_name}:v0")
    # Download the table artifact
    table = table_artifact.get(table_name)
    # Convert the table to a pandas dataframe
    df = pd.DataFrame(data=table.data, columns=table.columns)
    return df

def download_table_from_artifact(artifact, table_name="sample_predictions"):
    artifact = wandb.use_artifact(artifact, type='predictions')
    table = artifact.get(table_name)
    df = pd.DataFrame(data=table.data, columns=table.columns)
    return df

async def judge_row(client, row, model1_name, model2_name, system_prompt=config.system_prompt):
    "Judge two generations from the same prompt. Baseline should come first."
    instruction = row.prompt
    gen1 = row[model1_name]
    gen2 = row[model2_name]
    message = "{instruction}\n Answer 1: \n{gen1}\n Answer 2:\n{gen2}".format(instruction=instruction, gen1=gen1, gen2=gen2)

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system",
                   "content": system_prompt,
                  },
                  {"role": "user",
                   "content": message,
                  },],
        response_model=Choice,
    )
    return instruction, gen1, gen2, response.choice, response.reason


async def judge(merged_df, model1_name, model2_name):
    "Apply the judge stuff!"
    client = instructor.patch(openai.AsyncOpenAI())
    gpt4_results = await asyncio.gather(
        *[judge_row(client, row, model1_name, model2_name) for _ , row in merged_df.iterrows()]
    )
    return gpt4_results
    
if __name__ == "__main__":
    parse_args(config)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # create a run to have lineage
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="eval", tags=["gpt-4"], config=config)
    
    config = wandb.config
    # let's make sure we log exactly which model is the baseline and which is being evaluated against it
    eval_model = wandb.use_artifact(f'{config.wandb_model}:{config.alias_eval}', type="model")
    eval_model_path = f"{config.wandb_model}:{eval_model.version}"

    baseline_df = download_table_from_artifact(config.baseline_artifact)
    eval_df = download_table_from_model(config.alias_eval)

    # merge both
    merged_df = pd.merge(
        baseline_df[["prompt", "generation"]], 
        eval_df[["prompt", "generation"]], on="prompt", suffixes=config.model_names,
        how='inner'
    )
    model1_name, model2_name = (f"generation{s}" for s in config.model_names)
    loop = asyncio.get_event_loop()

    gpt_results = loop.run_until_complete(judge(merged_df.iloc[:config.num_samples], model1_name, model2_name))
    results_df = pd.DataFrame(
        gpt_results,
        columns=["prompt", config.model_names[0], config.model_names[1], "choice", "reason"])

    candidate_preference_score = np.mean(results_df["choice"].values)
    print(f"The candidate preference score on a scale from -1 to 1 is: {candidate_preference_score:.2f}")
    wandb.log({"candidate_preference_score":candidate_preference_score,
               "eval_model":eval_model_path,
    })

    results_df.to_csv(out_dir/"gpt4_eval.csv")
    gpt4_table = wandb.Table(dataframe=results_df)
    wandb.log({"gpt4_eval":gpt4_table})   