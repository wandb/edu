"""
Made with love by W&B
@wandbcode{emm_course}
"""

import asyncio
import random
from pathlib import Path
from types import SimpleNamespace

import instructor
import numpy as np
import openai
import pandas as pd
from mini_llm.utils import parse_args
from pydantic import BaseModel, Field
from typing_extensions import Literal

import wandb

# Configuration for the evaluation script
config = SimpleNamespace(
    openai_model = "gpt-4-0613",
    system_prompt = ("You will be presented with a choice of Model A and Model B response for an instruction.\n"
                     "The reponse should follow the instructions and use the provided context if there is some.\n"
                     "Think step by step which response is better and provide the reason why.\n"
                     "Answer with your choice: A, B, or equivalent."),
    model_names=["baseline", "candidate"],
    num_samples=-1,
    out_dir="./output",
    wandb_project='tinyllama',
    wandb_entity='reviewco',
    wandb_model='reviewco/model-registry/Small-Instruct-LLM',
    table_name='sample_predictions',
    alias_eval='candidate',
    alias_baseline='baseline',
)

# Pydantic model for structured response from GPT-4
class Choice(BaseModel):
    chain_of_thought: str = Field(description="Think step by step to decide if A or B is better response, or equivalent.")
    choice: Literal["A", "B", "equivalent"]

# Evaluator class that runs the evaluation
class Evaluator:
    def __init__(self, config):
        self.config = config
        self.client = instructor.patch(openai.AsyncOpenAI())
        self.model1_name, self.model2_name = (f"generation{s}" for s in config.model_names)

    async def judge_row(self, row):
        instruction = row.prompt
        gen1 = row[self.model1_name]
        gen2 = row[self.model2_name]

        # Randomly shuffle the answers
        answers = [(gen1, -1), (gen2, 1)]
        random.shuffle(answers)
        shuffled_gen1, choice1 = answers[0]
        shuffled_gen2, choice2 = answers[1]

        message = "{instruction}\nModel A:\n{gen1}\nModel B:\n{gen2}".format(instruction=instruction, gen1=shuffled_gen1, gen2=shuffled_gen2)

        response = await self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "system", "content": self.config.system_prompt},
                      {"role": "user", "content": message}],
            response_model=Choice,
            temperature=0.,
        )

        # Adjust the choice based on the shuffle
        if response.choice == "A":
            response.choice = choice1
        elif response.choice == "B":
            response.choice = choice2
        elif response.choice == "equivalent":
            response.choice = 0
        else:
            raise ValueError(f"Unexpected choice: {response.choice}")

        return instruction, gen1, gen2, response.choice, response.chain_of_thought

    async def judge(self, merged_df):
        gpt4_results = await asyncio.gather(
            *[self.judge_row(row) for _ , row in merged_df.iterrows()]
        )
        return gpt4_results

def download_table_from_model(wandb_model, alias, table_name="sample_predictions"):                  
    artifact = wandb.use_artifact(f"{wandb_model}:{alias}", type="model")
    producer_run_id = artifact.logged_by().id
    table_artifact = wandb.use_artifact(f"run-{producer_run_id}-{table_name}:v0")
    table = table_artifact.get(table_name)
    df = pd.DataFrame(data=table.data, columns=table.columns)
    return df

if __name__ == "__main__":
    parse_args(config)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=config.wandb_project, entity=config.wandb_entity, job_type="eval", config=config)
    config = wandb.config

    # This code will add lineage to the evaluation results, let's store the model paths in the config
    eval_model = wandb.use_artifact(f'{config.wandb_model}:{config.alias_eval}', type="model")
    eval_model_path = f"{config.wandb_model}:{eval_model.version}"
    baseline_model = wandb.use_artifact(f'{config.wandb_model}:{config.alias_baseline}', type="model")
    baseline_model_path = f"{config.wandb_model}:{baseline_model.version}"

    wandb.config.update({
        "eval_model_path": eval_model_path,
        "baseline_model_path": baseline_model_path,
    })

    # Download and merge the baseline and evaluation dataframes
    eval_df = download_table_from_model(config.wandb_model, config.alias_eval)
    baseline_df = download_table_from_model(config.wandb_model, config.alias_baseline)
    merged_df = pd.merge(
        baseline_df[["prompt", "generation"]], 
        eval_df[["prompt", "generation"]], on="prompt", suffixes=config.model_names,
        how='inner'
    )

    # Run the evaluation and calculate the preference score
    evaluator = Evaluator(config)
    gpt_results = asyncio.run(evaluator.judge(merged_df.iloc[:evaluator.config.num_samples]))
    results_df = pd.DataFrame(
        gpt_results,
        columns=["prompt", config.model_names[0], config.model_names[1], "choice", "reason"])

    # Calculate the preference score
    candidate_preference_score = np.mean(results_df["choice"].values)
    print(f"The candidate preference score on a scale from -1 to 1 is: {candidate_preference_score:.2f}")
    wandb.log({"candidate_preference_score":candidate_preference_score})

    # Save the results and log them to Weights & Biases
    results_df.to_csv(out_dir/"gpt4_eval.csv")
    gpt4_table = wandb.Table(dataframe=results_df)
    wandb.log({"gpt4_eval":gpt4_table})