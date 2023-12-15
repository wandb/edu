import wandb
import json
from pathlib import Path
from types import SimpleNamespace
from tqdm.auto import tqdm
import pandas as pd

from mini_llm.utils import parse_args
from mini_llm.openai import completion_with_backoff

WANDB_PROJECT = "tinyllama"
WANDB_ENTITY = "reviewco"
WANDB_MODEL = "reviewco/model-registry/Small-Instruct-LLM"
TABLE_NAME = "sample_predictions"
BASELINE_ARTIFACT = "reviewco/tinyllama/baseline_predictions:latest"
ALIAS_EVAL = "candidate"


config = SimpleNamespace(
    openai_model = "gpt-4",
    temperature=1,
    system_prompt = ("You will be presented with a choice of two possible responses for an instruction"
                     "You have to pick the best one and give a reason why.\n"
                     "The reponse should follow the instructions and use the provided context if there is some\n"
                     "If both answers are equivalent, pick the value 0"),
    model_names=["ft_model", "gpt35"],
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

def gpt4_judge(instruction, gen1, gen2, model=config.openai_model, system_prompt=config.system_prompt):
    message = "{instruction}\n Answer 1: \n{gen1}\n Answer 2:\n{gen2}".format(instruction=instruction, gen1=gen1, gen2=gen2)
    completion = completion_with_backoff(
        model="gpt-4",
        messages=[{"role": "system",
                   "content": system_prompt,
                  },
                  {"role": "user",
                   "content": message,
                  },],
        function_call = {"name": "make_choice"},
        functions = [
            {
                "name": "make_choice",
                "description": "Select the best generation and argument why",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "choice": {
                            "type": "integer",
                            "description": "the choosen alternative, zero if equivalent",
                        },
                        "reason":{
                            "type": "string",
                            "description": "Reason why the choice was made",
                        },
                    }
                },
                    "required": ["choice", "reason"],
            },
        ],
    )
    return completion

def judge_row(row, model1_name, model2_name):
    "Judge with inversion of prompt order (2x more expensive)"
    prompt = row.prompt
    gen1 = row[model1_name]
    gen2 = row[model2_name]
    res = gpt4_judge(prompt, gen1, gen2)
    res_inverted = gpt4_judge(prompt, gen2, gen1)
    return res, res_inverted

def extract_function_call(res):
    "Extract the function call arguments"
    try:
        response = json.loads(res.choices[0].message.function_call.arguments)
        choice = response["choice"]
        reason = response["reason"]
    except:
        choice = -1
        reason = "gpt4 fail"
    return choice, reason

def judge(merged_df, model1_name, model2_name):
    "Apply the judge stuff!"
    gpt4_results = []
    for i in tqdm(range(len(merged_df))):
        row = merged_df.iloc[i]
        res, res_inverted = judge_row(row, model1_name, model2_name)
        choice, reason = extract_function_call(res)
        print(choice, reason)
        choice_inverted, reason_inverted = extract_function_call(res_inverted)
        print(choice_inverted, reason_inverted)
        agree = (choice - choice_inverted) % 2 == 1
        # print(f"GPT4 prefers: {choice} and {choice_inverted}: Agree={agree}")
        gpt4_results.append((i, row.prompt, row[model1_name], row[model2_name], choice, choice_inverted, reason, reason_inverted))
    return gpt4_results

def agree_check(row):
    "Check where the GPT4 agrees with himself"
    if row.choice == 0 and row.choice_inverted==0:
        return True
    elif ((row.choice - row.choice_inverted) % 2 == 1):
        return True
    else:
        return False
    
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
        eval_df[["prompt", "generation"]], 
        baseline_df[["prompt", "generation"]], on="prompt", suffixes=config.model_names,
        how='inner'
    )
    model1_name, model2_name = (f"generation{s}" for s in config.model_names)
    gpt_results = judge(merged_df.iloc[:config.num_samples], model1_name, model2_name)
    results_df = pd.DataFrame(
        gpt_results,
        columns=["index", "prompt", config.model_names[0], config.model_names[1], 
                 "choice", "choice_inverted", "reason", "reason_inverted"]).set_index("index")
    results_df["agree"] = results_df.apply(agree_check, axis=1)

    final_results = results_df[results_df.agree]
    print(f"The judge agrees on: {len(final_results)} / {len(eval_df)}")
    
    # Let's log those also:
    disagree_df = results_df[~results_df.agree]

    # Lets use a better naming than 0,1,2
    choices = ["both",] + config.model_names
    final_results["choice_name"] = [choices[c] for c in final_results["choice"]]
    print("\n### GPT JUDGE RESULTS ###")
    print("###########################")
    print(final_results["choice_name"].value_counts())
    print("###########################")

    # calculate a single metric as percentage of eval model preference over baseline
    final_results["ft_pref"] = final_results["choice"] == 1
    ft_pref = final_results["ft_pref"].mean()
    print(f"Candidate model preference: {ft_pref}")
    wandb.log({"eval_pref":ft_pref,
               "eval_model":eval_model_path,
    })

    final_results.to_csv(out_dir/"gpt4_eval.csv")
    gpt4_table = wandb.Table(dataframe=final_results)
    wandb.log({"gpt4_eval":gpt4_table})   

    disagree_df["choice"] = [choices[c] for c in disagree_df["choice"]]
    choices_inverted = ["both",] + config.model_names[::-1]
    disagree_df["choice_inverted"] = [choices_inverted[c] for c in disagree_df["choice_inverted"]]

    disagree_df.to_csv(out_dir/"gpt4_eval_disagree.csv")
    gpt4_eval_disagree_table  = wandb.Table(dataframe=disagree_df)
    wandb.log({"gpt4_eval_disagree":gpt4_eval_disagree_table})