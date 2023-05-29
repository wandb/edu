from langchain.evaluation.qa import QAEvalChain
from langchain.chat_models import ChatOpenAI

import pandas as pd
from pathlib import Path
from tqdm import tqdm

import wandb
from wandb.integration.langchain import WandbTracer

from config import default_config, wandb_config
from chain import load_chain
from prompts import load_eval_prompt


def load_eval_dataset(config):
    # we will load data from a wandb Table  artifact
    artifact = wandb.use_artifact(config.eval_artifact)
    # download artifact
    artifact_dir = Path(artifact.download())
    # load data
    eval_dataset = pd.read_csv(artifact_dir / "generated_examples.csv")
    return eval_dataset

def generate_answers(eval_dataset, qa_chain):
    answers = []
    for query in tqdm(eval_dataset["question"], total=len(eval_dataset)):
        answer = qa_chain.run(query=query, callbacks=[WandbTracer(wandb_config)])
        answers.append(answer)
    eval_dataset["model_answer"] = answers
    eval_dataset.to_csv("eval_with_answers.csv", index=False)
    return eval_dataset

def evaluate_answers(eval_dataset, config):
        chat_prompt = load_eval_prompt()
        llm = ChatOpenAI(
            model_name=config.eval_model,
            temperature=0,
        )
        eval_chain = QAEvalChain.from_llm(llm, prompt=chat_prompt)

        examples = []
        predictions = []
        for i in range(len(eval_dataset)):
            examples.append(
                {
                    "query": eval_dataset["question"].iloc[i],
                    "answer": eval_dataset["answer"].iloc[i],
                }
            )
            predictions.append(
                {
                    "query": eval_dataset["question"].iloc[i],
                    "answer": eval_dataset["answer"].iloc[i],
                    "result": eval_dataset["model_answer"].iloc[i],
                }
            )
        graded_outputs = eval_chain.evaluate(examples, predictions)
        eval_dataset["model_score"] = [x.get("text", "None") for x in graded_outputs]
        return eval_dataset

def log_results(eval_dataset):
        model_accuracy = len(eval_dataset[eval_dataset["model_score"] == "CORRECT"]) / len(eval_dataset)
        wandb.log({"model_accuracy": model_accuracy})
        eval_dataset.to_csv("eval_results.csv", index=False)
        artifact = wandb.Artifact("eval_results", type="eval_results")
        artifact.add_file("eval_results.csv")
        wandb.log_artifact(artifact)
        wandb.log({"eval_results": wandb.Table(dataframe=eval_dataset)})


if __name__ == "__main__":
    with wandb.init(project=default_config.project, job_type="eval"):
        eval_dataset = load_eval_dataset(default_config)
        qa_chain = load_chain(default_config)
        eval_dataset = generate_answers(eval_dataset, qa_chain)
        eval_dataset = evaluate_answers(eval_dataset, default_config)
        log_results(eval_dataset)
