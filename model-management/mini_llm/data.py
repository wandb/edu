import wandb
from .utils import load_jsonl

DEFAULT_ALPACA_SPLIT = 'capecape/alpaca_ft/alpaca_gpt4_splitted:v4'

def _prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)

def _prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)

def create_alpaca_prompt(row):
    return _prompt_no_input(row) if row["input"] == "" else _prompt_input(row)

def create_alpaca_prompt_with_response(row):
    instruct = _prompt_no_input(row) if row["input"] == "" else _prompt_input(row)
    return instruct + row["output"]

def get_alpaca_split(dataset_at = DEFAULT_ALPACA_SPLIT):
    artifact = wandb.use_artifact(dataset_at, type='dataset')
    dataset_dir = artifact.download()
    
    train_dataset = load_jsonl(f"{dataset_dir}/alpaca_gpt4_train.jsonl")
    eval_dataset = load_jsonl(f"{dataset_dir}/alpaca_gpt4_eval.jsonl")

    def _format_dataset(dataset):
        "No EOS token yet"
        return [{"prompt":create_alpaca_prompt(row), 
                 "output": row["output"]} for row in dataset]
        
    train_dataset = _format_dataset(train_dataset)
    eval_dataset = _format_dataset(eval_dataset)
    print(train_dataset[0])
    return train_dataset, eval_dataset