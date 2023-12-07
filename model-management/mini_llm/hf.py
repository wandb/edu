import os, glob, json, argparse
from ast import literal_eval
from functools import partial
from tqdm.auto import tqdm
from pathlib import Path

import wandb
import pandas as pd

from datasets import load_from_disk

import evaluate
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import Trainer, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import WandbCallback

def debug_trainer_data(trainer: Trainer):
    """Print a bunch of debug info about how the packed dataset is being constructed.
    We set everythin to finite to avoid iterating forever"""
    print("Computing Dataset Stats...")
    train_ds = trainer.train_dataset
    train_ds.infinite = False
    eval_ds = trainer.eval_dataset
    eval_ds.infinite = False
    len_train_ds = sum(1 for _ in train_ds)
    len_eval_ds = sum(1 for _ in eval_ds)
    print(
        f"  len(train_ds): {len_train_ds}\n"
        f"  len(eval_ds) : {len_eval_ds}\n"
    )
    train_dl = trainer.get_train_dataloader()
    train_dl.dataset.infinite = False
    eval_dl = trainer.get_eval_dataloader()
    eval_dl.dataset.infinite = False
    len_train_dl = sum(1 for _ in train_dl)
    len_eval_dl = sum(1 for _ in eval_dl)
    b = next(iter(train_dl))
    input_ids, labels = b["input_ids"], b["labels"]
    
    print(
        f"  len(train_dl): {len_train_dl}\n"
        f"  len(eval_dl) : {len_eval_dl}\n"
        f"  batch_shape  : {input_ids.shape}\n"
    )
    tokenizer = trainer.tokenizer
    decoded_ids = tokenizer.decode(input_ids[0])[0:80]
    decoded_labels = tokenizer.decode(labels[0])[0:80]
    print("First batch:\n"
          f"input_ids:\n{decoded_ids}\n"
          f"labels:\n{decoded_labels}\n")

DEFAULT_LORA_CONFIG = LoraConfig(
    r=64,  # the rank of the LoRA matrices
    lora_alpha=16, # the weight
    lora_dropout=0.1, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
)

def create_peft_model(
        model, 
        gradient_checkpointing=False, 
        peft_config=DEFAULT_LORA_CONFIG):
    # create the LoRA config

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) 

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

def load_ds_from_artifact(at_address, at_type="dataset"):
    "Load a HF dataset from a W&B artifact"
    artifact = wandb.use_artifact(at_address, type=at_type)
    artifact_dir = artifact.download()
    return load_from_disk(artifact_dir)


def model_class(model_path):
    if list(model_path.glob("*adapter*")) and not list(model_path.glob("model*.safetensors")):
        return AutoPeftModelForCausalLM
    return AutoModelForCausalLM

def load_model_from_artifact(model_at, **model_kwargs):
    """Load model and tokenizer from W&B
    If the tokenizer is not present, we load a pretrained from the hub"""
    if not wandb.run:
        from wandb import Api
        api = Api()
        artifact = api.artifact(model_at, type="model")
    else:
        artifact = wandb.use_artifact(model_at, type="model")
    artifact_dir = Path(artifact.download())
    
    model = model_class(artifact_dir).from_pretrained(
        artifact_dir,
        **model_kwargs)
    try:
        tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
        tokenizer.pad_token = tokenizer.eos_token
    except:
        model_id = artifact.metadata.get("model_id")
        if model_id is None:
            producer_run = artifact.logged_by()
            config = producer_run.config
            model_id = config.get("model_id")
        print(f"Tokenizer not found.\nLoading a pretrained tokenizer from the HF-hub: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, artifact_dir
    
def _generate(prompt, model, tokenizer, gen_config):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(tokenized_prompt, 
                                generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(inputs=tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    
    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        return records_table
        
    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})

def param_count(model):
    params = sum([p.numel() for p in model.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
    return params, trainable_params
    
def freeze(model, n_freeze, freeze_embed, module_name="layers"):
    if n_freeze > 0:
        def _find_mod(model, module_name):
            for name, mod in model.named_modules():
                if name.endswith(module_name):
                    return mod
        # freeze layers (disable gradients)
        for param in model.parameters(): param.requires_grad = False

        # never freeze the head
        for param in model.lm_head.parameters(): param.requires_grad = True
    
        layers = _find_mod(model, module_name)
        for param in layers[n_freeze:].parameters(): param.requires_grad = True
    
    # Freeze embeddings for small memory decrease
    if freeze_embed:
        embed_tokens = _find_mod(model, "embed_tokens")
        embed_tokens.weight.requires_grad_(False);