from transformers import Trainer
from peft import LoraConfig, get_peft_model

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