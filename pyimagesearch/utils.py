import re, wandb, torch, random
from pathlib import Path
import numpy as np
from fastprogress import progress_bar

VAL_DATA_AT = 'fastai/fmnist_pt/validation_data:latest'


def to_snake_case(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

def get_class_name_in_snake_case(obj):
    class_name = obj.__class__.__name__
    return to_snake_case(class_name)

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_device(t, device):
    if isinstance(t, (tuple, list)):
        return [_t.to(device) for _t in t]
    elif isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        raise("Not a Tensor or list of Tensors")

def model_size(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def log_model_preds(test_dl, preds, n=5, th=0.5):
    wandb_table = wandb.Table(columns=["images", "predictions", "groudn_truth"])

    preds = preds[:n]
    for (image, target), pred in progress_bar(zip(test_dl.dataset, preds), total=len(preds)):
        wandb_table.add_data(wandb.Image(image), pred>th, target==1)
    # Log Table
    wandb.log({"predictions": wandb_table})

def save_model(model, model_name):
    "Save the model to wandb"
    model_name = f"{wandb.run.id}_{model_name}"
    models_folder = Path("models")
    if not models_folder.exists():
        models_folder.mkdir()
    torch.save(model.state_dict(), models_folder/f"{model_name}.pth")
    at = wandb.Artifact(model_name, type="model")
    at.add_file(f"models/{model_name}.pth")
    wandb.log_artifact(at)