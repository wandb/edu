## This notebooks comes from 04_refactor_baseline.ipynb

import argparse, os
import wandb
from pathlib import Path
import torchvision.models as tvmodels
import pandas as pd
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

from params import BDD_CLASSES
from utils import get_predictions, create_dice_table

# defaults
default_config = SimpleNamespace(
    WANDB_PROJECT="BDD100k",
    ENTITY = None, # wandb team
    RAW_DATA_AT = 'bdd_sample_1k',
    PROCESSED_DATA_AT = 'bdd_sample_1k_split',
    framework="fastai",
    img_size=45, #(45, 80) in 16:9 proportions,
    batch_size=2, #8 keep small in Colab to be manageable
    epochs=1, # for brevity, increase for better results :)
    lr=2e-3,
    arch="resnet18",
    augment=True, # use data augmentation
    seed=42,
    log_preds=True,
    pretrained=True,  # whether to use pretrained encoder,
    mixed_precision=False, # use automatic mixed precision
)

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=default_config.img_size, help='image size')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config.arch, help='timm backbone architecture')
    argparser.add_argument('--augment', type=bool, default=default_config.augment, help='Use image augmentation techniques')
    argparser.add_argument('--seed', type=int, default=default_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=bool, default=default_config.log_preds, help='log model predictions')
    argparser.add_argument('--pretrained', type=bool, default=default_config.pretrained, help='Use pretrained image model')
    argparser.add_argument('--mixed_precision', type=bool, default=default_config.mixed_precision, help='Use Mixed Precision')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def download_dataset(at_name):
    "Grab data from artifact"
    processed_data_at = wandb.run.use_artifact(f'{at_name}:latest')
    return Path(processed_data_at.download())

def label_func(fname):
    return (fname.parent.parent/"labels")/f"{fname.stem}_mask.png"
        
def prepare_df(processed_dataset_dir, label_func, is_test=False):
    "Set absolute path image names and split"
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    if is_test: 
        # grab the test part of the split
        df = df[df.Stage == 'test'].reset_index(drop=True)
    else:
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    
    # assign paths
    df["image_fname"] = [processed_dataset_dir/f'images/{f}' for f in df.File_Name.values]
    df["label_fname"] = [label_func(f) for f in df.image_fname.values]
    return df

def get_data(df, bs=4, img_size=180, augment=True):
    block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=BDD_CLASSES)),
                  get_x=ColReader("image_fname"),
                  get_y=ColReader("label_fname"),
                  splitter=ColSplitter(),
                  item_tfms=Resize((img_size, int(img_size * 16 / 9))),
                  batch_tfms=aug_transforms() if augment else None,
                 )
    return block.dataloaders(df, bs=bs)

def log_predictions(learn):
    samples, outputs, predictions = get_predictions(learn)
    table = create_dice_table(samples, outputs, predictions, BDD_CLASSES)
    wandb.log({"pred_table":table})
    
def train(config):
    set_seed(config.seed)
    with wandb.init(project=config.WANDB_PROJECT, entity=config.ENTITY, job_type="training", config=config):
        
        # good practice to inject params using sweeps
        config = wandb.config
        
        # prepare data
        processed_dataset_dir = download_dataset(config.PROCESSED_DATA_AT)
        proc_df = prepare_df(processed_dataset_dir, label_func)
        dls = get_data(proc_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)
        
        cbs = [SaveModelCallback()] + ([MixedPrecision()] if config.mixed_precision else [])
        
        learn = unet_learner(dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, 
                             metrics=[foreground_acc, DiceMulti()], cbs=cbs)
        
        learn.fit_one_cycle(config.epochs, config.lr, cbs=[WandbCallback(log_preds=False, log_model=True)])
        if config.log_preds:
            log_predictions(learn)

if __name__ == '__main__':
    parse_args()
    train(default_config)


