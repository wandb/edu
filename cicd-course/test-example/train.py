## This script comes from 04_refactor_baseline.ipynb

import argparse, os
import wandb
from pathlib import Path
import torchvision.models as tvmodels
import pandas as pd
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

import params
from utils import get_predictions, create_iou_table, MIOU, BackgroundIOU, \
                  RoadIOU, TrafficLightIOU, TrafficSignIOU, PersonIOU, VehicleIOU, BicycleIOU, t_or_f
# defaults
default_config = SimpleNamespace(
    framework="fastai",
    img_size=180, #(180, 320) in 16:9 proportions,
    batch_size=8, #8 keep small in Colab to be manageable
    augment=True, # use data augmentation
    epochs=10, # for brevity, increase for better results :)
    lr=2e-3,
    pretrained=True,  # whether to use pretrained encoder,
    mixed_precision=True, # use automatic mixed precision
    arch="resnet18",
    seed=42,
    log_preds=False,
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=default_config.img_size, help='image size')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config.arch, help='timm backbone architecture')
    argparser.add_argument('--augment', type=t_or_f, default=default_config.augment, help='Use image augmentation')
    argparser.add_argument('--seed', type=int, default=default_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=t_or_f, default=default_config.log_preds, help='log model predictions')
    argparser.add_argument('--pretrained', type=t_or_f, default=default_config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=default_config.mixed_precision, help='use fp16')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def download_data():
    "Grab dataset from artifact"
    processed_data_at = wandb.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def label_func(fname):
    return (fname.parent.parent/"labels")/f"{fname.stem}_mask.png"
        
def get_df(processed_dataset_dir, is_test=False):
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    
    if not is_test:
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    else:
        df = df[df.Stage == 'test'].reset_index(drop=True)
        
    
    # assign paths
    df["image_fname"] = [processed_dataset_dir/f'images/{f}' for f in df.File_Name.values]
    df["label_fname"] = [label_func(f) for f in df.image_fname.values]
    return df

def get_data(df, bs=4, img_size=180, augment=True):
    block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=params.BDD_CLASSES)),
                  get_x=ColReader("image_fname"),
                  get_y=ColReader("label_fname"),
                  splitter=ColSplitter(),
                  item_tfms=Resize((img_size, int(img_size * 16 / 9))),
                  batch_tfms=aug_transforms() if augment else None,
                 )
    return block.dataloaders(df, bs=bs)


def log_predictions(learn):
    "Log a Table with model predictions and metrics"
    samples, outputs, predictions = get_predictions(learn)
    table = create_iou_table(samples, outputs, predictions, params.BDD_CLASSES)
    wandb.log({"pred_table":table})
    
def final_metrics(learn):
    "Log latest metrics values"
    scores = learn.validate()
    metric_names = ['final_loss'] + [f'final_{x.name}' for x in learn.metrics]
    final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
    for k,v in final_results.items(): 
        wandb.summary[k] = v

def train(config, nrows=None):
    set_seed(config.seed)
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training", config=config)
        
    # good practice to inject params using sweeps
    config = wandb.config

    # prepare data
    processed_dataset_dir = download_data()
    proc_df = get_df(processed_dataset_dir).head(nrows)
    dls = get_data(proc_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

    metrics = [MIOU(), BackgroundIOU(), RoadIOU(), TrafficLightIOU(),
               TrafficSignIOU(), PersonIOU(), VehicleIOU(), BicycleIOU()]

    cbs = [WandbCallback(log_preds=False, log_model=True), 
           SaveModelCallback(fname=f'run-{wandb.run.id}-model', monitor='miou')]
    cbs += ([MixedPrecision()] if config.mixed_precision else [])

    learn = unet_learner(dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, 
                         metrics=metrics)

    learn.fit_one_cycle(config.epochs, config.lr, cbs=cbs)
    if config.log_preds:
        log_predictions(learn)
    final_metrics(learn)
    
    wandb.finish()

if __name__ == '__main__':
    parse_args()
    train(default_config)


