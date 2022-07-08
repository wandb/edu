import argparse, os
from pathlib import Path
import pandas as pd
from ml_collections import config_dict

import wandb
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

def prepare_data(PROCESSED_DATA_AT):
    "Get/Download the datasets"
    processed_data_at = wandb.use_artifact(PROCESSED_DATA_AT)
    processed_dataset_dir = Path(processed_data_at.download())
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    df = df[df.stage != 'train'].reset_index(drop=True) # for eval we need test and validation datasets only
    df['valid'] = df.stage == 'valid'
    return df, processed_dataset_dir

def get_dataloaders(df, path, seed, target_column, img_size, bs):
    "Get test/valid PyTorch Dataloaders"
    set_seed(seed, reproducible=True)
    dls = ImageDataLoaders.from_df(df, path=path, seed=seed, fn_col='file_name', label_col=target_column, 
                                   valid_col='valid', item_tfms=Resize(img_size), bs=bs)
    return dls

def eval(cfg):
    
    run = wandb.init(project=cfg.PROJECT_NAME, entity=cfg.ENTITY, job_type="evaluation", tags=['staging'])

    artifact = run.use_artifact('wandb_course/model-registry/Lemon Mold Detector:v0', type='model')
    artifact_dir = artifact.download()
    model_path = Path(artifact_dir).absolute()/'model'
    
    producer_run = artifact.logged_by()
    cfg.img_size = producer_run.config['img_size']
    cfg.bs = producer_run.config['bs']
    cfg.arch = producer_run.config['arch']
        
    wandb.config.update(cfg)
    
    df, path = prepare_data(cfg.PROCESSED_DATA_AT)
    
    dls = ImageDataLoaders.from_df(df, path=path,
                                   fn_col='file_name', 
                                   label_col=cfg.target_column, 
                                   valid_col='valid', 
                                   item_tfms=Resize(cfg.img_size), 
                                   bs=cfg.bs
                                  )
    learn = vision_learner(dls, 
                           cfg.arch,
                           metrics=[accuracy, Precision(), Recall(), F1Score()])

    learn.load(model_path)

    _, val_accuracy, val_precision, val_recall, val_f1 = learn.validate(ds_idx=1)
    _, tst_accuracy, tst_precision, tst_recall, tst_f1 = learn.validate(ds_idx=0)

    wandb.log({
        "val/accuracy": val_accuracy,
        "val/precision": val_precision,
        "val/recall": val_recall,
        "val/f1": val_f1,
        "tst/accuracy": tst_accuracy,
        "tst/precision": tst_precision,
        "tst/recall": tst_recall,
        "tst/f1": tst_f1,        
    })

    run.finish()
  
if __name__ == '__main__':

    default_cfg = config_dict.ConfigDict()
    default_cfg.PROJECT_NAME = 'lemon-project'
    default_cfg.ENTITY = 'wandb_course'
    default_cfg.PROCESSED_DATA_AT = 'lemon_dataset_split_data:latest'
    default_cfg.target_column = 'mold'

    eval(default_cfg)


