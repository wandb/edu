import argparse, os
from pathlib import Path
import pandas as pd
from ml_collections import config_dict

import wandb
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

import params

def prepare_data(PROCESSED_DATA_AT):
    "Get/Download the datasets"
    processed_data_at = wandb.use_artifact(PROCESSED_DATA_AT)
    processed_dataset_dir = Path(processed_data_at.download())
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    df = df[df.stage != 'train'].reset_index(drop=True) # for eval we need test and validation datasets only
    df['test'] = df.stage == 'test'
    return df, processed_dataset_dir

def eval(cfg):
    
    set_seed(cfg.seed, reproducible=True)
    
    run = wandb.init(project=cfg.PROJECT_NAME, entity=cfg.ENTITY, job_type="evaluation", tags=['staging'])

    artifact = run.use_artifact('wandb_course/model-registry/Lemon Mold Detector:staging', type='model')
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
                                   valid_col='test', 
                                   item_tfms=Resize(cfg.img_size), 
                                   bs=cfg.bs,
                                   shuffle=False
                                  )
    learn = vision_learner(dls, 
                           cfg.arch,
                           metrics=[accuracy, Precision(), Recall(), F1Score()])

    learn.load(model_path)

    val_loss, val_accuracy, val_precision, val_recall, val_f1 = learn.validate(ds_idx=0)
    tst_loss, tst_accuracy, tst_precision, tst_recall, tst_f1 = learn.validate(ds_idx=1)

    wandb.summary["valid_loss"] = val_loss
    wandb.summary["accuracy"] = val_accuracy
    wandb.summary["precision_score"] = val_precision
    wandb.summary["recall_score"] = val_recall
    wandb.summary["f1_score"] = val_f1
    wandb.summary["tst/accuracy"] = tst_accuracy
    wandb.summary["tst/precision_score"] = tst_precision
    wandb.summary["tst/recall_score"] = tst_recall
    wandb.summary["tst/f1_score"] = tst_f1
    wandb.summary["tst/loss"] = tst_loss

    run.finish()
  
if __name__ == '__main__':

    default_cfg = config_dict.ConfigDict()
    default_cfg.PROJECT_NAME = params.PROJECT_NAME
    default_cfg.ENTITY = params.ENTITY
    default_cfg.PROCESSED_DATA_AT = f'{params.DATA_AT}:latest'
    default_cfg.target_column = 'mold'
    default_cfg.seed = 42

    eval(default_cfg)

