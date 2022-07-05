import argparse, os
from pathlib import Path
import pandas as pd
from ml_collections import config_dict

import wandb
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

# defaults
default_cfg = config_dict.ConfigDict()
default_cfg.img_size = 256
default_cfg.target_column = 'mold'
default_cfg.bs = 16
default_cfg.seed = 42
default_cfg.epochs = 2
default_cfg.lr = 2e-3
default_cfg.arch = 'resnet18'
default_cfg.log_model = False
default_cfg.PROJECT_NAME = 'lemon-project'
default_cfg.ENTITY = 'wandb_course'
default_cfg.PROCESSED_DATA_AT = 'lemon_dataset_split_data:latest'

# optional
def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=default_cfg.img_size, help='image size')
    argparser.add_argument('--bs', type=int, default=default_cfg.bs, help='batch size')
    argparser.add_argument('--seed', type=int, default=default_cfg.seed, help='random seed')
    argparser.add_argument('--epochs', type=int, default=default_cfg.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_cfg.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_cfg.arch, help='timm backbone architecture')
    argparser.add_argument('--log_model', action="store_true", help='log best model W&B')
    return argparser.parse_args()


def prepare_data(PROCESSED_DATA_AT):
    "Get/Download the datasets"
    processed_data_at = wandb.use_artifact(PROCESSED_DATA_AT)
    processed_dataset_dir = Path(processed_data_at.download())
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    df = df[df.stage != 'test'].reset_index(drop=True)
    df['valid'] = df.stage == 'valid'
    return df, processed_dataset_dir

def get_dataloaders(df, path, seed, target_column, img_size, bs):
    "Get train/valid PyTorch Dataloaders"
    set_seed(seed, reproducible=True)
    dls = ImageDataLoaders.from_df(df, path=path, seed=seed, fn_col='file_name', label_col=target_column, 
                                   valid_col='valid', item_tfms=Resize(img_size), bs=bs)
    return dls

def log_predictions(learn):
    "Log a wandb.Table with model predictions on the validation dataset"
    inp,preds,targs,out = learn.get_preds(with_input=True, with_decoded=True)
    imgs = [wandb.Image(t.permute(1,2,0)) for t in inp]
    pred_proba = preds[:,1].numpy().tolist()
    targets = targs.numpy().tolist()
    predictions = out.numpy().tolist()
    df = pd.DataFrame(list(zip(imgs, pred_proba, predictions, targets)),
               columns =['image', 'probability', 'prediction', 'target'])
    wandb.log({'predictions_table': wandb.Table(dataframe=df)})

def train(cfg):
    with wandb.init(project=cfg.PROJECT_NAME, entity=cfg.ENTITY, job_type="training", config=dict(cfg)):
        cfg = wandb.config
        df, path = prepare_data(cfg.PROCESSED_DATA_AT)
        dls = get_dataloaders(df, path, cfg.seed, cfg.target_column, cfg.img_size, cfg.bs)
        learn = vision_learner(dls, 
                               cfg.arch,
                               metrics=[accuracy, Precision(), Recall(), F1Score()],
                               cbs=[WandbCallback(log_preds=False, log_model=cfg.log_model), 
                                    SaveModelCallback(monitor='f1_score')]).to_fp16()
        learn.fine_tune(cfg.epochs, cfg.lr)   
        if cfg.log_model:
            log_predictions(learn)
        

if __name__ == '__main__':
    default_cfg.update(vars(parse_args()))
    train(default_cfg)


