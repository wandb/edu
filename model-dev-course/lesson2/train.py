import os
import pandas as pd
from ml_collections import config_dict
from fastai.vision.all import Path, ImageDataLoaders, vision_learner, \
    accuracy, Precision, Recall, F1Score, SaveModelCallback, Resize
from fastai.callback.wandb import WandbCallback
import timm
import argparse

import wandb

def prepare_data(cfg, run):
    raw_data_at = run.use_artifact(f'{cfg.RAW_DATA_AT}:latest')
    raw_dataset_dir = raw_data_at.download()
    processed_data_at = run.use_artifact(f'{cfg.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = processed_data_at.download()
    df = pd.read_csv(os.path.join(processed_dataset_dir, 'data_split.csv'))
    df = df[df.stage != 'test'].reset_index(drop=True)
    df['valid'] = df.stage == 'valid'
    path = Path(raw_dataset_dir)
    return df, path

def run(cfg):
    run = wandb.init(project=cfg.PROJECT_NAME, entity=cfg.ENTITY, job_type="training", tags=['fastai'])
    wandb.config.update(cfg)
    df, path = prepare_data(cfg, run)
    dls = ImageDataLoaders.from_df(df, path=path,
                                   seed=cfg.seed, 
                                   fn_col='file_name', 
                                   label_col=cfg.target_column, 
                                   valid_col='valid', 
                                   item_tfms=Resize(cfg.img_size), 
                                   bs=cfg.bs
                                  )
    learn = vision_learner(dls, 
                           cfg.arch,
                           metrics=[accuracy, Precision(), Recall(), F1Score()],
                           cbs=[WandbCallback(log_preds=False), SaveModelCallback(monitor='f1_score')])
    learn.fine_tune(cfg.epochs)   
    inp,preds,targs,out = learn.get_preds(with_input=True, with_decoded=True)
    imgs = [wandb.Image(t.permute(1,2,0)) for t in inp]
    pred_proba = preds[:,1].numpy().tolist()
    targets = targs.numpy().tolist()
    predictions = out.numpy().tolist()
    ids = list(range(len(predictions)))
    df = pd.DataFrame(list(zip(ids, imgs, pred_proba, predictions, targets)),
               columns =['id', 'image', 'probability', 'prediction', 'target'])
    run.log({'predictions_table': wandb.Table(dataframe=df)})
    run.finish()
  
if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Process hyper-parameters')

    argparser.add_argument('--img_size', type=int, default=256, help='image size')
    argparser.add_argument('--bs', type=int,   default=32, help='batch size')
    argparser.add_argument('--seed', type=int,   default=42, help='random seed')
    argparser.add_argument('--epochs', type=int,   default=2, help='number of training epochs')
    argparser.add_argument('--arch', type=str,   default="resnet18", help='timm backbone architecture')

    args = argparser.parse_args()

    cfg = config_dict.ConfigDict(vars(args))
    cfg.target_column = 'mold'
    cfg.PROJECT_NAME = 'lemon-test1'
    cfg.ENTITY = 'wandb_course'
    cfg.RAW_DATA_AT = 'lemon_dataset_raw_data_2690'
    cfg.PROCESSED_DATA_AT = 'lemon_dataset_processed_data'

    run(cfg)


