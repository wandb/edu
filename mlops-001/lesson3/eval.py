import wandb
import torchvision.models as tvmodels
import pandas as pd
from fastai.vision.all import *

import params
from utils import get_predictions, create_iou_table, MIOU, BackgroundIOU, \
                  RoadIOU, TrafficLightIOU, TrafficSignIOU, PersonIOU, VehicleIOU, BicycleIOU, \
                  t_or_f, display_diagnostics

def download_data():
    """Grab dataset from artifact
    @wandbcode{course-lesson3}
    """
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
        df = df[df.Stage != 'train'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
        # when passed to datablock, this will return test at index 0 and valid at index 1
    
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
    wandb.log({"val_pred_table":table})
    
def count_by_class(arr, cidxs): 
    return [(arr == n).sum(axis=(1,2)).numpy() for n in cidxs]

def log_hist(c):
    _, bins, _ = plt.hist(target_counts[c],  bins=10, alpha=0.5, density=True, label='target')
    _ = plt.hist(pred_counts[c], bins=bins, alpha=0.5, density=True, label='pred')
    plt.legend(loc='upper right')
    plt.title(params.BDD_CLASSES[c])
    img_path = f'hist_val_{params.BDD_CLASSES[c]}'
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_path: wandb.Image(f'{img_path}.png', caption=img_path)})

run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="evaluation", tags=['staging'])

artifact = run.use_artifact('av-team/model-registry/BDD Semantic Segmentation:latest', type='model')

artifact_dir = Path(artifact.download())

_model_pth = artifact_dir.ls()[0]
model_path = _model_pth.parent.absolute()/_model_pth.stem

producer_run = artifact.logged_by()
wandb.config.update(producer_run.config)
config = wandb.config

processed_dataset_dir = download_data()
test_valid_df = get_df(processed_dataset_dir, is_test=True)
test_valid_dls = get_data(test_valid_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

metrics = [MIOU(), BackgroundIOU(), RoadIOU(), TrafficLightIOU(),
           TrafficSignIOU(), PersonIOU(), VehicleIOU(), BicycleIOU()]

cbs = [MixedPrecision()] if config.mixed_precision else []

learn = unet_learner(test_valid_dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, 
                     metrics=metrics)

learn.load(model_path);

val_metrics = learn.validate(ds_idx=1)
test_metrics = learn.validate(ds_idx=0)

val_metric_names = ['val_loss'] + [f'val_{x.name}' for x in learn.metrics]
val_results = {val_metric_names[i] : val_metrics[i] for i in range(len(val_metric_names))}
for k,v in val_results.items(): 
    wandb.summary[k] = v

test_metric_names = ['test_loss'] + [f'test_{x.name}' for x in learn.metrics]
test_results = {test_metric_names[i] : test_metrics[i] for i in range(len(test_metric_names))}
for k,v in test_results.items(): 
    wandb.summary[k] = v
    
log_predictions(learn)

val_probs, val_targs = learn.get_preds(ds_idx=1)
val_preds = val_probs.argmax(dim=1)
class_idxs = params.BDD_CLASSES.keys()

target_counts = count_by_class(val_targs, class_idxs)
pred_counts = count_by_class(val_preds, class_idxs)

for c in class_idxs:
    log_hist(c)
    
val_count_df, val_disp = display_diagnostics(learner=learn, ds_idx=1, return_vals=True)
wandb.log({'val_confusion_matrix': val_disp.figure_})
val_ct_table = wandb.Table(dataframe=val_count_df)
wandb.log({'val_count_table': val_ct_table})

test_count_df, test_disp = display_diagnostics(learner=learn, ds_idx=0, return_vals=True)
wandb.log({'test_confusion_matrix': test_disp.figure_})
test_ct_table = wandb.Table(dataframe=test_count_df)
wandb.log({'test_count_table': test_ct_table})

run.finish()
