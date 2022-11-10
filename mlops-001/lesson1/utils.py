import wandb

from fastai.vision.all import *

import params

CLASS_INDEX = {v:k for k,v in params.BDD_CLASSES.items()}

def iou_per_class(inp, targ):
    "Compute iou per class"
    iou_scores = []
    eps = 1e-6
    for c in range(inp.shape[0]):
        dec_preds = inp.argmax(dim=0)
        p = torch.where(dec_preds == c, 1, 0)
        t = torch.where(targ == c, 1, 0)
        c_inter = (p * t).float().sum().item()
        c_union = (p + t).float().sum().item()
        if c_union:
            iou_scores.append(c_inter / (c_union - c_inter + eps))
        else:
            iou_scores.append(-1)
    return iou_scores

def create_row(sample, pred_label, prediction, class_labels):
    """"A simple function to create a row of (img, target, prediction, and scores...)"""
    (image, label) = sample
    # compute metrics
    iou_scores = iou_per_class(prediction, label)
    image = image.permute(1, 2, 0)
    row =[wandb.Image(
                image,
                masks={
                    "predictions": {
                        "mask_data": pred_label[0].numpy(),
                        "class_labels": class_labels,
                    },
                    "ground_truths": {
                        "mask_data": label.numpy(),
                        "class_labels": class_labels,
                    },
                },
            ),
            *iou_scores,
    ]
    return row

def create_iou_table(samples, outputs, predictions, class_labels):
    "Creates a wandb table with predictions and targets side by side"

    def _to_str(l):
        return [f'{str(x)} IoU' for x in l]
    
    items = list(zip(samples, outputs, predictions))
    
    table = wandb.Table(
        columns=["Image"]
        + _to_str(class_labels.values()),
    )
    # we create one row per sample
    for item in progress_bar(items):
        table.add_data(*create_row(*item, class_labels=class_labels))
    
    return table

def get_predictions(learner, test_dl=None, max_n=None):
    """Return the samples = (x,y) and outputs (model predictions decoded), and predictions (raw preds)"""
    test_dl = learner.dls.valid if test_dl is None else test_dl
    inputs, predictions, targets, outputs = learner.get_preds(
        dl=test_dl, with_input=True, with_decoded=True
    )
    x, y, samples, outputs = learner.dls.valid.show_results(
        tuplify(inputs) + tuplify(targets), outputs, show=False, max_n=max_n
    )
    return samples, outputs, predictions

    def value(self): return self.inter/(self.union-self.inter) if self.union > 0 else None

class MIOU(DiceMulti):
    @property
    def value(self):
        binary_iou_scores = np.array([])
        for c in self.inter:
            binary_iou_scores = np.append(binary_iou_scores, \
                                          self.inter[c]/(self.union[c]-self.inter[c]) if self.union[c] > 0 else np.nan)
        return np.nanmean(binary_iou_scores)
    
@patch
def _iou(self:DiceMulti, nm):
    c=CLASS_INDEX[nm]
    return self.inter[c]/(self.union[c]-self.inter[c]) if self.union[c] > 0 else np.nan

class BackgroundIOU(DiceMulti):
    @property
    def value(self): return self._iou('background')

class RoadIOU(DiceMulti):
    @property
    def value(self): return self._iou('road')
    
class TrafficLightIOU(DiceMulti):
    @property
    def value(self): return self._iou('traffic light')
    
class TrafficSignIOU(DiceMulti):
    @property
    def value(self): return self._iou('traffic sign')  
    
class PersonIOU(DiceMulti):
    @property
    def value(self): return self._iou('person')

class VehicleIOU(DiceMulti):
    @property
    def value(self): return self._iou('vehicle')
    
class BicycleIOU(DiceMulti):
    @property
    def value(self): return self._iou('bicycle')
