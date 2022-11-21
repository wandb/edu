import wandb
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display, Markdown
from fastai.vision.all import *

import params

CLASS_INDEX = {v:k for k,v in params.BDD_CLASSES.items()}

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False

def iou_per_class(inp, targ):
    "Compute iou per class"
    iou_scores = []
    for c in range(inp.shape[0]):
        dec_preds = inp.argmax(dim=0)
        p = torch.where(dec_preds == c, 1, 0)
        t = torch.where(targ == c, 1, 0)
        c_inter = (p * t).float().sum().item()
        c_union = (p + t).float().sum().item()
        iou_scores.append(c_inter / (c_union - c_inter) if c_union > 0 else np.nan)
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
    
class IOU(DiceMulti):
    @property
    def value(self): 
        c=CLASS_INDEX[self.nm]
        return self.inter[c]/(self.union[c]-self.inter[c]) if self.union[c] > 0 else np.nan

class BackgroundIOU(IOU): nm = 'background'
class RoadIOU(IOU): nm = 'road'
class TrafficLightIOU(IOU): nm = 'traffic light'
class TrafficSignIOU(IOU): nm = 'traffic sign'
class PersonIOU(IOU): nm = 'person'
class VehicleIOU(IOU): nm = 'vehicle'
class BicycleIOU(IOU): nm = 'bicycle'


class IOUMacro(DiceMulti):
    @property
    def value(self): 
        c=CLASS_INDEX[self.nm]
        if c not in self.count: return np.nan
        else: return self.macro[c]/self.count[c] if self.count[c] > 0 else np.nan

    def reset(self): self.macro,self.count = {},{}

    def accumulate(self, learn):
        pred,targ = learn.pred.argmax(dim=self.axis), learn.y
        for c in range(learn.pred.shape[self.axis]):
            p = torch.where(pred == c, 1, 0)
            t = torch.where(targ == c, 1, 0)
            c_inter = (p*t).float().sum(dim=(1,2))
            c_union = (p+t).float().sum(dim=(1,2))
            m = c_inter / (c_union - c_inter)
            macro = m[~torch.any(m.isnan())]
            count = macro.shape[1]

            if count > 0:
                msum = macro.sum().item()
                if c in self.count:
                    self.count[c] += count
                    self.macro[c] += msum
                else:
                    self.count[c] = count
                    self.macro[c] = msum


class MIouMacro(IOUMacro):
    @property
    def value(self):
        binary_iou_scores = np.array([])
        for c in self.count:
            binary_iou_scores = np.append(binary_iou_scores, self.macro[c]/self.count[c] if self.count[c] > 0 else np.nan)
        return np.nanmean(binary_iou_scores)


class BackgroundIouMacro(IOUMacro): nm = 'background'
class RoadIouMacro(IOUMacro): nm = 'road'
class TrafficLightIouMacro(IOUMacro): nm = 'traffic light'
class TrafficSignIouMacro(IOUMacro): nm = 'traffic sign'
class PersonIouMacro(IOUMacro): nm = 'person'
class VehicleIouMacro(IOUMacro): nm = 'vehicle'
class BicycleIouMacro(IOUMacro): nm = 'bicycle'


def display_diagnostics(learner, dls=None, return_vals=False):
    """
    Display a confusion matrix for the unet learner.
    If `dls` is None it will get the validation set from the Learner
    
    You can create a test dataloader using the `test_dl()` method like so:
    >> dls = ... # You usually create this from the DataBlocks api, in this library it is get_data()
    >> tdls = dls.test_dl(test_dataframe, with_labels=True)
    
    See: https://docs.fast.ai/tutorial.pets.html#adding-a-test-dataloader-for-inference
    
    """
    probs, targs = learner.get_preds(dl = dls)
    preds = probs.argmax(dim=1)
    classes = list(params.BDD_CLASSES.values())
    y_true = targs.flatten().numpy()
    y_pred = preds.flatten().numpy()
    
    tdf, pdf = [pd.DataFrame(r).value_counts().to_frame(c) for r,c in zip((y_true, y_pred) , ['y_true', 'y_pred'])]
    countdf = tdf.join(pdf, how='outer').reset_index(drop=True).fillna(0).astype(int).rename(index= params.BDD_CLASSES)
    countdf = countdf/countdf.sum() 
    display(Markdown('### % Of Pixels In Each Class'))
    display(countdf.style.format('{:.1%}'))
    
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred,
                                                   display_labels=classes,
                                                   normalize='pred')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(10) 
    disp.ax_.set_title('Confusion Matrix (by Pixels)', fontdict={'fontsize': 32, 'fontweight': 'medium'})
    fig.show()
    
    if return_vals: return countdf, disp

