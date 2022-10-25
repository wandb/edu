from fastai.vision.all import *
import wandb

def faccuracy(inp, targ):
    "Foreground accuracy per item"
    return foreground_acc(inp.unsqueeze(0), targ.unsqueeze(0)).item()

def dice_per_class(inp, targ):
    "Compute dice per class"
    dice_scores = []
    eps = 1e-6
    for c in range(inp.shape[0]):
        dec_preds = inp.argmax(dim=0)
        p = torch.where(dec_preds == c, 1, 0)
        t = torch.where(targ == c, 1, 0)
        c_inter = (p * t).float().sum().item()
        c_union = (p + t).float().sum().item()
        if c_union:
            dice_scores.append(2.0 * c_inter / (c_union + eps))
        else:
            dice_scores.append(-1)
    return dice_scores

def create_row(sample, pred_label, prediction, class_labels):
    """"A simple function to create a row of (img, target, prediction, and scores...)"""
    (image, label) = sample
    # compute metrics
    dice_scores = dice_per_class(prediction, label)
    facc = faccuracy(prediction, label)
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
            *dice_scores,
            facc,
    ]
    return row

def create_dice_table(samples, outputs, predictions, class_labels):
    "Creates a wandb table with predictions and targets side by side"

    def _to_str(l):
        return [str(x) for x in l]
    
    items = list(zip(samples, outputs, predictions))
    
    table = wandb.Table(
        columns=["Image"]
        + _to_str(class_labels.values())
        + ["Foreground Acc"],
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