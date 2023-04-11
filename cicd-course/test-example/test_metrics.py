import torch
import utils
from fastai.vision.all import Learner
import math

#For testing: a fake learner and a metric that isn't an average
class TstLearner(Learner):
    def __init__(self,dls=None,model=None,**kwargs): self.pred,self.xb,self.yb = None,None,None

#Go through a fake cycle with various batch sizes and computes the value of met
def compute_val(met, x1, x2):
    met.reset()
    vals = [0,6,15,20]
    learn = TstLearner()
    for i in range(3):
        learn.pred,learn.yb = x1[vals[i]:vals[i+1]],(x2[vals[i]:vals[i+1]],)
        met.accumulate(learn)
    return met.value
    
def test_metrics():
    x1a = torch.ones(20,1,1,1) # predicting background pixels
    x1b = torch.clone(x1a)*0.3
    x1c = torch.clone(x1a)*0.1
    x1 = torch.cat((x1a,x1b,x1c),dim=1)   # Prediction: 20xClass0
    x2 = torch.zeros(20,1,1)              # Target: 20xClass0

    assert compute_val(utils.BackgroundIOU(), x1, x2) == 1.
    road_iou = compute_val(utils.RoadIOU(), x1, x2)
    assert math.isnan(road_iou)


    x1b = torch.ones(20,1,1,1) # predicting road pixels 
    x1a = torch.clone(x1a)*0.3
    x1c = torch.clone(x1a)*0.1
    x1 = torch.cat((x1a,x1b,x1c),dim=1)   # Prediction: 20xClass1
    x2 = torch.ones(20,1,1)               # Target: 20xClass1
    background_iou = compute_val(utils.BackgroundIOU(), x1, x2)
    assert math.isnan(background_iou)
    assert compute_val(utils.RoadIOU(), x1, x2) == 1.