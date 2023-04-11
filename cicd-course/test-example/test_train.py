import train
from fastai.vision.all import SimpleNamespace

def test_train():
    default_config = SimpleNamespace(
        framework="fastai",
        img_size=30, # small size for the smoke test
        batch_size=5, # low bs to fit on CPU if needed
        augment=True, # use data augmentation
        epochs=1,
        lr=2e-3,
        pretrained=True,  # whether to use pretrained encoder,
        mixed_precision=True, # use automatic mixed precision
        arch="resnet18",
        seed=42,
        log_preds=False,
    )
    train.train(default_config, nrows=20)
