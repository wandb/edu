import os
from pathlib import Path
import requests
import shutil

import pytorch_lightning as pl
import torch
import wandb

try:
    import torchviz
    no_torchviz = False
except ImportError:
    no_torchviz = True


class LoggedLitModule(pl.LightningModule):
    """LightningModule plus wandb features and simple training/val steps.
    Assumes that your training loop involves inputs (xs)
    producing outputs (y_hats)
    that are compared to targets (ys)
    by a loss and by metrics,
    where each batch == (xs, ys).
    """

    def __init__(self, max_logged_inputs=0):
        super().__init__()

        self.training_metrics = torch.nn.ModuleList([])
        self.validation_metrics = torch.nn.ModuleList([])

        self.max_logged_inputs = max_logged_inputs
        self.graph_logged = False

    def on_pretrain_routine_start(self):
        print(self)
        print(f"Parameter Count: {self.count_params()}")

    def training_step(self, xys, idx):
        xs, ys = xys
        y_hats = self.forward(xs)
        loss = self.loss(y_hats, ys)

        logging_scalars = {"loss": loss}
        for metric in self.training_metrics:
            self.add_metric(metric, logging_scalars, y_hats, ys)

        self.do_logging(xs, ys, idx, y_hats, logging_scalars)

        return loss

    def validation_step(self, xys, idx):
        xs, ys = xys
        y_hats = self.forward(xs)
        loss = self.loss(y_hats, ys)

        logging_scalars = {"loss": loss}
        for metric in self.training_metrics:
            self.add_metric(metric, logging_scalars, y_hats, ys)

        self.do_logging(xs, ys, idx, y_hats, logging_scalars, step="validation")

    def do_logging(self, xs, ys, idx, y_hats, scalars, step="training"):
        self.log_dict(
            {step + "/" + name: value for name, value in scalars.items()})

        if idx == 0:
            if "x_range" not in wandb.run.config.keys():
                wandb.run.config["x_range"] = [float(torch.min(xs)), float(torch.max(xs))]
            if "loss" not in wandb.run.config.keys():
                wandb.run.config["loss"] = self.detect_loss()
            if "optimizer" not in wandb.run.config.keys():
                wandb.run.config["optimizer"] = self.detect_optimizer()
            if "nparams" not in wandb.run.config.keys():
                wandb.run.config["nparams"] = self.count_params()
            if "dropout" not in wandb.run.config.keys():
                wandb.run.config["dropout"] = self.detect_dropout()
            if step == "training":
                if self.max_logged_inputs > 0:
                    self.log_examples(xs, ys, y_hats)
                if not (self.graph_logged or no_torchviz):
                    self.log_graph(y_hats)

    def detect_loss(self):
        classname = self.loss.__class__.__name__
        if classname in ["method", "function"]:
            return "unknown"
        else:
            return classname

    def detect_optimizer(self):
        return self.optimizers().__class__.__name__

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def detect_dropout(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                return module.p
        return 0

    def log_graph(self, y_hats):
        params_dict = dict(list(self.named_parameters()))
        graph = torchviz.make_dot(y_hats, params=params_dict)
        graph.format = "png"
        fname = Path(self.logger.experiment.dir) / "graph"
        graph.render(fname)
        wandb.save(str(fname.with_suffix("." + graph.format)))
        self.graph_logged = True

    def log_examples(*args, **kwargs):
        raise NotImplementedError
        
    def add_metric(self, metric, logging_scalars, y_hats, ys):
        metric_str = metric.__class__.__name__.lower()
        value = metric(y_hats, ys)
        logging_scalars[metric_str] = value


class LoggedImageClassifierModule(LoggedLitModule):

    def __init__(self, max_images_to_display=32, labels=None):

        super().__init__(max_logged_inputs=max_images_to_display)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

        self.training_metrics.append(self.train_acc)
        self.validation_metrics.append(self.valid_acc)

        self.labels = labels

    def log_examples(self, xs, ys, y_hats):
        xs, ys, y_hats = (xs[:self.max_logged_inputs],
                          ys[:self.max_logged_inputs],
                          y_hats[:self.max_logged_inputs])

        preds = self.preds_from_y_hats(y_hats)
        
        if self.labels is not None:
            preds = [self.labels[int(pred)] for pred in preds]

        images_with_predictions = [
            wandb.Image(x, caption=f"Pred: {pred}")
            for x, pred in zip(xs, preds)]

        self.logger.experiment.log({"predictions": images_with_predictions,
                                    "global_step": self.global_step}, commit=False)
        
    def add_metric(self, metric, logging_scalars, y_hats, ys):
        metric_str = metric.__class__.__name__.lower()
        if metric_str == "accuracy":
            preds = self.preds_from_y_hats(y_hats)
            value = metric(preds, ys)
        else:
            value = metric(y_hats, ys)
        logging_scalars[metric_str] = value
        
    @staticmethod
    def preds_from_y_hats(y_hats):
        if y_hats.shape[-1] == 1:  # handle single-class case
            preds = torch.greater(y_hats, 0.5)
            preds = [bool(pred) for pred in preds]
        else:  # assume we are in the typical one-hot case
            preds = torch.argmax(y_hats, 1)
        return preds


class MNISTDataModule(pl.LightningDataModule):
    """DataModule for the MNIST handwritten digit dataset.
  
    Only downloads the training set, but performs a validation split in the
    setup step.
  
    Converts images to float and normalizes to [0, 1] in setup.
    """

    def __init__(self, batch_size=64):
        super().__init__()  # ⚡: we inherit from LightningDataModule
        self.batch_size = batch_size

    def setup(self, validation_size=10_000): # ⚡: how do we set up the data?
        # download the data from the internet
        if not os.path.exists("MNIST"):
            self._download_MNIST(dir=".")
        mnist = torchvision.datasets.MNIST(".", train=True, download=False)

        # set up shapes and types
        self.digits = mnist.data.float()
        self.digits = torch.divide(self.digits, 255.)
    
        self.training_data = torch.utils.data.TensorDataset(self.digits[:-validation_size, None, :, :],
                                                            self.digits[:-validation_size, None, :, :])
        self.validation_data = torch.utils.data.TensorDataset(self.digits[-validation_size:, None, :, :],
                                                              self.digits[-validation_size:, None, :, :])
        self.validation_size = validation_size

    def train_dataloader(self):  # ⚡: how do we go from dataset to dataloader?
        """The DataLoaders returned by a DataModule produce data for a model.
        
        This DataLoader is used during training."""
        return DataLoader(self.training_data, batch_size=self.batch_size)

    def val_dataloader(self):  # ⚡: what about during validation?
        """The DataLoaders returned by a DataModule produce data for a model.
        
        This DataLoader is used during validation, at the end of each epoch."""
        return DataLoader(self.validation_data, batch_size=self.validation_size)

    @staticmethod
    def _download_MNIST(url=None, dir="."):
        if url is None:
            url = "http://www.di.ens.fr/~lelarge/MNIST.tar.gz"
        compressed_bytes = requests.get(url)
      
        path = Path(dir) / "MNIST.tar.gz"
        with open(path, "wb") as f:
            f.write(compressed_bytes.content)
      
        shutil.unpack_archive(path)
        
        
class FilterLogger(pl.Callback):
    """PyTorch Lightning Callback for logging the "filters" of a PyTorch Module.
  
    Filters are weights that touch input or output, and so are often interpretable.
    In particular, these weights are most often interpretable for networks that
    consume or produce images, because they can be viewed as images.
  
    This Logger selects the input and/or output filters (set by log_input and
    log_output boolean flags) for logging and sends them to Weights & Biases as
    images.
    """
    def __init__(self, image_size, log_input=False, log_output=False):
        super().__init__()
        if len(image_size) == 2:
            image_size = [1] + list(image_size)
        self.image_size = image_size
        self.log_input, self.log_output = log_input, log_output
  
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.log_input:
            input_filters = self.fetch_filters(pl_module, reversed=False, output_shape=self.image_size)
            self.log_filters(input_filters, "filters/input", trainer)
    
        if self.log_output:
            output_filters = self.fetch_filters(pl_module, reversed=True, output_shape=self.image_size)
            self.log_filters(output_filters, "filters/output", trainer)
        
    def log_filters(self, filters, key, trainer):
        trainer.logger.experiment.log({
            key: wandb.Image(filters.cpu()),
            "global_step": trainer.global_step
        })
  
    def fetch_filters(self, module, reversed=False, output_shape=None):
    
        weights = self.get_weights(module)
        assert len(weights), "could not find any weights"

        if reversed:
            filter_weights = torch.transpose(weights[-1], -2, -1)
        else:
            filter_weights = weights[0]
      
        filters = self.extract_filters(filter_weights, output_shape=output_shape)
      
        return filters
    
    def extract_filters(self, filter_weights, output_shape=None):
        is_convolutional = len(filter_weights.shape) == 4
        if is_convolutional:
            assert filter_weights.shape[1] in [1, 3], "convolutional filters must return luminance or RGB"
            return filter_weights
        else:
            assert len(filter_weights.shape) == 2, "last weights in module neither convolutional nor linear"
            assert output_shape is not None, "no output_shape provided but last weights are linear"
            filter_weights = self.reshape_linear_weights(filter_weights, output_shape)
            return filter_weights
    
    @staticmethod
    def reshape_linear_weights(filter_weights, output_shape):
        assert len(output_shape) >= 2, "output_shape must be H x W"
        assert np.prod(output_shape) == filter_weights.shape[1], "filter_weights did not match output_shape"
        return torch.reshape(filter_weights, [-1] + list(output_shape))
    
    @staticmethod
    def get_weights(module):
        weights = [parameter for name, parameter in module.named_parameters() if name.endswith("weight")]
        return weights
    
    
class ImageLogger(pl.Callback):
    """Logs the input and output and output images produced by a module.
  
    Useful in combination with, e.g., an autoencoder architecture,
    a convolutional GAN, or any image-to-image transformation network.
    """
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, _ = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
    
        outs = pl_module(val_imgs)
    
        mosaics = torch.cat([outs, val_imgs], dim=-2)
        caption = f"Top: Output, Bottom: Input"
        trainer.logger.experiment.log({
            "test/examples": [wandb.Image(mosaic, caption=caption) 
                              for mosaic in mosaics],
            "global_step": trainer.global_step
            })
