# Register models

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/Model_Registry_E2E.ipynb)


The model registry is a central place to house and organize all the model tasks and their associated artifacts being worked on across an org:
- Model checkpoint management
- Document your models with rich model cards
- Maintain a history of all the models being used/deployed
- Facilitate clean hand-offs and stage management of models
- Tag and organize various model tasks
- Set up automatic notifications when models progress

This tutorial will walkthrough how to track the model development lifecycle for a simple image classification task. 
 

### 🛠️ Install `wandb`
 
```bash
!pip install -q wandb onnx pytorch-lightning
```

## Login to W&B
- You can explicitly login using `wandb login` or `wandb.login()` (See below)
- Alternatively you can set environment variables. There are several env variables which you can set to change the behavior of W&B logging. The most important are:
    - `WANDB_API_KEY` - find this in your "Settings" section under your profile 
    - `WANDB_BASE_URL` - this is the url of the W&B server
- Find your API Token in "Profile" -> "Setttings" in the W&B App

![api_token](https://drive.google.com/uc?export=view&id=1Xn7hnn0rfPu_EW0A_-32oCXqDmpA0-kx) 

```notebook
!wandb login
```

:::note
When connecting to a [W&B Server](..//guides/hosting/intro.md) deployment (either **Dedicated Cloud**  or **Self-managed**), use the --relogin and --host options like:

```notebook
!wandb login --relogin --host=http://your-shared-local-host.com
```

If needed, ask your deployment admin for the hostname.
:::

## Log Data and Model Checkpoints as Artifacts  
W&B Artifacts allows you to track and version arbitrary serialized data (e.g. datasets, model checkpoints, evaluation results). When you create an artifact, you give it a name and a type, and that artifact is forever linked to the experimental system of record. If the underlying data changes, and you log that data asset again, W&B will automatically create new versions through checksummming its contents. W&B Artifacts can be thought of as a lightweight abstraction layer on top of shared unstructured file systems. 

### Anatomy of an artifact 

The `Artifact` class will correspond to an entry in the W&B Artifact registry.  The artifact has 
* a name
* a type
* metadata
* description
* files, directory of files, or references

Example usage:
```python
run = wandb.init(project="my-project")
artifact = wandb.Artifact(name="my_artifact", type="data")
artifact.add_file("/path/to/my/file.txt")
run.log_artifact(artifact)
run.finish()
```

In this tutorial, the first thing we will do is download a training dataset and log it as an artifact to be used downstream in the training job.  

```python
# @title Enter your W&B project and entity

# FORM VARIABLES
PROJECT_NAME = "model-registry-tutorial"  # @param {type:"string"}
ENTITY = None  # @param {type:"string"}

# set SIZE to "TINY", "SMALL", "MEDIUM", or "LARGE"
# to select one of these three datasets
# TINY dataset: 100 images, 30MB
# SMALL dataset: 1000 images, 312MB
# MEDIUM dataset: 5000 images, 1.5GB
# LARGE dataset: 12,000 images, 3.6GB

SIZE = "TINY"

if SIZE == "TINY":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_100.zip"
    src_zip = "nature_100.zip"
    DATA_SRC = "nature_100"
    IMAGES_PER_LABEL = 10
    BALANCED_SPLITS = {"train": 8, "val": 1, "test": 1}
elif SIZE == "SMALL":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_1K.zip"
    src_zip = "nature_1K.zip"
    DATA_SRC = "nature_1K"
    IMAGES_PER_LABEL = 100
    BALANCED_SPLITS = {"train": 80, "val": 10, "test": 10}
elif SIZE == "MEDIUM":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    src_zip = "nature_12K.zip"
    DATA_SRC = "inaturalist_12K/train"  # (technically a subset of only 10K images)
    IMAGES_PER_LABEL = 500
    BALANCED_SPLITS = {"train": 400, "val": 50, "test": 50}
elif SIZE == "LARGE":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    src_zip = "nature_12K.zip"
    DATA_SRC = "inaturalist_12K/train"  # (technically a subset of only 10K images)
    IMAGES_PER_LABEL = 1000
    BALANCED_SPLITS = {"train": 800, "val": 100, "test": 100}
```


```notebook
%%capture
!curl -SL $src_url > $src_zip
!unzip $src_zip
```


```python
import wandb
import pandas as pd
import os

with wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="log_datasets") as run:
    img_paths = []
    for root, dirs, files in os.walk("nature_100", topdown=False):
        for name in files:
            img_path = os.path.join(root, name)
            label = img_path.split("/")[1]
            img_paths.append([img_path, label])

    index_df = pd.DataFrame(columns=["image_path", "label"], data=img_paths)
    index_df.to_csv("index.csv", index=False)

    train_art = wandb.Artifact(
        name="Nature_100",
        type="raw_images",
        description="nature image dataset with 10 classes, 10 images per class",
    )
    train_art.add_dir("nature_100")

    # Also adding a csv indicating the labels of each image
    train_art.add_file("index.csv")
    wandb.log_artifact(train_art)
```

### Using Artifact names and aliases to easily hand-off and abstract data assets
- By simply referring to the `name:alias` combination of a dataset or model, we can better standardize components of a workflow
- For instance, you can build PyTorch `Dataset`'s or `DataModule`'s which take as arguments W&B Artifact names and aliases to load appropriately

You can now see all the metadata associated with this dataset, the W&B runs consuming it, and the whole lineage of upstream and downstream artifacts!

![api_token](https://drive.google.com/uc?export=view&id=1fEEddXMkabgcgusja0g8zMz8whlP2Y5P) 

```python
from torchvision import transforms
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io, transform
from torchvision import transforms, utils, models
import math


class NatureDataset(Dataset):
    def __init__(
        self,
        wandb_run,
        artifact_name_alias="Nature_100:latest",
        local_target_dir="Nature_100:latest",
        transform=None,
    ):
        self.local_target_dir = local_target_dir
        self.transform = transform

        # Pull down the artifact locally to load it into memory
        art = wandb_run.use_artifact(artifact_name_alias)
        path_at = art.download(root=self.local_target_dir)

        self.ref_df = pd.read_csv(os.path.join(self.local_target_dir, "index.csv"))
        self.class_names = self.ref_df.iloc[:, 1].unique().tolist()
        self.idx_to_class = {k: v for k, v in enumerate(self.class_names)}
        self.class_to_idx = {v: k for k, v in enumerate(self.class_names)}

    def __len__(self):
        return len(self.ref_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.ref_df.iloc[idx, 0]

        image = io.imread(img_path)
        label = self.ref_df.iloc[idx, 1]
        label = torch.tensor(self.class_to_idx[label], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


class NatureDatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        wandb_run,
        artifact_name_alias: str = "Nature_100:latest",
        local_target_dir: str = "Nature_100:latest",
        batch_size: int = 16,
        input_size: int = 224,
        seed: int = 42,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.artifact_name_alias = artifact_name_alias
        self.local_target_dir = local_target_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.seed = seed

    def setup(self, stage=None):
        self.nature_dataset = NatureDataset(
            wandb_run=self.wandb_run,
            artifact_name_alias=self.artifact_name_alias,
            local_target_dir=self.local_target_dir,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(self.input_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        )

        nature_length = len(self.nature_dataset)
        train_size = math.floor(0.8 * nature_length)
        val_size = math.floor(0.2 * nature_length)
        self.nature_train, self.nature_val = random_split(
            self.nature_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        return self

    def train_dataloader(self):
        return DataLoader(self.nature_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.nature_val, batch_size=self.batch_size)

    def predict_dataloader(self):
        pass

    def teardown(self, stage: str):
        pass
```

## Model Training 

### Writing the Model Class and Validation Function 

```python
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import onnx


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """Alexnet"""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """Squeezenet"""
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = torch.nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """Densenet"""
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class NaturePyTorchModule(torch.nn.Module):
    def __init__(self, model_name, num_classes=10, feature_extract=True, lr=0.01):
        """method used to define our model parameters"""
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.lr = lr
        self.model, self.input_size = initialize_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            feature_extract=True,
        )

    def forward(self, x):
        """method used for inference input -> output"""
        x = self.model(x)

        return x


def evaluate_model(model, eval_data, idx_to_class, class_names, epoch_ndx):
    device = torch.device("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    actual = []

    val_table = wandb.Table(columns=["pred", "actual", "image"])

    with torch.no_grad():
        for data, target in eval_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            preds += list(pred.flatten().tolist())
            actual += target.numpy().tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

            for idx, img in enumerate(data):
                img = img.numpy().transpose(1, 2, 0)
                pred_class = idx_to_class[pred.numpy()[idx][0]]
                target_class = idx_to_class[target.numpy()[idx]]
                val_table.add_data(pred_class, target_class, wandb.Image(img))

    test_loss /= len(eval_data.dataset)
    accuracy = 100.0 * correct / len(eval_data.dataset)
    conf_mat = wandb.plot.confusion_matrix(
        y_true=actual, preds=preds, class_names=class_names
    )
    return test_loss, accuracy, preds, val_table, conf_mat
```

### Tracking the Training Loop
During training, it is a best practice to checkpoint your models overtime, so if training gets interrupted or your instance crashes you can resume from where you left off. With artifact logging, we can track all our checkpoints with W&B and attach any metadata we want (like format of serialization, class labels, etc.). That way, when someone needs to consume a checkpoint they know how to use it. When logging models of any form as artifacts, ensure to set the `type` of the artifact to `model`. 
 

```python
run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    job_type="training",
    config={
        "model_type": "squeezenet",
        "lr": 1.0,
        "gamma": 0.75,
        "batch_size": 16,
        "epochs": 5,
    },
)

model = NaturePyTorchModule(wandb.config["model_type"])
wandb.watch(model)

wandb.config["input_size"] = 224

nature_module = NatureDatasetModule(
    wandb_run=run,
    artifact_name_alias="Nature_100:latest",
    local_target_dir="Nature_100:latest",
    batch_size=wandb.config["batch_size"],
    input_size=wandb.config["input_size"],
)
nature_module.setup()

# Train the model
learning_rate = wandb.config["lr"]
gamma = wandb.config["gamma"]
epochs = wandb.config["epochs"]

device = torch.device("cpu")
optimizer = optim.Adadelta(model.parameters(), lr=wandb.config["lr"])
scheduler = StepLR(optimizer, step_size=1, gamma=wandb.config["gamma"])

best_loss = float("inf")
best_model = None

for epoch_ndx in range(epochs):
    model.train()
    for batch_ndx, batch in enumerate(nature_module.train_dataloader()):
        data, target = batch[0].to("cpu"), batch[1].to("cpu")
        optimizer.zero_grad()
        preds = model(data)
        loss = F.nll_loss(preds, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        ### Log your metrics ###
        wandb.log(
            {
                "train/epoch_ndx": epoch_ndx,
                "train/batch_ndx": batch_ndx,
                "train/train_loss": loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    ### Evaluation at the end of each epoch ###
    model.eval()
    test_loss, accuracy, preds, val_table, conf_mat = evaluate_model(
        model,
        nature_module.val_dataloader(),
        nature_module.nature_dataset.idx_to_class,
        nature_module.nature_dataset.class_names,
        epoch_ndx,
    )

    is_best = test_loss < best_loss

    wandb.log(
        {
            "eval/test_loss": test_loss,
            "eval/accuracy": accuracy,
            "eval/conf_mat": conf_mat,
            "eval/val_table": val_table,
        }
    )

    ### Checkpoing your model weights ###
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    art = wandb.Artifact(
        f"nature-{wandb.run.id}",
        type="model",
        metadata={
            "format": "onnx",
            "num_classes": len(nature_module.nature_dataset.class_names),
            "model_type": wandb.config["model_type"],
            "model_input_size": wandb.config["input_size"],
            "index_to_class": nature_module.nature_dataset.idx_to_class,
        },
    )

    art.add_file("model.onnx")

    ### Add aliases to keep track of your best checkpoints over time
    wandb.log_artifact(art, aliases=["best", "latest"] if is_best else None)
    if is_best:
        best_model = art
```

### Manage all your model checkpoints for a project under one roof. 

![api_token](https://drive.google.com/uc?export=view&id=1z7nXRgqHTPYjfR1SoP-CkezyxklbAZlM) 
### Note: Syncing with W&B Offline
If for some reason, network communication is lost during the course of training, you can always sync progress with `wandb sync`

The W&B sdk caches all logged data in a local directory `wandb` and when you call `wandb sync`, this syncs the your local state with the web app.  
## Model Registry 
After logging a bunch of checkpoints across multiple runs during experimentation, now comes time to hand-off the best checkpoint to the next stage of the workflow (e.g. testing, deployment). 

The Model Registry is a central page that lives above individual W&B projects. It houses **Registered Models**, portfolios that store "links" to the valuable checkpoints living in individual W&B Projects. 

The model registry offers a centralized place to house the best checkpoints for all your model tasks. Any `model` artifact you log can be "linked" to a Registered Model.  
### Creating **Registered Models** and Linking through the UI
#### 1. Access your team's model registry by going the team page and selecting `Model Registry`

![model registry](https://drive.google.com/uc?export=view&id=1ZtJwBsFWPTm4Sg5w8vHhRpvDSeQPwsKw)

#### 2. Create a new Registered Model. 

![model registry](https://drive.google.com/uc?export=view&id=1RuayTZHNE0LJCxt1t0l6-2zjwiV4aDXe)

#### 3. Go to the artifacts tab of the project that holds all your model checkpoints

![model registry](https://drive.google.com/uc?export=view&id=1LfTLrRNpBBPaUb_RmBIE7fWFMG0h3e0E)

#### 4. Click "Link to Registry" for the model artifact version you want.  
### Creating Registered Models and Linking through the **API**
You can [link a model via api](https://docs.wandb.ai/guides/models) with `wandb.run.link_artifact` passing in the artifact object, and the name of the **Registered Model**, along with aliases you want to append to it. **Registered Models** are entity (team) scoped in W&B so only members of a team can see and access the **Registered Models** there. You indicate a registered model name via api with `<entity>/model-registry/<registered-model-name>`. If a Registered Model doesn't exist, one will be created automatically. 

```python
if ENTITY:
    wandb.run.link_artifact(
        best_model,
        f"{ENTITY}/model-registry/Model Registry Tutorial",
        aliases=["staging"],
    )
else:
    print("Must indicate entity where Registered Model will exist")
wandb.finish()
```

### What is "Linking"?
When you link to the registry, this creates a new version of that Registered Model, which is just a pointer to the artifact version living in that project. There's a reason W&B segregates the versioning of artifacts in a project from the versioning of a Registered Model. The process of linking a model artifact version is equivalent to "bookmarking" that artifact version under a Registered Model task. 

Typically during R&D/experimentation, researchers generate 100s, if not 1000s of model checkpoint artifacts, but only one or two of them actually "see the light of day." This process of linking those checkpoints to a separate, versioned registry helps delineate the model development side from the model deployment/consumption side of the workflow. The globally understood version/alias of a model should be unpolluted from all the experimental versions being generated in R&D and thus the versioning of a Registered Model increments according to new "bookmarked" models as opposed to model checkpoint logging.  
## Create a Centralized Hub for all your models
- Add a model card, tags, slack notifactions to your Registered Model
- Change aliases to reflect when models move through different phases
- Embed the model registry in reports for model documentation and regression reports. See this report as an [example](https://api.wandb.ai/links/wandb-smle/r82bj9at)
![model registry](https://drive.google.com/uc?export=view&id=1lKPgaw-Ak4WK_91aBMcLvUMJL6pDQpgO)
 
### Set up Slack Notifications when new models get linked to the registry

![model registry](https://drive.google.com/uc?export=view&id=1RsWCa6maJYD5y34gQ0nwWiKSWUCqcjT9) 
## Consuming a Registered Model
You now can consume any registered model via API by referring the corresponding `name:alias`. Model consumers, whether they are engineers, researchers, or CI/CD processes, can go to the model registry as the central hub for all models that should "see the light of day": those that need to go through testing or move to production.  

```notebook
%%wandb -h 600

run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='inference')
artifact = run.use_artifact(f'{ENTITY}/model-registry/Model Registry Tutorial:staging', type='model')
artifact_dir = artifact.download()
wandb.finish()
```

# What's next?
In the next tutorial, you will learn how to iterate on large language models and debug using W&B Prompts:
## 👉 [Iterate on LLMs](prompts)