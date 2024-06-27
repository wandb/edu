---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Setup

### How can I configure the name of the run in my training code?

At the top of your training script when you call `wandb.init`, pass in an experiment name, like this: `wandb.init(name="my_awesome_run")`.

### Can I run wandb offline?

If you're training on an offline machine and want to upload your results to our servers afterwards, we have a feature for you!

1. Set the environment variable `WANDB_MODE=offline` to save the metrics locally, no internet required.
2. When you're ready, run `wandb init` in your directory to set the project name.
3. Run `wandb sync YOUR_RUN_DIRECTORY` to push the metrics to our cloud service and see your results in our hosted web app.

You can check via API whether your run is offline by using `run.settings._offline` or `run.settings.mode` after your wandb.init().

#### Some use-cases where you can use [`wandb sync`](../../ref/cli/wandb-sync.md)

* If you don’t have internet.
* If you need to fully disable things.
* To sync your run later due to any reason. For instance: if you want to avoid using resources on a training machine.

### Does this only work for Python?

Currently, the library only works with Python 2.7+ & 3.6+ projects. The architecture mentioned above should enable us to integrate with other languages easily. If you have a need for monitoring other languages, send us a note at [contact@wandb.com](mailto:contact@wandb.com).

### Is there an anaconda package?

Yes! You can either install with `pip` or with `conda`. For the latter, you'll need to get the package from the [conda-forge](https://conda-forge.org) channel.

<Tabs
  defaultValue="pip"
  values={[
    {label: 'pip', value: 'pip'},
    {label: 'conda', value: 'conda'},
  ]}>
  <TabItem value="pip">

```bash
# Create a conda env
conda create -n wandb-env python=3.8 anaconda
# Activate created env
conda activate wandb-env
# install wandb with pip in this conda env
pip install wandb
```

  </TabItem>
  <TabItem value="conda">

```
conda activate myenv
conda install wandb --channel conda-forge
```

  </TabItem>
</Tabs>


If you run into issues with this install, please let us know. This Anaconda [doc on managing packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) has some helpful guidance.

### How do I install the wandb Python library in environments without gcc?

If you try to install `wandb` and see this error:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

You can install `psutil` directly from a pre-built wheel. Find your Python version and OS here: [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil)

For example, to install `psutil` on Python 3.8 in Linux:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

After `psutil` has been installed, you can install wandb with `pip install wandb`.

### Does the W&B client support Python 2? <a href="#eol-python27" id="eol-python27"></a>

The W&B client library supported both Python 2.7 and Python 3 through version 0.10. Due to the Python 2 end of life, support for Python 2.7 was discontinued as of version 0.11. Users who run`pip install --upgrade wandb` on a Python 2.7 system will get new releases of the 0.10.x series only. Support for the 0.10.x series will be limited to critical bugfixes and patches. Currently, version 0.10.33 is the last version of the 0.10.x series that supports Python 2.7.

### Does the W&B client support Python 3.5? <a href="#eol-python35" id="eol-python35"></a>

The W&B client library supported both Python 3.5 through version 0.11. Due to the Python 3.5 end of life, support was discontinued as of [version 0.12](https://github.com/wandb/wandb/releases/tag/v0.12.0).
