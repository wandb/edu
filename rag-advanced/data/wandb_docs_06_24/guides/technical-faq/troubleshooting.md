---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Troubleshooting

### If wandb crashes, will it possibly crash my training run?

It is extremely important to us that we never interfere with your training runs. We run wandb in a separate process to make sure that if wandb somehow crashes, your training will continue to run. If the internet goes out, wandb will continue to retry sending data to [wandb.ai](https://wandb.ai).

### Why is a run marked crashed in W&B when it’s training fine locally?

This is likely a connection problem — if your server loses internet access and data stops syncing to W&B, we mark the run as crashed after a short period of retrying.

### Does logging block my training?

"Is the logging function lazy? I don't want to be dependent on the network to send the results to your servers and then carry on with my local operations."

Calling `wandb.log` writes a line to a local file; it does not block any network calls. When you call `wandb.init` we launch a new process on the same machine that listens for filesystem changes and talks to our web service asynchronously from your training process.

### How do I stop wandb from writing to my terminal or my jupyter notebook output?

Set the environment variable [`WANDB_SILENT`](../track/environment-variables.md) to `true`.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Jupyter Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'command-line'},
  ]}>
  <TabItem value="python">

```python
os.environ["WANDB_SILENT"] = "true"
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="command-line">

```python
WANDB_SILENT=true
```

  </TabItem>
</Tabs>


### How do I kill a job with wandb?

Press `Ctrl+D` on your keyboard to stop a script that is instrumented with wandb.

### How do I deal with network issues?

If you're seeing SSL or network errors:`wandb: Network error (ConnectionError), entering retry loop`. You can try a couple of different approaches to solving this issue:

1. Upgrade your SSL certificate. If you're running the script on an Ubuntu server, run `update-ca-certificates` We can't sync training logs without a valid SSL certificate because it's a security vulnerability.
2. If your network is flaky, run training in [offline mode](../track/launch.md) and sync the files to us from a machine that has Internet access.
3. Try running [W&B Private Hosting](../hosting/intro.md), which operates on your machine and doesn't sync files to our cloud servers.

`SSL CERTIFICATE_VERIFY_FAILED`: this error could be due to your company's firewall. You can set up local CAs and then use:

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`

### What happens if internet connection is lost while I'm training a model?

If our library is unable to connect to the internet it will enter a retry loop and keep attempting to stream metrics until the network is restored. During this time your program is able to continue running.

If you need to run on a machine without internet, you can set `WANDB_MODE=offline` to only have metrics stored locally on your hard drive. Later you can call `wandb sync DIRECTORY` to have the data streamed to our server.
