---
displayed_sidebar: default
---

# General

### What does `wandb.init` do to my training process?

When `wandb.init()` is called from your training script an API call is made to create a run object on our servers. A new process is started to stream and collect metrics, thereby keeping all threads and logic out of your primary process. Your script runs normally and writes to local files, while the separate process streams them to our servers along with system metrics. You can always turn off streaming by running `wandb off` from your training directory, or setting the `WANDB_MODE` environment variable to `offline`.

### Does your tool track or store training data?

You can pass a SHA or other unique identifier to `wandb.config.update(...)` to associate a dataset with a training run. W&B does not store any data unless `wandb.save` is called with the local file name.

### What formula do you use for your smoothing algorithm?

We use the same exponential moving average formula as TensorBoard. You can find an [explanation here](https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar).

### How do I get the random run name in my script?

Call `wandb.run.save()` and then get the name with `wandb.run.name` .

### What is the difference between `.log()` and `.summary`?

The summary is the value that shows in the table while the log will save all the values for plotting later.

For example, you might want to call `wandb.log` every time the accuracy changes. Usually, you can just use .log. `wandb.log()` will also update the summary value by default unless you have set the summary manually for that metric

The scatterplot and parallel coordinate plots will also use the summary value while the line plot plots all of the values set by .log

The reason we have both is that some people like to set the summary manually because they want the summary to reflect for example the optimal accuracy instead of the last accuracy logged.

### How is W&B different from TensorBoard?

We love the TensorBoard folks, and we have a [TensorBoard integration](../integrations/tensorboard.md)! We were inspired to improve experiment tracking tools for everyone. When the co-founders started working on W&B, they were inspired to build a tool for the frustrated TensorBoard users at OpenAI. Here are a few things we focused on improving:

1. **Reproduce models**: W&B is good for experimentation, exploration, and reproducing models later. We capture not just the metrics, but also the hyperparameters and version of the code, and we can save your model checkpoints for you so your project is reproducible.
2. **Automatic organization**: If you hand off a project to a collaborator or take a vacation, W&B makes it easy to see all the models you've tried so you're not wasting hours re-running old experiments.
3. **Fast, flexible integration**: Add W&B to your project in 5 minutes. Install our free open-source Python package and add a couple of lines to your code, and every time you run your model you'll have nice logged metrics and records.
4. **Persistent, centralized dashboard**: Anywhere you train your models, whether on your local machine, your lab cluster, or spot instances in the cloud, we give you the same centralized dashboard. You don't need to spend your time copying and organizing TensorBoard files from different machines.
5. **Powerful table**: Search, filter, sort, and group results from different models. It's easy to look over thousands of model versions and find the best-performing models for different tasks. TensorBoard isn't built to work well on large projects.
6. **Tools for collaboration**: Use W&B to organize complex machine learning projects. It's easy to share a link to W&B, and you can use private teams to have everyone send results to a shared project. We also support collaboration via reports— add interactive visualizations and describe your work in markdown. This is a great way to keep a work log, share findings with your supervisor, or present findings to your lab.

Get started with a [free account](http://app.wandb.ai)

### How does wandb stream logs and writes to disk?

W&B queues in memory but also [write the events to disk](https://github.com/wandb/wandb/blob/7cc4dd311f3cdba8a740be0dc8903075250a914e/wandb/sdk/internal/datastore.py) asynchronously to handle failures and for the `WANDB_MODE=offline` case where you can sync the data after it's been logged.

In your terminal, you can see a path to the local run directory. This directory will contain a `.wandb` file that is the datastore above. If you're also logging images, we write them to `media/images` in that directory before uploading them to cloud storage.

### How to get multiple charts with different selected runs?

With wandb reports the procedure is as follows:

* Have multiple panel grids.
* Add filters to filter the run sets of each panel grid. This will help in selecting the runs that you want to portray in the respective panels.
* Create the charts you want in the panel grids.

### How is access to the API controlled?

For simplicity, W&B uses API keys for authorization when accessing the API. You can find your API keys in your [settings](https://app.wandb.ai/settings). Your API key should be stored securely and never checked into version control. In addition to personal API keys, you can add Service Account users to your team.

### Does W&B support SSO for Multi-tenant?

Yes, W&B supports setting up Single Sign-On (SSO) for the Multi-tenant offering via Auth0. W&B support SSO integration with any OIDC compliant identity provider(ex: Okta, AzureAD etc.). If you have an OIDC provider, please follow the steps below:

* Create a `Single Page Application (SPA)` on your Identity Provider.
* Set `grant_type` to `implicit` flow.
* Set the callback URI to `https://wandb.auth0.com/login/callback`.

**What W&B needs?**

Once you have the above setup, contact your customer success manager(CSM) and let us know the `Client ID` and `Issuer URL` associated with the application.

We'll then set up an Auth0 connection with the above details and enable SSO.

### What is a service account, and why is it useful?

A service account (Enterprise-only feature) is an API key that has permissions to write to your team, but is not associated with a particular user. Among other things, service accounts are useful for tracking automated jobs logged to wandb, like periodic retraining, nightly builds, and so on. If you'd like, you can associate a username with one of these machine-launched runs with the [environment variable](../track/environment-variables.md) `WANDB_USERNAME`.

Refer to [Team Service Account Behavior](../app/features/teams.md#team-service-account-behavior) for more information.

You can get the API key in your Team Settings page `/teams/<your-team-name>` where you invite new team members. Select service and click create to add a service account.

![Create a service account on your team settings page for automated jobs](/images/technical_faq/what_is_service_account.png)

### How can I rotate or revoke access?

Both personal and service account keys can be rotated or revoked. Simply create a new API Key or Service Account user and reconfigure your scripts to use the new key. Once all processes are reconfigured, you can remove the old API key from your profile or team.

### How do I switch between accounts on the same machine?

If you have two W&B accounts working from the same machine, you'll need a nice way to switch between your different API keys. You can store both API keys in a file on your machine then add code like the following to your repos. This is to avoid checking your secret key into a source control system, which is potentially dangerous.

```python
if os.path.exists("~/keys.json"):
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```

### Is there a dark mode?

Yes. To enable dark mode:

1. Navigate to your account settings at [https://wandb.ai/settings](https://wandb.ai/settings).
2. Scroll to the **Beta Features** section.
3. Toggle the **Night mode** option.

### Can I disable wandb when testing my code?

By using `wandb.init(mode="disabled")` or by setting `WANDB_MODE=disabled` you will make wandb act like a NOOP which is perfect for testing your code.

**Note**: Setting `wandb.init(mode=“disabled”)` does not prevent `wandb` from saving artifacts to `WANDB_CACHE_DIR`
