---
description: Send alerts, triggered from your Python code, to your Slack or email
displayed_sidebar: default
---

# Send Alerts with wandb.alert

<head>
  <title>Send Alerts from your Python Code</title>
</head>


[**Try in a Colab Notebook here →**](http://wandb.me/alerts-colab)

With W&B Alerts you can be notified via Slack or email if your W&B Run has crashed or whether a custom trigger, such as your loss going to NaN or a step in your ML pipeline has completed, has been reached. W&B Alerts apply all projects where you launch runs, including both personal and Team projects.

You can set an alert like this:

```python
text = f"Accuracy {acc} is below acceptable threshold {thresh}"

wandb.alert(title="Low accuracy", text=text)
```

And then see W&B Alerts messages in Slack (or your email):

![](/images/track/send_alerts_slack.png)

## Getting Started

:::info
Please note that these following steps are to turn on alerts in public cloud _only_.

If you're using [W&B Server](../hosting/intro.md) in your Private Cloud or on W&B Dedicated Cloud, then please refer to [this documentation](../hosting/monitoring-usage/slack-alerts.md) to setup Slack alerts.
:::


There are 2 steps to follow the first time you'd like to send a Slack or email alert, triggered from your code:

1. Turn on Alerts in your W&B [User Settings](https://wandb.ai/settings)
2. Add `wandb.alert()` to your code

### 1. Turn on Alerts in your W&B User Settings

In your [User Settings](https://wandb.ai/settings):

* Scroll to the **Alerts** section
* Turn on **Scriptable run alerts** to receive alerts from `wandb.alert()`
* Use **Connect Slack** to pick a Slack channel to post alerts. We recommend the **Slackbot** channel because it keeps the alerts private.
* **Email** will go to the email address you used when you signed up for W&B. We recommend setting up a filter in your email so all these alerts go into a folder and don't fill up your inbox.

You will only have to do this the first time you set up W&B Alerts, or when you'd like to modify how you receive alerts.

![Alerts settings in W&B User Settings](/images/track/demo_connect_slack.png)

### 2. Add \`wandb.alert()\` to Your Code

Add `wandb.alert()` to your code (either in a Notebook or Python script) wherever you'd like it to be triggered

```python
wandb.alert(title="High Loss", text="Loss is increasing rapidly")
```

#### Check your Slack or email

Check your Slack or emails for the alert message. If you didn't receive any, make sure you've got emails or Slack turned on for **Scriptable Alerts** in your [User Settings](https://wandb.ai/settings)

## Using \`wandb.alert()\`

| Argument                   | Description                                                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `title` (string)           | A short description of the alert, for example "Low accuracy"                                                                                          |
| `text` (string)            | A longer, more detailed description of what happened to trigger the alert                                                                             |
| `level` (optional)         | How important the alert is — must be either `AlertLevel.INFO`, `AlertLevel.WARN`, or `AlertLevel.ERROR.` You can import `AlertLevel.xxx` from `wandb` |
|                            |                                                                                                                                                       |
| `wait_duration` (optional) | How many seconds to wait before sending another alert with the same **title.** This helps reduce alert spam                                           |

### Example

This simple alert sends a warning when accuracy falls below a threshold. In this example, it only sends alerts at least 5 minutes apart.

[Run the code →](http://wandb.me/alerts)

```python
import wandb
from wandb import AlertLevel

if acc < threshold:
    wandb.alert(
        title="Low accuracy",
        text=f"Accuracy {acc} is below the acceptable threshold {threshold}",
        level=AlertLevel.WARN,
        wait_duration=300,
    )
```

## More Info

### Tagging / Mentioning Users

When sending Alerts on Slack you can @ yourself or your colleagues by adding their Slack user id as `<@USER_ID>` in either the title or the text of the Alert. You can find a Slack user id from their Slack profile page.

```python
wandb.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

### W&B Team Alerts

Team admins can set up alerts for the team on the team settings page: wandb.ai/teams/`your-team`. These alerts apply to everyone on your team. We recommend using the **Slackbot** channel because it keeps the alerts private.

### Changing Slack Channels

To change what channel you're posting to, click **Disconnect Slack** and then reconnect, picking a different destination channel.

## FAQ(s)

#### Do "Run Finished" Alerts work in Jupyter notebooks?

Note that **"Run Finished"** alerts (turned on with the **"Run Finished"** setting in User Settings) only work with Python scripts and are disabled in Jupyter Notebook environments to prevent alert notifications on every cell execution. Use `wandb.alert()` in Jupyter Notebook environments instead.

#### How to enable alerts with [W&B Server](../hosting/intro.md)?

<!-- If you are self-hosting using W&B Server you will need to follow [these steps](../../hosting/setup/configuration#slack) before enabling Slack alerts. -->
