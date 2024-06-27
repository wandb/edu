---
displayed_sidebar: default
---

# Slack alerts

Integrate W&B Server with [Slack](https://slack.com/).

## Create the Slack application

Follow the procedure below to create a Slack application.

1. Visit https://api.slack.com/apps and select **Create an App**.

![](/images/hosting/create_an_app.png)

2. Provide a name for your app in the **App Name** field.
3. Select a Slack workspace where you want to develop your app in. Ensure that the Slack workspace you use is the same workspace you intend to use for alerts.

![](/images/hosting/name_app_workspace.png)

## Configure the Slack application

1. On the left sidebar, select **OAth & Permissions**.

![](/images/hosting/add_an_oath.png)

2. Within the Scopes section, provide the bot with the **incoming_webhook** scope. Scopes give your app permission to perform actions in your development workspace.

   For more information about OAuth scopes for Bots, see the Understanding OAuth scopes for Bots tutorial in the Slack api documentation.

![](/images/hosting/save_urls.png)

3. Configure the Redirect URL to point to your W&B installation. Use the same URL that your host URL is set to in your local system settings. You can specify multiple URLs if you have different DNS mappings to your instance.

![](/images/hosting/redirect_urls.png)

4. Select **Save URLs**.
5. You can optionally specify an IP range under **Restrict API Token Usage**, allow-list the IP or IP range of your W&B instance(s). Limiting the allowed IP address helps further secure your Slack application.

## Register your Slack application with W&B

1. Navigate to the **System Settings** page of your W&B instance. Toggle the **Enable a custom Slack application to dispatch alerts** to enable a custom Slack application:

![](/images/hosting/register_slack_app.png)

You will need to supply your Slack application's client ID and secret. Navigate to Basic Information in Settings to find your application’s client ID and secret.

2. Verify that everything is working by setting up a Slack integration in the W&B app.
