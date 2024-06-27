---
description: Use an Automation for model CI (automated model evaluation pipelines) and model deployment.
title:  Model registry automations 
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';



# Triggering CI/CD events with model registry changes

Create an automation to trigger workflow steps, such as automated model testing and deployment. To create an automation, define the [action](#action-types) you want to occur based on an [event type](#event-types).

For example, you can create a trigger that automatically deploys a model to GitHub when you add a new version of a registered model.

:::info
Looking for companion tutorials for automations? 
1. [This](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) tutorial shows you how to set up an automation that triggers a Github Action for model evaluation and deployment
2. [This](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared) video series shows webhook basics and how to set them up in W&B.
3. [This](https://www.youtube.com/watch?v=s5CMj_w3DaQ) demo details how to setup an automation to deploy a model to a Sagemaker Endpoint
:::

## Event types
An *event* is a change that takes place in the W&B ecosystem. The Model Registry supports two event types: **Linking a new artifact to a registered model** and **Adding a new alias to a version of the registered model**.

See [Link a model version](./link-model-version.md) for information on how to link model versions and [Create a custom alias](../artifacts/create-a-custom-alias.md) for information an artifact aliases.



:::tip
Use the **Linking a new artifact to a registered model** event type to test new model candidates. Use the **Adding a new alias to a version of the registered model** event type to specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied.
:::


## Action types
An action is a responsive mutation (internal or external) that occurs as a result of some trigger. There are two types of actions you can create in the Model Registry: [webhooks](#create-a-webhook-automation) and [W&B Launch Jobs](../launch/intro.md).

* Webhooks: Communicate with an external web server from W&B with HTTP requests.
* W&B Launch job: [Jobs](../launch/create-launch-job.md) are reusable, configurable run templates that allow you to quickly launch new [runs](../runs/intro.md) locally on your desktop or external compute resources such as Kubernetes on EKS, Amazon SageMaker, and more. 


The following sections describe how to create an automation with webhooks and W&B Launch.

## Create a webhook automation 
Automate a webhook based on an action with the W&B App UI. To do this, first establish a webhook, then configure the webhook automation. 

:::info
Specify an endpoint for your webhook that has an Address record (A record). W&B does not support connecting to endpoints that are exposed directly with IP addresses such as `[0-255].[0-255].[0-255].[0.255]` or endpoints exposed as `localhost`. This restriction helps protect against server-side request forgery (SSRF) attacks and other related threat vectors.
:::

### Add a secret for authentication or authorization
Secrets are team-level variables that let you obfuscate private strings such as credentials, API keys, passwords, tokens, and more. W&B recommends you use secrets to store any string that you want to protect the plain text content of.

To use a secret in your webhook, you must first add that secret to your team's secret manager.

:::info
* Only W&B Admins can create, edit, or delete a secret.
* Skip this section if the external server you send HTTP POST requests to does not use secrets.  
* Secrets are also available if you use [W&B Server](../hosting/intro.md) in an Azure, GCP, or AWS deployment. Connect with your W&B account team to discuss how you can use secrets in W&B if you use a different deployment type.
:::

There are two types of secrets W&B suggests that you create when you use a webhook automation:

* **Access tokens**: Authorize senders to help secure webhook requests 
* **Secret**: Ensure the authenticity and integrity of data transmitted from payloads

Follow the instructions below to create a webhook:

1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
3. Scroll down the page until you find the **Team secrets** section.
4. Click on the **New secret** button.
5. A modal will appear. Provide a name for your secret in the **Secret name** field.
6. Add your secret into the **Secret** field. 
7. (Optional) Repeat steps 5 and 6 to create another secret (such as an access token) if your webhook requires additional secret keys or tokens to authenticate your webhook.

Specify the secrets you want to use for your webhook automation when you configure the webhook. See the [Configure a webhook](#configure-a-webhook) section for more information. 

:::tip
Once you create a secret, you can access that secret in your W&B workflows with `$`.
:::

:::caution
Considerations if you use secrets in W&B Server:

You are responsible for configuring security measures that satisfy your security needs. 

W&B strongly recommends that you store secrets in a W&B instance of a cloud secrets manager provided by AWS, GCP, or Azure. Secret managers provided by AWS, GCP, and Azure are configured with advanced security capabilities.  

W&B does not recommend that you use a Kubernetes cluster as the backend of your secrets store. Consider a Kubernetes cluster only if you are not able to use a W&B instance of a cloud secrets manager (AWS, GCP, or Azure), and you understand how to prevent security vulnerabilities that can occur if you use a cluster.
:::

### Configure a webhook
Before you can use a webhook, first configure that webhook in the W&B App UI. 

:::info
* Only W&B Admins can configure a webhook for a W&B Team.
* Ensure you already [created one or more secrets](#add-a-secret-for-authentication-or-authorization) if your webhook requires additional secret keys or tokens to authenticate your webhook.
:::

1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
4. Scroll down the page until you find the **Webhooks** section.
5. Click on the **New webhook** button.  
6. Provide a name for your webhook in the **Name** field.
7. Provide the endpoint URL for the webhook in the **URL** field.
8. (Optional) From the **Secret** dropdown menu, select the secret you want to use to authenticate the webhook payload.
9. (Optional) From the **Access token** dropdown menu, select the access token you want to use to authorize the sender.
9. (Optional) From the **Access token** dropdown menu select additional secret keys or tokens required to authenticate a webhook  (such as an access token).

:::note
See the [Troubleshoot your webhook](#troubleshoot-your-webhook) section to view where the secret and access token are specified in
the POST request.
:::


### Add a webhook 
Once you have a webhook configured and (optionally) a secret, navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).

1. From the **Event type** dropdown, select an [event type](#event-types).
![](/images/models/webhook_select_event.png)
2. (Optional) If you selected **A new version is added to a registered model** event, provide the name of a registered model from the **Registered model** dropdown. 
![](/images/models/webhook_new_version_reg_model.png)
3. Select **Webhooks** from the **Action type** dropdown. 
4. Click on the **Next step** button.
5. Select a webhook from the **Webhook** dropdown.
![](/images/models/webhooks_select_from_dropdown.png)
6. (Optional) Provide a payload in the JSON expression editor. See the [Example payload](#example-payloads) section for common use case examples.
7. Click on **Next step**.
8. Provide a name for your webhook automation in the **Automation name** field. 
![](/images/models/webhook_name_automation.png)
9. (Optional) Provide a description for your webhook. 
10. Click on the **Create automation** button.



### Example payloads

The following tabs demonstrate example payloads based on common use cases. Within the examples they reference the following keys to refer to condition objects in the payload parameters:
* `${event_type}` Refers to the type of event that triggered the action.
* `${event_author}` Refers to the user that triggered the action.
* `${artifact_version}` Refers to the specific artifact version that triggered the action. Passed as an artifact instance.
* `${artifact_version_string}` Refers to the specific artifact version that triggered the action. Passed as a string.
* `${artifact_collection_name}` Refers to the name of the artifact collection that the artifact version is linked to.
* `${project_name}` Refers to the name of the project owning the mutation that triggered the action.
* `${entity_name}` Refers to the name of the entity owning the mutation that triggered the action.


<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
Verify that your access tokens have required set of permissions to trigger your GHA workflow. For more information, [see these GitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event). 
:::
  
  Send a repository dispatch from W&B to trigger a GitHub action. For example, suppose you have workflow that accepts a repository dispatch as a trigger for the `on` key:

  ```yaml
  on:
    repository_dispatch:
      types: BUILD_AND_DEPLOY
  ```

  The payload for the repository might look something like:

  ```json
  {
    "event_type": "BUILD_AND_DEPLOY",
    "client_payload": 
    {
      "event_author": "${event_author}",
      "artifact_version": "${artifact_version}",
      "artifact_version_string": "${artifact_version_string}",
      "artifact_collection_name": "${artifact_collection_name}",
      "project_name": "${project_name}",
      "entity_name": "${entity_name}"
      }
  }

  ```
:::note
The `event_type` key in the webhook payload must match the `types` field in the GitHub workflow YAML file.
:::

  The contents and positioning of rendered template strings depends on the event or model version the automation is configured for. `${event_type}` will render as either "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS". See below for an example mapping:

  ```json
  ${event_type} --> "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
  ${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
  ${artifact_collection_name} --> "<registered_model_name>"
  ${project_name} --> "model-registry"
  ${entity_name} --> "<entity>"
  ```

  Use template strings to dynamically pass context from W&B to GitHub Actions and other tools. If those tools can call Python scripts, they can consume the registered model artifacts through the [W&B API](../artifacts/download-and-use-an-artifact.md).

  For more information about repository dispatch, see the [official documentation on the GitHub Marketplace](https://github.com/marketplace/actions/repository-dispatch).  

  See [Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) and [Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) for step by step YouTube videos on how to create automations for model evaluation and deployment, respectively. 

  See this W&B [report](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) to learn how to use a Github Actions webhook automation for Model CI. Check out this [GitHub repository](https://github.com/hamelsmu/wandb-modal-webhook) to learn how to create model CI with a Modal Labs webhook. 


  </TabItem>
  <TabItem value="microsoft">

  Configure an ‘Incoming Webhook' to get the webhook URL for your Teams Channel by configuring. The following is an example payload:
  
  ```json 
  {
  "@type": "MessageCard",
  "@context": "http://schema.org/extensions",
  "summary": "New Notification",
  "sections": [
    {
      "activityTitle": "Notification from WANDB",
      "text": "This is an example message sent via Teams webhook.",
      "facts": [
        {
          "name": "Author",
          "value": "${event_author}"
        },
        {
          "name": "Event Type",
          "value": "${event_type}"
        }
      ],
      "markdown": true
    }
  ]
  }
  ```
  You can use template strings to inject W&B data into your payload at the time of execution (as shown in the Teams example above).


  </TabItem>
  <TabItem value="slack">

  Setup your Slack app and add an incoming webhook integration with the instructions highlighted in the [Slack API documentation](https://api.slack.com/messaging/webhooks). Ensure that you have the secret specified under `Bot User OAuth Toke`n as your W&B webhook’s access token. 
  
  The following is an example payload:

  ```json
    {
        "text": "New alert from WANDB!",
    "blocks": [
        {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Registry event: ${event_type}"
            }
        },
            {
                "type":"section",
                "text": {
                "type": "mrkdwn",
                "text": "New version: ${artifact_version_string}"
            }
            },
            {
            "type": "divider"
        },
            {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Author: ${event_author}"
            }
            }
        ]
    }
  ```

  </TabItem>
</Tabs>

### Troubleshoot your webhook

Interactively troubleshoot your webhook with the W&B App UI or programmatically with a Bash script. You can troubleshoot a webhook when you create a new webhook or edit an existing webhook.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

Interactively test a webhook with the W&B App UI. 

1. Navigate to your W&B Team Settings page.
2. Scroll to the **Webhooks** section.
3. Click on the horizontal three docs (meatball icon) next to the name of your webhook.
4. Select **Test**.
5. From the UI panel that appears, paste your POST request to the field that appears. 
![](/images/models/webhook_ui.png)
6. Click on **Test webhook**.

Within the W&B App UI, W&B posts the response made by your endpoint.

![](/images/models/webhook_ui_testing.gif)

See [Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) YouTube video to view a real-world example.

  </TabItem>
  <TabItem value="bash">

The following bash script generates a POST request similar to the POST request W&B sends to your webhook automation when it is triggered.

Copy and paste the code below into a shell script to troubleshoot your webhook. Specify your own values for the following:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`


```sh title="webhook_test.sh"
#!/bin/bash

# Your access token and secret
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# The data you want to send (for example, in JSON format)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# Generate the HMAC signature
# For security, Wandb includes the X-Wandb-Signature in the header computed 
# from the payload and the shared secret key associated with the webhook 
# using the HMAC with SHA-256 algorithm.
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

# Make the cURL request
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT
```

  </TabItem>
</Tabs>



## Create a launch automation
Automatically start a W&B Job. 

:::info
This section assumes you already have created a job, a queue, and have an active agent polling. For more information, see the [W&B Launch docs](../launch/intro.md). 
:::


1. From the **Event type** dropdown, select an event type. See the [Event type](#event-types) section for information on supported events.
2. (Optional) If you selected **A new version is added to a registered model** event, provide the name of a registered model from the **Registered model** dropdown. 
3. Select **Jobs** from the **Action type** dropdown. 
4. Select a W&B Launch job from the **Job** dropdown.  
5. Select a version from the **Job version** dropdown.
6. (Optional) Provide hyperparameter overrides for the new job.
7. Select a project from the **Destination project** dropdown.
8. Select a queue to enqueue your job to.  
9. Click on **Next step**.
10. Provide a name for your webhook automation in the **Automation name** field. 
11. (Optional) Provide a description for your webhook. 
12. Click on the **Create automation** button.

See this example [report](https://wandb.ai/examples/wandb_automations/reports/Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx) for an end to end example on how to create an automation for model CI with W&B Launch.
## View automation

View automations associated to a registered model from the W&B App UI. 

1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select on a registered model. 
3. Scroll to the bottom of the page to the **Automations** section.

Within the Automations section you can find the following properties of automations created for the model you selected:

- **Trigger type**: The type of trigger that was configured.
- **Action type**: The action type that triggers the automation. Available options are Webhooks and Launch.
- **Action name**: The action name you provided when you created the automation.
- **Queue**: The name of the queue the job was enqueued to. This field is left empty if you selected a webhook action type.

## Delete an automation
Delete an automation associated with a model. Actions in progress are not affected if you delete that automation before the action completes. 

1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Click on a registered model. 
3. Scroll to the bottom of the page to the **Automations** section.
4. Hover your mouse next to the name of the automation and click on the kebob (three vertical dots) menu. 
5. Select **Delete**.


