---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Enqueue jobs

The following page describes how to add launch jobs to a launch queue.

:::info
Ensure that you, or someone on your team, has already configured a launch queue. For more information, see the [Set up Launch](./setup-launch.md) page.
:::



## Add jobs to your queue

Add jobs to your queue interactively with the W&B App or programmatically with the W&B CLI.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'W&B CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
Add a job to your queue programmatically with the W&B App.

1. Navigate to your W&B Project Page.
2. Select the **Jobs** icon on the left panel:
  ![](/images/launch/project_jobs_tab_gs.png)
3. The **Jobs** page displays a list of W&B launch jobs that were created from previously executed W&B runs. 
  ![](/images/launch/view_jobs.png)
4. Select the **Launch** button next to the name of the Job name. A modal will appear on the right side of the page.
5. From the **Job version** dropdown, select the version of hte launch job you want to use. Launch jobs are versioned like any other [W&B Artifact](../artifacts/create-a-new-artifact-version.md). Different versions of the same launch job will be created if you make modifications to the software dependencies or source code used to run the job.
6. Within the **Overrides** section, provide new values for any inputs that are configured for your launch job. Common overrides include a new entrypoint command, arguments, or values in the `wandb.config` of your new W&B run.  
  ![](/images/launch/create_starter_queue_gs.png)
  You can copy and paste values from other W&B runs that used your launch job by clicking on the **Paste from...** button.
7. From the **Queue** dropdown, select the name of the launch queue you want to add your launch job to. 
8. Use the **Job Priority** dropdown to specify the priority of your launch job.  A launch job's priority is set to "Medium" if the launch queue does not support prioritization.
9. **(Optional) Follow this step only if a queue config template was created by your team admin**  
Within the **Queue Configurations** field, provide values for configuration options that were created by the admin of your team.  
For example, in the following example, the team admin configured AWS instance types that can be used by the team. In this case, team members can pick either the `ml.m4.xlarge` or `ml.p3.xlarge` compute instance type to train their model.
![](/images/launch/team_member_use_config_template.png)
10. Select the **Destination project**, where the resulting run will appear.  This project needs to belong to the same entity as the queue.
11. Select the **Launch now** button. 


  </TabItem>
    <TabItem value="cli">

Use the `wandb launch` command to add jobs to a queue. Create a JSON configuration with hyperparameter overrides. For example, using the script from the [Quickstart](./walkthrough.md) guide, we create a JSON file with the following overrides:

```json title="config.json"
{
  "overrides": {
      "args": [],
      "run_config": {
          "learning_rate": 0,
          "epochs": 0
      },   
      "entry_point": []
  }
}
```

:::note
W&B Launch will use the default parameters if you do not provide a JSON configuration file.
:::

If you want to override the queue configuration, or if your launch queue does not have a configuration resource defined, you can specify the `resource_args` key in your config.json file. For example, following continuing the example above, your config.json file might look similar to the following:

```json title="config.json"
{
  "overrides": {
      "args": [],
      "run_config": {
          "learning_rate": 0,
          "epochs": 0
      },
      "entry_point": []
  },
  "resource_args": {
        "<resource-type>" : {
            "<key>": "<value>"
        }
  }
}
```

Replace values within the `<>` with your own values.



Provide the name of the queue for the `queue`(`-q`) flag, the name of the job for the `job`(`-j`) flag, and the path to the configuration file for the `config`(`-c`) flag.

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
If you work within a W&B Team, we suggest you specify the `entity` flag (`-e`) to indicate which entity the queue will use.

  </TabItem>
</Tabs>


