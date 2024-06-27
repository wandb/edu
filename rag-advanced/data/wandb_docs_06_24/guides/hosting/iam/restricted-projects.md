---
description: Restricted Projects for collaborating on AI workflows with sensitive data
displayed_sidebar: default
---

# Project visibility

Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it. Only the owner of the project or a team admin can set or edit a project's visibility.

## Visibility scopes
There are four project visibility scopes you can choose from. In order of most public to most private, they are: 
* _Open_: Anyone can submit runs or reports.
* _Public_: Anyone can view this project. Only your team can edit.
* _Team_: Only your team can view and edit this project.
* _Restricted_: Only invited members can view this project. Public sharing is disabled.

:::tip
Set a project's scope to **Restricted** if you want to collaborate on workflows related to sensitive or confidential data. When you create a restricted project within a team, you can invite or add specific members from the team to collaborate on relevant experiments, artifacts, reports, and so forth. 

Unlike other project scopes, all members of a team do not get implicit access to a restricted project. At the same time, team admins can join restricted projects to monitor team activity.
:::

## Set visibility scope on a new or existing project

Set a project's visibility scope when you create a project or when editing it later.

:::info
* Only the owner of the project or a team admin can set or edit its visibility scope.
* When a team admin enables **Make all future team projects private (public sharing not allowed)** within a team's privacy setting, that disables **Open** and **Public** project visibility scopes for that team. In this case, your team can only use **Team** and **Restricted** scopes.
:::

### Set visibility scope when you create a new project

1. Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home).
2. Click the **New Project** button in the upper right hand corner.
3. From the **Project Visibility** dropdown, select the desired scope.
![](/images/hosting/restricted_project_add_new.gif)

Complete the following step if you select **Restricted** visibility. 

4. Provide names of one or more W&B team members in the **Invite team members** field. Add only those members who are essential to collaborate on the project.
![](/images/hosting/restricted_project_2.png)

### Edit visibility scope of an existing project

1. Navigate to your W&B Project.
2. Select the **Overview** tab on the left column.
3. Click the **Edit Project Details** button on the upper right corner.  
4. From the **Project Visibility** dropdown, select the desired scope.
![](/images/hosting/restricted_project_edit.gif)

Complete the following step if you select **Restricted** visibility. 

5. Provide one or more names of W&B Team members in the **Invite team members** field.

:::caution
* All members of a team lose access to a project if you change its visibility scope from **Team** to **Restricted**, unless you invite the required team members to the project.
* All members of a team get access to a project if you change its visibility scope from **Restricted** to **Team**.
* If you remove a team member when editing a restricted project, they lose access to that project.
:::

## Other key things to note for restricted scope

* If you want to use a team-level service account in a restricted project, you should invite or add that specifically to the project. Otherwise a team-level service account can not access a restricted project by default.
* You can not move runs from a restricted project, but you can move runs from a non-restricted project to a restricted one.
* You can convert the visibility of a restricted project to only **Team** scope, irrespective of the team privacy setting **Make all future team projects private (public sharing not allowed)**.
* If the owner of a restricted project is not part of the parent team anymore, the team admin should change the owner to ensure seamless operations in the project.
