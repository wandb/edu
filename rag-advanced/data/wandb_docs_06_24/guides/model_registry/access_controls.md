---
description: Use model registry role based access controls (RBAC) to control who can update protected aliases.
displayed_sidebar: default
---

# Data governance and access control

Use *protected aliases* to represent key stages of your model development pipeline. Only *Model Registry Administrators* can add, modify, or remove protected aliases. Model registry admins can define and use protected aliases. W&B blocks non admin users from adding or removing protected aliases from model versions.

:::info
Only Team admins or current registry admins can manage the list of registry admins.
:::

For example, suppose you set `staging` and `production` as protected aliases. Any member of your team can add new model versions. However, only admins can add a `staging` or `production` alias.


## Set up access control
The following steps describe how to set up access controls for your team’s model registry.

1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select the gear button on the top right of the page.
![](/images/models/rbac_gear_button.png)
3. Select the **Manage registry admins** button. 
4. Within the **Members** tab, select the users you want to grant access to add and remove protected aliases from model versions.
![](/images/models/access_controls_admins.gif)


## Add protected aliases
1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select the gear button on the top right of the page.
![](/images/models/rbac_gear_button.png)
3. Scroll down to the **Protected Aliases** section.
4. Click on the plus icon (**+**) icon to add new a new alias.
![](/images/models/access_controls_add_protected_aliases.gif)
