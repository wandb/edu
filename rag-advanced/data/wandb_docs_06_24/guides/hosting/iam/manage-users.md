---
displayed_sidebar: default
---
# Manage users
Manage W&B users in your organization or team.

W&B strongly recommends and encourages that users authenticate to an organization using Single Sign-On (SSO). To learn more about how to setup SSO with W&B Server, refer to [SSO with OIDC](./sso.md) or [SSO with LDAP](./ldap.md).

:::note
`W&B Server` refers to both **Dedicated Cloud** or **Self-managed** hosting options.
:::

:::note
`Instance` or `organization` terms are used interchangeably within the context of W&B Server.

W&B is actively developing support for multiple organizations in an enterprise instance of W&B Server. If you're interested in utilizing that capability, reach out to your W&B team.
:::

## Instance Admins
The first user to sign up after the W&B Server instance is initially deployed, is automatically assigned the instance `admin` role. The admin can then add additional users to the organization and create teams.

:::note
W&B recommends to have more than one instance admin in an organization. It is a best practice to ensure that admin operations can continue when the primary admin is not available. 
:::

## Manage your organization
As an instance admin, you can invite, remove, and change a user's role. To do so, navigate to the Organization dashboard and follow the instructions described below.

1. Select your profile image in the upper right hand corner.
2. A dropdown will appear, click on **Organization dashboard**.

![](/images/hosting/how_get_to_dashboard.png)

If you are looking to simplify user management in your organization, refer to [Automate user and team management](./automate_iam.md).

### Invite users
1. Navigate to the W&B Organization dashboard.
2. Click the **Add user** button.
3. Add the user's email in the Email field.
4. Select the role you want to assign to the user, from `Admin, Member or Viewer`. By default, all users are assigned a `Member` role.
    - **Admin**: A instance admin who can add or remove other users to the organization, change user roles, manage custom roles, add teams and more. W&B recommends more than one admin for an enterprise W&B server instance.
    - **Member** - A regular user of the organization, invited by an instance admin. A organization user cannot invite other users or manage existing users in the organization. `Team admins` could add specific organization users to their respective teams (team-level roles described below in **Team roles**).
    - **Viewer** - A view-only user of your organization, invited by an instance admin. A viewer only has read access to the organization and the underlying teams that they are a part of.
5. Click the **Add new user** button.

![](/images/hosting/org_dashboard_add_user.png)

An invite link will be sent to the user by email. Once the user accepts the invite, they will have access to the W&B instance (organization).

:::info
The **Add user** option might be not be available if there are no more seats in the license. Reach out to your W&B team if you have difficulty adding users. 
:::

:::note
W&B uses a third-party email server to send the user invites. If you've a self-managed W&B Server instance and your organization firewall rules restrict sending traffic outside the corporate network, W&B provides an option to configure an internal SMTP server in the instance. Please refer to [these instructions](../smtp.md) to setup the SMTP server.
:::

### User auto-provisioning
If Single Sign-On (SSO) is setup for your enterprise W&B Server instance, any user in your company who has access to the instance URL can sign-in to the organization, provided the settings in your SSO provider allow so. When a user signs in for the first time using SSO, their W&B organization user will be automatically created without needing an instance admin to generate a user invite. This is a good alternative for adding users to your W&B organization at scale.

User auto-provisioning with SSO on by default for W&B Server. It is possible to turn it `off` if you would like to selectively add specific users to your W&B organization. If you're on **Dedicated Cloud**, reach out to your W&B team. If you've a **Self-managed** deployment, you can configure the setting `DISABLE_SSO_PROVISIONING=true` for your W&B Server instance.

:::note
If auto-provisioning is on for your W&B Server instance, there may be a way to control which specific users can sign-in to the organization with your SSO provider to restrict the product use to relevant personnel. Extent of that configurability will depend on your SSO provider and is outside the scope of W&B documentation.
:::

### Remove a user
1. Navigate to the W&B Organization dashboard.
2. Search for the user you want to modify in the search bar.
3. Click on the meatball menu (three horizontal dots).
4. Select **Remove user**.

![](/images/hosting/remove_user_from_org.png)

### Change a user's organization-level role
1. Navigate to the W&B Organization dashboard.
2. Search for the user you want to modify in the search bar.
3. Hover your mouse to the **Role** column. Click on the pencil icon that appears.
4. From the dropdown, select a different role you want to assign.

## Manage a team
Use a team home page as a central hub to explore projects, reports, and runs. Within the team home page there is a **Settings** tab. Use the Settings tab to manage users, set a team avatar, adjust privacy settings, set up alerts, track usage, and more. For more information, see the [Team settings](../../app/settings-page/team-settings.md) page.

:::tip
Team admins can add and remove users in their teams. Add a users to team with the user's email or use the user's organization-level username. A non-admin user in a team cannot invite other users to that team, **unless** team admin has enabled the relevant team setting.

See **Team roles** below for what roles are available at the team-level.
:::

If you're looking to simplify team management in your organization, refer to [Automate user and team management](./automate_iam.md).

### Create a team
1. Navigate to the W&B Organization dashboard.
2. Select the **Create new team** button on the left navigation panel.
![](/images/hosting/create_new_team.png)
3. A modal will appear. Prove a name for your team in the **Team name** field. 
4. Select a storage type. 
5. Click on the **Create team** button.

This will redirect you to a newly created Team home page. 

### Team roles
When you (team admin) invite a user to a team you can assign them one of the following roles:

| Role   |   Definition   |
|-----------|---------------------------|
| Admin     | A user who can add and remove other users in the team, change user roles, and configure team settings.   |
| Member    | A regular user of a team, invited by email or their organization-level username by the team admin. A member user cannot invite other users to the team.  |
| View-Only (Enterprise-only feature) | A view-only user of a team, invited by email or their organization-level username by the team admin. A view-only user only has read access to the team and its contents.  |
| Service (Enterprise-only feature)   | A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure to set the environment variable `WANDB_USERNAME`  to correctly attribute runs to the appropriate user. |
| Custom Roles (Enterprise-only feature)   | Custom roles allow organization admins to compose new roles by inheriting from the above View-Only or Member roles, and adding additional permissions to achieve fine-grained access control. Team admins can then assign any of those custom roles to users in their respective teams. Refer to [this article](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) for details. |

:::note
W&B recommends to have more than one admin in a team. It is a best practice to ensure that admin operations can continue when the primary admin is not available.
:::

:::note
Refer to [Team Service Account Behavior](../../app/features/teams.md#team-service-account-behavior) for more information.
:::

:::note
If you're on W&B Server (Dedicated Cloud or Self-managed deployment), you will need an updated enterprise license to use the **Custom Roles** feature.
:::

### Invite users to a team
Use the `Members` tab in the Team's settings page to invite users to your team.

:::info
Members of a team inherit the organization that the team is a part of.
:::

1. Navigate to the Team's Settings page.
2. Select the **Members** tab.
3. Enter an email or W&B username in the search bar.
4. Once you have found the user, click the **Invite** button.

### Remove users from a team
Use the `Members` tab in the Team's settings page to remove users from your team.

1. Navigate to the Team's settings page.
2. Select the Delete button next the to user's name.

:::info
W&B keeps runs logged by team members, even if they are no longer on the team.
:::



