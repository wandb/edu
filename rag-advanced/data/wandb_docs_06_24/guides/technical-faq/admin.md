---
displayed_sidebar: default
---

# Admin

### What is the difference between team and organization?

A team is a collaborative workspace for a group of users working on the same projects, while an organization is a higher-level entity that may consist of multiple teams and is often related to billing and account management.

### What is the difference between team and entity? As a user - what does entity mean for me?

A team is a collaborative workspace for a group of users working on the same projects, while an entity refers to either a username or a team name. When you log runs in W&B, you can set the entity to your personal account or a team account `wandb.init(entity="example-team")`.

### What is a team and where can I find more information about it?

If you want to know more about teams, visit the [teams section](../app/features/teams.md).

### When should I log to my personal entity against my team entity?

Personal Entities are no longer available for accounts created after May 21st, 2024. W&B encourages all users, regardless of sign up date, to log new projects to a Team so you have the option to share your results with others.

### Who can create a team? Who can add or delete people from a team? Who can delete projects?

You can check the different roles and permissions [here](../app/features/teams.md#team-roles-and-permissions).

### What type of roles are available and what are the differences between them?

Go to [this](../app/features/teams.md#team-roles-and-permissions) page to see the different roles and permissions available.

### What are service accounts, and how do we add one to our team? 

Check [this](./general.md#what-is-a-service-account-and-why-is-it-useful) page from our docs to know more about service accounts.

### How can I see the bytes stored, bytes tracked and tracked hours of my organization?

* You can check the bytes stored of your organization at `https://<host-url>/usage/<team-name>`.
* You can check the bytes tracked of your organization at `https://<host-url>/usage/<team-name>/tracked`.
* You can check the tracked hours of your organization at `https://<host-url>/usage/<team-name>/computehour`.

### What really good functionalities are hidden and where can I find those?

We have some functionalities hidden under a feature flag in the “Beta Features” section. These can be enabled under the user settings page.

![Available beta features hidden under a feature flag](/images/technical_faq/beta_features.png)

### Which files should I check when my code crashes? 

For the affected run, you should check `debug.log` and `debug-internal.log`. These files are under your local folder `wandb/run-<date>_<time>-<run-id>/logs` in the same directory where you’re running your code.

### On a local instance, which files should I check when I have issues?

You should check the `Debug Bundle`. An admin of the instance can get it from the `/system-admin` page -> top right corner W&B icon -> `Debug Bundle`.

![Access System settings page as an Admin of a local instance](/images/technical_faq/local_system_settings.png)
![Download the Debug Bundle as an Admin of a local instance](/images/technical_faq/debug_bundle.png)

### If I am the admin of my local instance, how should I manage it?

If you are the admin of your instance, go through our [User Management](../hosting/iam/manage-users.md) section to learn how to add users to the instance and create teams.