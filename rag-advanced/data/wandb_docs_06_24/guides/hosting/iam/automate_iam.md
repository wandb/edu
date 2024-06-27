---
displayed_sidebar: default
---

# Automate user and team management

## SCIM API

Use SCIM API to manage users, and the teams they belong to, in an efficient and repeatable manner. You can also use the SCIM API to manage custom roles or assign roles to users in your W&B organization. Role endpoints are not part of the official SCIM schema. W&B adds role endpoints to support automated management of custom roles.

SCIM API is especially useful if you want to:

* manage user provisioning and de-provisioning at scale
* manage users with a SCIM-supporting Identity Provider

There are broadly three categories of SCIM API - **User**, **Group**, and **Roles**.

### User SCIM API

[User SCIM API](./scim.md#user-resource) allows for creating, deactivating, getting the details of a user, or listing all users in a W&B organization. This API also supports assigning predefined or custom roles to users in an organization.

:::info
Deactivate a user within a W&B organization with the `DELETE User` endpoint. Deactivated users can no longer sign in. However, deactivated users still appears in the organization's user list.

To fully remove a deactivated user from the user list, you must [remove the user from the organization](#remove-a-user).

It is possible to re-enable a deactivated user, if needed.
:::

### Group SCIM API

[Group SCIM API](./scim.md#group-resource) allows for managing W&B teams, including creating or removing teams in an organization. Use the `PATCH Group` to add or remove users in an existing team.

:::info
There is no notion of a `group of users having the same role` within W&B. A W&B team closely resembles a group, and allows diverse personas with different roles to work collaboratively on a set of related projects. Teams can consist of different groups of users. Assign each user in a team a role: team admin, member, viewer, or a custom role.

W&B maps Group SCIM API endpoints to W&B teams because of the similarity between groups and W&B teams.
:::

### Custom role API

[Custom role SCIM API](./scim.md#role-resource) allows for managing custom roles, including creating, listing, or updating custom roles in an organization.

:::caution
Delete a custom role with caution.

Delete a custom role within a W&B organization with the `DELETE Role` endpoint. The predefined role that the custom role inherits is assigned to all users that are assigned the custom role before the operation.

Update the inherited role for a custom role with the `PUT Role` endpoint. This operation doesn't affect any of the existing, that is, non-inherited custom permissions in the custom role.
:::

## W&B Python SDK API

Just like how SCIM API allows you to automate user and team management, you can also use some of the methods available in the [W&B Python SDK API](../../../ref/python/public-api/api.md) for that purpose. Keep a note of the following methods:

| Method name | Purpose |
|-------------|---------|
| `create_user(email, admin=False)` | Add a user to the organization and optionally make them the organization admin. |
| `user(userNameOrEmail)` | Return an existing user in the organization. |
| `user.teams()` | Return the teams for the user. You can get the user object using the user(userNameOrEmail) method. |
| `create_team(teamName, adminUserName)` | Create a new team and optionally make an organization-level user the team admin. |
| `team(teamName)` | Return an existing team in the organization. |
| `Team.invite(userNameOrEmail, admin=False)` | Add a user to the team. You can get the team object using the team(teamName) method. |
| `Team.create_service_account(description)` | Add a service account to the team. You can get the team object using the team(teamName) method. |
|` Member.delete()` | Remove a member user from a team. You can get the list of member objects in a team using the team object's `members` attribute. And you can get the team object using the team(teamName) method. |