---
displayed_sidebar: default
---

# Configure registry access
<!-- A registry, and the linked artifacts inside a registry, belong to an organization. This means that teams within an organization can publish and consume artifacts linked to a registry, if that team has correct access control. -->

Registry admins can limit who can access a registry by navigating to a registry's settings and assigning a user's role to [Admin, Member, or Viewer](#registry-roles-permissions). Users can have different roles in different registries. For example, a user can have a view role in "Registry A" and a member role in the "Registry B". 

:::note
Only registry admins can [restrict visibility](#restrict-visibility-to-a-registry), [configure user roles](#configure-user-roles-in-a-registry), or [remove users](#remove-a-user-from-a-registry) from registries in an organization.
:::

## Registry roles permissions

A user within an organization can have different roles, and therefore permissions, for each registry in their organization.

:::info W&B role types
W&B has three different types of roles: Organization roles, [team roles](../app/features/teams.md#team-roles-and-permissions), and [registry roles](#registry-roles-permissions).

Your role in a team has no impact or relationship on your role in any registry.
:::

The proceeding table lists the different roles a user can have and their permissions:


| Permission                                                     | Permission Group | Viewer | Member | Admin | Owner |
|--------------------------------------------------------------- |------------------|--------|--------|-------|-------|
| View a collection’s details                                    | Read             |   X    |   X    |   X   |   X   |
| View a linked artifact’s details                               | Read             |   X    |   X    |   X   |   X   |
| Usage: Consume an artifact in a registry with use_artifact     | Read             |   X    |   X    |   X   |   X   |
| Download a linked artifact                                     | Read             |   X    |   X    |   X   |   X   |
| Download files from an artifact’s file viewer                  | Read             |   X    |   X    |   X   |   X   |
| Search a registry                                              | Read             |   X    |   X    |   X   |   X   |
| View a registry’s settings and user list                       | Read             |   X    |   X    |   X   |   X   |
| Create a new automation for a collection                       | Create           |        |   X    |   X   |   X   |
| Turn on Slack notifications for new version being added        | Create           |        |   X    |   X   |   X   |
| Create a new collection                                        | Create           |        |   X    |   X   |   X   |
| Create a new custom registry                                   | Create           |        |   X    |   X   |   X   |
| Edit collection card (description)                             | Update           |        |   X    |   X   |   X   |
| Edit linked artifact description                               | Update           |        |   X    |   X   |   X   |
| Add or delete a collection’s tag                               | Update           |        |   X    |   X   |   X   |
| Add or delete an alias from a linked artifact                  | Update           |        |   X    |   X   |   X   |
| Link a new artifact                                            | Update           |        |   X    |   X   |   X   |
| Edit allowed types list for a registry                         | Update           |        |   X    |   X   |   X   |
| Edit custom registry name                                      | Update           |        |   X    |   X   |   X   |
| Delete a collection                                            | Delete           |        |   X    |   X   |   X   |
| Delete an automation                                           | Delete           |        |   X    |   X   |   X   |
| Unlink an artifact from a registry                             | Delete           |        |   X    |   X   |   X   |
| Edit accepted artifact types for a registry                    | Admin            |        |        |   X   |   X   |
| Change registry visibility (Organization or Restricted)        | Admin            |        |        |   X   |   X   |
| Add users to a registry                                        | Admin            |        |        |   X   |   X   |
| Assign or change a user's role in a registry                   | Admin            |        |        |   X   |   X   |



## Configure user roles in a registry
1. Navigate to the **Registry** App in the W&B App UI.
2. Select the registry you want to configure.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Within the **Member** field, search for the user you want to edit permissions for.
6. Click on the user's role within the **Registry role** column. 
7. From the dropdown, select the role you want to assign to the user.

![](/images/registry/configure_role_registry.gif)

## Remove a user from a registry
1. Navigate to the **Registry** App in the W&B App UI.
2. Select a core or custom registry.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section and type in the username of the member you want to remove.
5. Click the **Delete** button.



## Registry visibility types

There are two registry visibility types: restricted or organization visibility. The following table describes who has access to the registry by default:

| Visibility | Description | Default role | Example |
| --- | --- | --- | --- |
| Organization | Everyone in the org can access the registry. | By default, organization administrators are an admin for the registry. All other users are a viewer in the registry by default. | Core registry |
| Restricted   | Only invited org members can access the registry.| The user who created the restricted registry is the only user in the registry by default, and is the organization's owner. | Custom registry or core registry |


## Restrict visibility to a registry
<!-- Who can do this? -->
Restrict who can view and access a custom registry. You can restrict visibility to a registry when you create a custom registry or after you create a custom registry. A custom registry can have either restricted or organization visibility. For more information on registry visibilities, see [Registry visibility types](./configure_registry.md#registry-visibility-types).

<!-- | Visibility | Description |
| --- | --- |
| Organization | Anyone in the organization can view the registry. |
| Restricted   | Only invited organization members can view and edit the registry.|  -->

The following steps describe how to restrict the visibility of a custom registry that already exists:

1. Navigate to the **Registry** App in the W&B App UI.
2. Select a registry.
3. Click on the gear icon on the upper right hand corner.
4. From the **Registry visibility** dropdown, select the desired registry visibility.

Continue if you select **Restricted visibility**:

5. Add members of your organization that you want to have access to this registry. Scroll to the **Registry members and roles** section and click on the **Add member** button. 
6. Within the **Member** field, add the email or username of the member you want to add.
7. Click **Add new member**.

![](/images/registry/change_registry_visibility.gif)





