---
description: >-
  Manage a team's members, avatar, alerts, and privacy settings with the Team
  Settings page.
displayed_sidebar: default
---

# Team settings

Navigate to your team’s profile page and select the **Team settings** icon to manage team settings. Not all members in a team can modify team settings. Only team admins can view a team's settings and access team level TTL settings. The account type (Administrator, Member, or Service) of a member determines what settings that member can modify. 

:::info
Only Administration account types can change team settings or remove a member from a team.
:::

## Members

The **Members** section shows a list of all pending invitations and the members that have either accepted the invitation to join the team. Each member listed displays a member’s name, username, and account type. There are three account types: Administrator (Admin), Member, and Service.

### Change a member's role in the team

Complete the proceeding steps to change a member's role in a team:

1. Select the account type icon next to the name of a given team member. A modal will appear.
2. Select the drop-down menu.
3. From the drop-down, choose the account type you want that team member to posses.

### Remove a member from a team

Select the trash can icon next to the name of the member you want to remove from the team.

:::info
Runs created in a team account are preserved when the member who created those runs are removed from the team.
:::

### Match members to a team organization during signup

Allow new users within your organization discover Teams within your organization when they sign-up. New users must have a verified email domain that matches your organization's verified email domain. Verified new users will see a list of verified teams that belong to an organization when they sign up for a W&B account.

An organization administrator (Admin) must enable this feature. To enable this feature, follow these steps:

1. Navigate to the **Privacy** section of the Teams Setting page.
2. Select the **Request Access** button next to text "Allow users with matching organization email domain to join this team". W&B Support will be notified of the request.
3. The **Request Access** button will disappear and the toggle is enabled when W&B Support verifies the request.
4. Select the newly enabled toggle.

## Avatar

Set an avatar by navigating to the **Avatar** section and uploading an image.

1. Select the **Update Avatar** to prompt a file dialog to appear.
2. From the file dialog, choose the image you want to use.

## Alerts

Notify your team when runs crash, finish, or set custom alerts. Your team can receive alerts either through email or Slack.

Toggle the switch next to the event type you want to receive alerts from. Weights and Biases provides the following event type options be default:

* **Runs finished**: whether a Weights and Biases run successfully finished.
* **Run crashed**: if a run has failed to finish.

For more information about how to set up and manage alerts, see [Send alerts with wandb.alert](../../runs/alert.md).

## Privacy

Navigate to the **Privacy** section to change privacy settings. Only members with Administrative roles can modify privacy settings. Administrator roles can:

* Force projects in the team to be private.
* Enable code saving by default.

## Usage

The **Usage** section describes the total memory usage the team has consumed on the Weights and Biases servers. The default storage plan is 100GB. For more information about storage and pricing, see the [Pricing](https://wandb.ai/site/pricing) page.

## Storage

The **Storage** section describes the cloud storage bucket configuration that is being used for the team's data. For more information, see [Secure Storage Connector](../features/teams.md#secure-storage-connector) or check out our [W&B Server](../../hosting/data-security/secure-storage-connector.md) docs if you are self-hosting. 