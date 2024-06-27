---
displayed_sidebar: default
---

# Advanced configuration

In addition to basic [environment variables](../env-vars.md), you can use environment variables to configure IAM options for your [Dedicated Cloud](../hosting-options/dedicated_cloud.md) or [Self-managed](../hosting-options/self-managed.md) instance.

Choose any of the following environment variables for your instance depending on your IAM needs.

| Environment variable | Description |
|----------------------|-------------|
| DISABLE_SSO_PROVISIONING | Set this to `true` to disable user auto-provisioning in your W&B instance. |
| SESSION_LENGTH | If you would like to change the default user session expiry time, set this variable to the desired number of hours. For example, set SESSION_LENGTH to `24` to configure session expiry time to 24 hours. The default value is 720 hours. |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | If you are using OIDC based SSO, set this variable to `true` to automate W&B team membership in your instance based on your OIDC groups. Add a `groups` claim to user OIDC token. It should be a string array where each entry is the name of a W&B team that the user should belong to. The array should include all the teams that a user is a part of. |
| GORILLA_LDAP_GROUP_SYNC | If you are using LDAP based SSO, set it to `true` to automate W&B team membership in your instance based on your LDAP groups. |
| GORILLA_OIDC_CUSTOM_SCOPES | If you are using OIDC based SSO, you can specify additional [scopes](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes) that W&B instance should request from your identity provider. W&B does not change the SSO functionality due to these custom scopes in any way. |
| GORILLA_USE_IDENTIFIER_CLAIMS | If you are using OIDC based SSO, set this variable to `true` to enforce username and full name of your users using specific OIDC claims from your identity provider. If set, ensure that you configure the enforced username and full name in the `preferred_username` and `name` OIDC claims respectively. |
| GORILLA_DISABLE_PERSONAL_ENTITY | Set this to `true` to disable personal user projects in your W&B instance. If set, users can not create new personal projects in their personal entities, plus writes to existing personal projects are disabled. |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | Set this to `true` to restrict Organization or Instance Admins from self-joining or adding themselves to a W&B team, thus ensuring that only Data & AI personas have access to the projects within the teams. |

:::caution
W&B advises to exercise caution and understand all implications before enabling some of these settings, like `GORILLA_DISABLE_ADMIN_TEAM_ACCESS`. Reach out to your W&B team for any questions.
:::