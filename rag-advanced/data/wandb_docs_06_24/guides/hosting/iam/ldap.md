---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SSO using LDAP

Authenticate your credentials with the W&B Server LDAP server. The following guide explains how to configure the settings for W&B Server. It covers mandatory and optional configurations, as well as instructions for configuring the LDAP connection from systems settings UI. it also provides information on the different inputs of the LDAP configuration, such as the address, base distinguished name, and attributes. You can specify these attributes from the W&B App UI or using environment variables. You can setup either an anonymous bind, or bind with an administrator DN and Password.

<!-- :::tip
As a W&B Team Admin you can setup either an anonymous bind, or bind with an administrator DN and Password.
::: -->

:::tip
Only W&B Admin roles can enable and configure LDAP authentication.
:::

## Configure LDAP connection

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'Environment variables', value: 'env'},
    
  ]}>
  <TabItem value="app">

1. Navigate to the W&B App. 
2. Select your profile icon from the upper right. From the dropdown, select **System Settings**. 
3. Toggle **Configure LDAP Client**.
4. Add the details in the form. Refer to **Configuring Parameters** section for details on each input.
5. Click on **Update Settings** to test your settings. This will establish a test client/connection with the W&B server.
6. If your connection is verified, toggle the **Enable LDAP Authentication** and select the **Update Settings** button.

<!-- Why is step # 6 necessary? -->

  </TabItem>
  <TabItem value="env">

Set LDAP an connection with the following environment variables:

| Environment variable          | Required | Example                         |
| ----------------------------- | -------- | ------------------------------- |
| `LOCAL_LDAP_ADDRESS`          | Yes      | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`          | Yes      | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`          | No       | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`          | No       |                                 |
| `LOCAL_LDAP_ATTRIBUTES`       | Yes      | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`       | No       |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | No       |                                 |
| `LOCAL_LDAP_LOGIN`            | No       |                                 |

See the [Configuration parameters](#configuration-parameters) section for definitions of each environment variable. Note that the environment variable prefix `LOCAL_LDAP` was omitted from the definition names for clarity.

  </TabItem>
</Tabs>

## Configuration parameters

<!-- |Environment variable|Definition| Required | Example |
|-----|-----|-----|-----|
|`LOCAL_LDAP_ADDRESS`| This is the address of your LDAP server within the VPC that hosts W&B Server.| Yes |`ldaps://ldap.example.com:636`|
|`LOCAL_LDAP_BASE_DN`|The root path searches start from and required for doing any queries into this directory.| Yes | |
|`LOCAL_LDAP_BIND_DN`|Path of the administrative user registered in the LDAP server. This is required if the LDAP server does not support unauthenticated binding. If specified, W&B Server connects to the LDAP server as this user. Otherwise, W&B Server connects using anonymous binding.| No | `cn=admin`, `dc=example,dc=org`|
|`LOCAL_LDAP_BIND_PW`|The password for administrative user, this is used to authenticate the binding. If left blank, W&B Server connects using anonymous binding.| No | |
|`LOCAL_LDAP_ATTRIBUTES`|Provide an email and group ID attribute names as comma separated string values. |Yes |`email=mail`, `group=gidNumber`|
|`LOCAL_LDAP_TLS_ENABLE`|Enable TLS.|No | |
|`LOCAL_LDAP_GROUP_ALLOW_LIST`|Group allowlist.| | |
|`LOCAL_LDAP_LOGIN`|This tells W&B Server to use LDAP to authenticate. Set to either `True` or `False`. Optionally set this to false to test the LDAP configuration. Set this to true to start LDAP authentication.| No |  | -->

The following table lists and describes required and optional LDAP configurations.

| Environment variable | Definition                                                                                                                                                                                                                                                              | Required |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --- |
| `ADDRESS`            | This is the address of your LDAP server within the VPC that hosts W&B Server.                                                                                                                                                                                           | Yes      |
| `BASE_DN`            | The root path searches start from and required for doing any queries into this directory.                                                                                                                                                                               | Yes      |
| `BIND_DN`            | Path of the administrative user registered in the LDAP server. This is required if the LDAP server does not support unauthenticated binding. If specified, W&B Server connects to the LDAP server as this user. Otherwise, W&B Server connects using anonymous binding. | No       |
| `BIND_PW`            | The password for administrative user, this is used to authenticate the binding. If left blank, W&B Server connects using anonymous binding.                                                                                                                             | No       |     |
| `ATTRIBUTES`         | Provide an email and group ID attribute names as comma separated string values.                                                                                                                                                                                         | Yes      |
| `TLS_ENABLE`         | Enable TLS.                                                                                                                                                                                                                                                             | No       |
| `GROUP_ALLOW_LIST`   | Group allowlist.                                                                                                                                                                                                                                                        | No       |
| `LOGIN`              | This tells W&B Server to use LDAP to authenticate. Set to either `True` or `False`. Optionally set this to false to test the LDAP configuration. Set this to true to start LDAP authentication.                                                                         | No       |
