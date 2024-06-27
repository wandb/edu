---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a collection

Create a collection within a registry to organize your artifacts. A *collection* is a set of linked artifact versions in a registry. Each collection represents a distinct task or use case and serves as a container for a curated selection of artifact versions related to that task.

![](/images/registry/what_is_collection.png)

For example, the preceding image shows a registry named "Forecast". Within the "Forecast" registry there are two collections called "LowLightPedRecog-YOLO" and "TextCat". 

:::tip
If you are familiar with W&B Model Registry, you might aware of "registered models". In W&B Registry, registered models are renamed to "collections". The way you [create a registered model in the Model Registry](../model_registry/create-registered-model.md) is nearly the same for creating a collection in the W&B Registry. The main difference being that a collection does not belong to an entity like registered models.
:::

The following steps describe how to create a collection within a registry using the W&B Registry App UI:

1. Navigate to the **Registry** App in the W&B App UI.
2. Select a registry.
3. Click on the **Create collection** button in the upper right hand corner.
4. Provide a name for your collection in the **Name** field. 
5. Select a type from the **Type** dropdown. Or, if the registry enables custom artifact types, provide one or more artifact types that this collection accepts.
:::info
An artifact type can not be removed from a registry once it is added and saved in the registry's settings.
:::
5. Optionally provide a description of your collection in the **Description** field.
6. Optionally add one or more tags in the **Tags** field. 
7. Click **Link version**.
8. From the **Project** dropdown, select the project where your artifact is stored.
9. From the **Artifact** collection dropdown, select your artifact.
10. From the **Version** dropdown, select the artifact version you want to link to your collection.
11. Click on the **Create collection** button.

![](/images/registry/create_collection.gif)



