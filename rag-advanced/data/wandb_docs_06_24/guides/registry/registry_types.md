---
displayed_sidebar: default
---

# Registry types

W&B supports two types of registries: [Core registries](#core-registry) and [Custom registries](#custom-registry). 

## Core registry
A core registry is a template for specific use cases: **Models** and **Datasets**.

By default, the **Models** registry is configured to accept `"model"` artifact types and the **Dataset** registry is configured to accept `"dataset"` artifact types. An admin can add additional accepted artifact types. 

<!-- For more information about artifact types, see [LINK]. -->

![](/images/registry/core_registry_example.png)

The preceding image shows the **Models** and the **Dataset** core registry along with a custom registry called **Fine_Tuned_Models** in the W&B Registry App UI.

A core registry has [organization visibility](./configure_registry.md#registry-visibility-types). A registry admin can not change the visibility of a core registry. 

## Custom registry
Custom registries are not restricted to `"model"` artifact types or `"dataset"` artifact types.

You can create a custom registry for each step in your machine learning pipeline, from initial data collection to final model deployment.

For example, you might create a registry called "Benchmark_Datasets" for organizing curated datasets to evaluate the performance of trained models. Within this registry, you might have a collection called "User_Query_Insurance_Answer_Test_Data" that contains a set of user questions and corresponding expert-validated answers that the model has never seen during training. 

![](/images/registry/custom_registry_example.png)

A custom registry can have either [organization or restricted visibility](./configure_registry.md#registry-visibility-types). A registry admin can change the visibility of a custom registry from organization to restricted. However, the registry admin can not change a custom registry's visibility from restricted to organizational visibility.

For information on how to create a custom registry, see [Create a custom registry](./create_collection.md).


## Summary
The proceeding table summarizes the differences between core and custom registries:

|                | Core  | Custom|
| -------------- | ----- | ----- |
| Visibility     | Organizational visibility only. Visibility can not be altered. | Either organization or restricted. Visibility can be altered from organization to restricted visibility.|
| Metadata       | Preconfigured and not editable by users. | Users can edit.  |
| Artifact types | Preconfigured and accepted artifact types cannot be removed. Users can add additional accepted artifact types. | Admin can define accepted types. |
| Customization    | Can add additional types to the existing list.|  Edit registry name, description, visibility, and accepted artifact types.|



