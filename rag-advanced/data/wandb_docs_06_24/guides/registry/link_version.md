---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Link an artifact version to a registry

Programmatically or interactively link artifact versions to a registry.

:::info
When you link an artifact to a registry, this "publishes" that artifact to that registry. Any user that has access to that registry can access linked artifact versions when you link an artifact to a collection.

In other words, linking an artifact to a registry collection brings that artifact version from a private, project-level scope, to the shared organization level scope.
:::

Based on your use case, follow the instructions described in the tabs below to link an artifact version.

<Tabs
  defaultValue="python_sdk"
  values={[
    {label: 'Python SDK', value: 'python_sdk'},
    {label: 'Registry App', value: 'registry_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="python_sdk">


Use the [`link_artifact`](../../ref/python/run.md#link_artifact) method to programmatically link an artifact to a registry. When you link an artifact, specify the path where you want artifact version to link to for the `target_path` parameter. The target path takes the form of `"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"`.

Replace values enclosed in `<>` with your own:

```python
import wandb

ARTIFACT_NAME = "<ARTIFACT-TO-LINK>"
ARTIFACT_TYPE = "ARTIFACT-TYPE"
ENTITY_NAME = "<TEAM-ARTIFACT-BELONGS-IN>"
PROJECT_NAME = "<PROJECT-ARTIFACT-TO-LINK-BELONGS-IN>"

ORG_ENTITY_NAME = "<YOUR ORG NAME>"
REGISTRY_NAME = "<REGISTRY-TO-LINK-TO>"
COLLECTION_NAME = "<REGISTRY-COLLECTION-TO-LINK-TO>"

run = wandb.init(entity=ENTITY_NAME, project=PROJECT_NAME)
artifact = wandb.Artifact(name=ARTIFACT_NAME, type=ARTIFACT_TYPE)
run.link_artifact(
    artifact=artifact,
    target_path=f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
run.finish()
```

If you want to link an artifact version to the **Models** registry or the **Dataset** registry, set the artifact type to `"model"` or `"dataset"`, respectively.

  </TabItem>
  <TabItem value="registry_ui">

1. Navigate to the Registry App.
![](/images/registry/navigate_to_registry_app.png)
2. Hover your mouse next to the name of the collection you want to link an artifact version to.
3. Select the meatball menu icon (three horizontal dots) next to  **View details**.
4. From the dropdown, select **Link new version**.
5. From the sidebar that appears, select the name of a team from the **Team** dropdown.
5. From the **Project** dropdown, select the name of the project that contains your artifact. 
6. From the **Artifact** dropdown, select the name of the artifact. 
7. From the **Version** dropdown, select the artifact version you want to link to the collection.

<!-- TO DO insert gif -->

  </TabItem>
  <TabItem value="artifacts_ui">

1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Select the Artifacts icon on the left sidebar.
3. Click on the artifact version you want to link to your registry.
4. Within the **Version overview** section, click the **Link to registry** button.
5. From the modal that appears on the right of the screen, select an artifact from the **Select a register model** menu dropdown. 
6. Click **Next step**.
7. (Optional) Select an alias from the **Aliases** dropdown. 
8. Click **Link to registry**. 

<!-- Update this gif -->
<!-- ![](/images/models/manual_linking.gif) -->

  </TabItem>
</Tabs>


:::tip Linked vs source artifact versions
* Source version: the artifact version inside a team's project that is logged to a [run](../runs/intro.md).
* Linked version: the artifact version that is published to the registry. This is a pointer to the source artifact, and is the exact same artifact version, just made available in the scope of the registry.
:::