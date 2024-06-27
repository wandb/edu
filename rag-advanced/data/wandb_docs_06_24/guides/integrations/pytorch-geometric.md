---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PyTorch Geometric

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) or PyG is one of the most popular libraries for geometric deep learning and W&B works extremely well with it for visualizing graphs and tracking experiments.

## Getting Started

After you have installed pytorch geometric, install the wandb library and login

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
pip install wandb
wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

## Visualizing the Graphs

You can save details about the input graphs including number of edges, number of nodes and more. W&B supports logging plotly charts and HTML panels so any visualizations you create for your graph can then also be logged to W&B.

### Using PyVis

The following snippet shows how you could do that with PyVis and HTML.

```python
from pyvis.network import Network
Import wandb

wandb.init(project=’graph_vis’)
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# Add the edges from the PyG graph to the PyVis network
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()

    net.add_node(dst)
    net.add_node(src)
    
    net.add_edge(src, dst, value=0.1)

# Save the PyVis visualisation to a HTML file
net.show("graph.html")
wandb.log({"eda/graph": wandb.Html("graph.html")})
wandb.finish()
```

| ![This image shows the input graph as an interactive HTML visualization.](@site/static/images/integrations/pyg_graph_wandb.png) | 
|:--:| 
| **This image shows the input graph as an interactive HTML visualization.** |

### Using Plotly

To use plotly to create a graph visualization, first you need to convert the PyG graph to a networkx object. Following this you will need to create Plotly scatter plots for both nodes and edges. The snippet below can be used for this task.

```python
def create_vis(graph):
    G = to_networkx(graph)
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        line_width=2
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout())

    return fig


wandb.init(project=’visualize_graph’)
wandb.log({‘graph’: wandb.Plotly(create_vis(graph))})
wandb.finish()
```

| ![This visualization was created using the function shown in the snippet above and longed inside a W&B Table.](@site/static/images/integrations/pyg_graph_plotly.png) | 
|:--:| 
| **This visualization was created using the function shown in the snippet above and longed inside a W&B Table.** |

## Logging Metrics

You can use W&B to track all your experiments along with metrics like loss functions, accuracy and more. Just add the following line to your training loop and you are good to go!

```python
wandb.log({
	‘train/loss’: training_loss,
	‘train/acc’: training_acc,
	‘val/loss’: validation_loss,
	‘val/acc’: validation_acc
})
```

| ![Plots from W&B showing how the hits@K metric changes over epochs for different values of K.](@site/static/images/integrations/pyg_metrics.png) | 
|:--:| 
| **Plots from W&B showing how the hits@K metric changes over epochs for different values of K.** |

## More Resources

- [Recommending Amazon Products using Graph Neural Networks in PyTorch Geometric](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [Point Cloud Classification using PyTorch Geometric](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [Point Cloud Segmentation using PyTorch Geometric](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)
