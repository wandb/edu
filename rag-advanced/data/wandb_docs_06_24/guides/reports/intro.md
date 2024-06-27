---
slug: /guides/reports
description: Project management and collaboration tools for machine learning projects
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Collaborative Reports

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb"/>

Use W&B Reports to organize Runs, embed and automate visualizations, describe your findings, and share updates with collaborators. Easily export your report as a LaTeX zip file or  convert the file to PDF.

<!-- {% embed url="https://www.youtube.com/watch?v=2xeJIv_K_eI" %} -->

The following image shows a section of a report created from metrics that were logged to W&B over the course of training. 

![](/images/reports/safe-lite-benchmark-with-comments.png)

View the report where the above image was taken from [here](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM).

## How it works
Create a collaborative report with a few clicks.

1. Navigate to your W&B project workspace in the W&B App.
2. Click the **Create report** button in the upper right corner of your workspace.

![](/images/reports/create_a_report_button.png)

3. A modal titled **Create Report** will appear. Select the charts and panels you want to add to your report. (You can add or remove charts and panels later).
4. Click **Create report**. 
5. Edit the report to your desired state. 
6. Click **Publish to project**.
7. Click the **Share** button to share your report with collaborators. 

See the [Create a report](./create-a-report.md) page for more information on how to create reports interactively an programmatically with the W&B Python SDK.

## How to get started
Depending on your use case, explore the following resources to get started with W&B Reports:

* Check out our [video demonstration](https://www.youtube.com/watch?v=2xeJIv\_K\_eI) to get an overview of W&B Reports.
* Explore the [Reports gallery](./reports-gallery.md) for examples of live reports.
* Read curated Reports in [W&B Fully Connected](http://wandb.me/fc).
