---
description: Visualize the gradients of your parameters
displayed_sidebar: default
---

# Gradient Panel

![Logged gradients get rendered as histograms](/images/app_ui/gradient_panels.png)

The gradient panel shows the histograms of the gradients, per time step.

Take the leftmost chart, `layer.10` weights. In the very first slice at Step 0, the grey shading indicates that the gradients for that layer had values between -40 and +40. The blue parts however indicate that most of those gradients were between -2 and +2 (roughly).

So, the shading represents the count of gradients in that particular histogram bin, for that particular time step.

Interpreting gradients can be tricky sometimes, but generally, these plots are useful to check that your gradients haven't exploded (big values on the y-axis) or collapsed (concentrated blue around 0 with little to no deviation).
