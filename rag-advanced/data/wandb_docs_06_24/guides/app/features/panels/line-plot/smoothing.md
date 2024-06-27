---
description: In line plots, use smoothing to see trends in noisy data.
displayed_sidebar: default
---

# Smoothing

In W&B line plots, we support three types of smoothing:

- [exponential moving average](smoothing.md#exponential-moving-average-default) (default)
- [gaussian smoothing](smoothing.md#gaussian-smoothing)
- [running average](smoothing.md#running-average)
- [exponential moving average - Tensorboard](smoothing.md#exponential-moving-average-tensorboard) (deprecated)

See these live in an [interactive W&B report](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc).

![](/images/app_ui/beamer_smoothing.gif)

## Exponential Moving Average (Default)

Exponential smoothing is a technique for smoothing time series data by exponentially decaying the weight of previous points. The range is 0 to 1. See [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing) for background. There is a de-bias term added so that early values in the time series are not biased towards zero.

The EMA algorithm takes the density of points on the line (i.e. the number of `y` values per unit of range on x-axis) into account. This allows consistent smoothing when displaying multiple lines with different characteristics simultaneously.

Here is sample code for how this works under the hood:

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE scales the result to the chart's x-axis range
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

Here's what this looks like [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

![](/images/app_ui/weighted_exponential_moving_average.png)

## Gaussian Smoothing

Gaussian smoothing (or gaussian kernel smoothing) computes a weighted average of the points, where the weights correspond to a gaussian distribution with the standard deviation specified as the smoothing parameter. See . The smoothed value is calculated for every input x value.

Gaussian smoothing is a good standard choice for smoothing if you are not concerned with matching TensorBoard's behavior. Unlike an exponential moving average the point will be smoothed based on points occurring both before and after the value.

Here's what this looks like [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

![](/images/app_ui/gaussian_smoothing.png)

## Running Average

Running average is a smoothing algorithm that replaces a point with the average of points in a window before and after the given x value. See "Boxcar Filter" at [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average). The selected parameter for running average tells Weights and Biases the number of points to consider in the moving average.

Consider using Gaussian Smoothing if your points are spaced unevenly on the x-axis.

The following image demonstrates how a running app looks like [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

![](/images/app_ui/running_average.png)

## Exponential Moving Average (Deprecated)

> The TensorBoard EMA algorithm has been deprecated as it cannot accurately smooth multiple lines on the same chart that do not have a consistent point density (number of points plotted per unit of x-axis).

Exponential moving average is implemented to match TensorBoard's smoothing algorithm. The range is 0 to 1. See [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing) for background. There is a debias term added so that early values in the time series are not biases towards zero.

Here is sample code for how this works under the hood:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

Here's what this looks like [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

![](/images/app_ui/exponential_moving_average.png)

## Implementation Details

All of the smoothing algorithms run on the sampled data, meaning that if you log more than 1500 points, the smoothing algorithm will run _after_ the points are downloaded from the server. The intention of the smoothing algorithms is to help find patterns in data quickly. If you need exact smoothed values on metrics with a large number of logged points, it may be better to download your metrics through the API and run your own smoothing methods.

## Hide original data

By default we show the original, unsmoothed data as a faint line in the background. Click the **Show Original** toggle to turn this off.

![](/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif)
