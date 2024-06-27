# Histogram

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/histogram.py#L18-L96' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


wandb class for histograms.

```python
Histogram(
    sequence: Optional[Sequence] = None,
    np_histogram: Optional['NumpyHistogram'] = None,
    num_bins: int = 64
) -> None
```

This object works just like numpy's histogram function
https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

#### Examples:

Generate histogram from a sequence

```python
wandb.Histogram([1, 2, 3])
```

Efficiently initialize from np.histogram.

```python
hist = np.histogram(data)
wandb.Histogram(np_histogram=hist)
```

| Arguments |  |
| :--- | :--- |
|  `sequence` |  (array_like) input data for histogram |
|  `np_histogram` |  (numpy histogram) alternative input of a precomputed histogram |
|  `num_bins` |  (int) Number of bins for the histogram. The default number of bins is 64. The maximum number of bins is 512 |

| Attributes |  |
| :--- | :--- |
|  `bins` |  ([float]) edges of bins |
|  `histogram` |  ([int]) number of elements falling in each bin |

| Class Variables |  |
| :--- | :--- |
|  `MAX_LENGTH`<a id="MAX_LENGTH"></a> |  `512` |
