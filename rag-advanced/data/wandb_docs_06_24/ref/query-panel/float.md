# float

## Chainable Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

Determines inequality of two values.

| Argument |  |
| :--- | :--- |
| `lhs` | The first value to compare. |
| `rhs` | The second value to compare. |

#### Return Value
Whether the two values are not equal.

<h3 id="number-modulo"><code>number-modulo</code></h3>

Divide a [number](https://docs.wandb.ai/ref/weave/number) by another and return remainder

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to divide |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to divide by |

#### Return Value
Modulo of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-mult"><code>number-mult</code></h3>

Multiply two [numbers](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `lhs` | First [number](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | Second [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
Product of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

Raise a [number](https://docs.wandb.ai/ref/weave/number) to an exponent

| Argument |  |
| :--- | :--- |
| `lhs` | Base [number](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | Exponent [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
The base [numbers](https://docs.wandb.ai/ref/weave/number) raised to nth power

<h3 id="number-add"><code>number-add</code></h3>

Add two [numbers](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `lhs` | First [number](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | Second [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
Sum of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-sub"><code>number-sub</code></h3>

Subtract a [number](https://docs.wandb.ai/ref/weave/number) from another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to subtract from |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to subtract |

#### Return Value
Difference of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-div"><code>number-div</code></h3>

Divide a [number](https://docs.wandb.ai/ref/weave/number) by another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to divide |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to divide by |

#### Return Value
Quotient of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-less"><code>number-less</code></h3>

Check if a [number](https://docs.wandb.ai/ref/weave/number) is less than another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare to |

#### Return Value
Whether the first [number](https://docs.wandb.ai/ref/weave/number) is less than the second

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

Check if a [number](https://docs.wandb.ai/ref/weave/number) is less than or equal to another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare to |

#### Return Value
Whether the first [number](https://docs.wandb.ai/ref/weave/number) is less than or equal to the second

<h3 id="number-equal"><code>number-equal</code></h3>

Determines equality of two values.

| Argument |  |
| :--- | :--- |
| `lhs` | The first value to compare. |
| `rhs` | The second value to compare. |

#### Return Value
Whether the two values are equal.

<h3 id="number-greater"><code>number-greater</code></h3>

Check if a [number](https://docs.wandb.ai/ref/weave/number) is greater than another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare to |

#### Return Value
Whether the first [number](https://docs.wandb.ai/ref/weave/number) is greater than the second

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

Check if a [number](https://docs.wandb.ai/ref/weave/number) is greater than or equal to another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare to |

#### Return Value
Whether the first [number](https://docs.wandb.ai/ref/weave/number) is greater than or equal to the second

<h3 id="number-negate"><code>number-negate</code></h3>

Negate a [number](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `val` | Number to negate |

#### Return Value
A [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-toString"><code>number-toString</code></h3>

Convert a [number](https://docs.wandb.ai/ref/weave/number) to a string

| Argument |  |
| :--- | :--- |
| `in` | Number to convert |

#### Return Value
String representation of the [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

Converts a [number](https://docs.wandb.ai/ref/weave/number) to a _timestamp_. Values less than 31536000000 will be converted to seconds, values less than 31536000000000 will be converted to milliseconds, values less than 31536000000000000 will be converted to microseconds, and values less than 31536000000000000000 will be converted to nanoseconds.

| Argument |  |
| :--- | :--- |
| `val` | Number to convert to a timestamp |

#### Return Value
Timestamp

<h3 id="number-abs"><code>number-abs</code></h3>

Calculates the absolute value of a [number](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `n` | A [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
The absolute value of the [number](https://docs.wandb.ai/ref/weave/number)


## List Ops
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

Determines inequality of two values.

| Argument |  |
| :--- | :--- |
| `lhs` | The first value to compare. |
| `rhs` | The second value to compare. |

#### Return Value
Whether the two values are not equal.

<h3 id="number-modulo"><code>number-modulo</code></h3>

Divide a [number](https://docs.wandb.ai/ref/weave/number) by another and return remainder

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to divide |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to divide by |

#### Return Value
Modulo of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-mult"><code>number-mult</code></h3>

Multiply two [numbers](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `lhs` | First [number](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | Second [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
Product of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

Raise a [number](https://docs.wandb.ai/ref/weave/number) to an exponent

| Argument |  |
| :--- | :--- |
| `lhs` | Base [number](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | Exponent [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
The base [numbers](https://docs.wandb.ai/ref/weave/number) raised to nth power

<h3 id="number-add"><code>number-add</code></h3>

Add two [numbers](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `lhs` | First [number](https://docs.wandb.ai/ref/weave/number) |
| `rhs` | Second [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
Sum of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-sub"><code>number-sub</code></h3>

Subtract a [number](https://docs.wandb.ai/ref/weave/number) from another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to subtract from |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to subtract |

#### Return Value
Difference of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-div"><code>number-div</code></h3>

Divide a [number](https://docs.wandb.ai/ref/weave/number) by another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to divide |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to divide by |

#### Return Value
Quotient of two [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-less"><code>number-less</code></h3>

Check if a [number](https://docs.wandb.ai/ref/weave/number) is less than another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare to |

#### Return Value
Whether the first [number](https://docs.wandb.ai/ref/weave/number) is less than the second

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

Check if a [number](https://docs.wandb.ai/ref/weave/number) is less than or equal to another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare to |

#### Return Value
Whether the first [number](https://docs.wandb.ai/ref/weave/number) is less than or equal to the second

<h3 id="number-equal"><code>number-equal</code></h3>

Determines equality of two values.

| Argument |  |
| :--- | :--- |
| `lhs` | The first value to compare. |
| `rhs` | The second value to compare. |

#### Return Value
Whether the two values are equal.

<h3 id="number-greater"><code>number-greater</code></h3>

Check if a [number](https://docs.wandb.ai/ref/weave/number) is greater than another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare to |

#### Return Value
Whether the first [number](https://docs.wandb.ai/ref/weave/number) is greater than the second

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

Check if a [number](https://docs.wandb.ai/ref/weave/number) is greater than or equal to another

| Argument |  |
| :--- | :--- |
| `lhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare |
| `rhs` | [number](https://docs.wandb.ai/ref/weave/number) to compare to |

#### Return Value
Whether the first [number](https://docs.wandb.ai/ref/weave/number) is greater than or equal to the second

<h3 id="number-negate"><code>number-negate</code></h3>

Negate a [number](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `val` | Number to negate |

#### Return Value
A [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

Finds the index of maximum [number](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `numbers` | _list_ of [numbers](https://docs.wandb.ai/ref/weave/number) to find the index of maximum [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
Index of maximum [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

Finds the index of minimum [number](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `numbers` | _list_ of [numbers](https://docs.wandb.ai/ref/weave/number) to find the index of minimum [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
Index of minimum [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

Average of [numbers](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `numbers` | _list_ of [numbers](https://docs.wandb.ai/ref/weave/number) to average |

#### Return Value
Average of [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-max"><code>numbers-max</code></h3>

Maximum number

| Argument |  |
| :--- | :--- |
| `numbers` | _list_ of [numbers](https://docs.wandb.ai/ref/weave/number) to find the maximum [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
Maximum [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-min"><code>numbers-min</code></h3>

Minimum number

| Argument |  |
| :--- | :--- |
| `numbers` | _list_ of [numbers](https://docs.wandb.ai/ref/weave/number) to find the minimum [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
Minimum [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

Standard deviation of [numbers](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `numbers` | _list_ of [numbers](https://docs.wandb.ai/ref/weave/number) to calculate the standard deviation |

#### Return Value
Standard deviation of [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

Sum of [numbers](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `numbers` | _list_ of [numbers](https://docs.wandb.ai/ref/weave/number) to sum |

#### Return Value
Sum of [numbers](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-toString"><code>number-toString</code></h3>

Convert a [number](https://docs.wandb.ai/ref/weave/number) to a string

| Argument |  |
| :--- | :--- |
| `in` | Number to convert |

#### Return Value
String representation of the [number](https://docs.wandb.ai/ref/weave/number)

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

Converts a [number](https://docs.wandb.ai/ref/weave/number) to a _timestamp_. Values less than 31536000000 will be converted to seconds, values less than 31536000000000 will be converted to milliseconds, values less than 31536000000000000 will be converted to microseconds, and values less than 31536000000000000000 will be converted to nanoseconds.

| Argument |  |
| :--- | :--- |
| `val` | Number to convert to a timestamp |

#### Return Value
Timestamp

<h3 id="number-abs"><code>number-abs</code></h3>

Calculates the absolute value of a [number](https://docs.wandb.ai/ref/weave/number)

| Argument |  |
| :--- | :--- |
| `n` | A [number](https://docs.wandb.ai/ref/weave/number) |

#### Return Value
The absolute value of the [number](https://docs.wandb.ai/ref/weave/number)

