test = {
    "name": "softmax",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(softmax)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(softmax(three_ones), np.ndarray)
                    True
                    >>> ## the entries are always non-negative
                    >>> np.all(np.greater_equal(softmax(rand_array), 0.))
                    True
                    >>> ## the entries should sum to 1
                    >>> np.isclose(np.sum(softmax(rand_array)), 1.)
                    True
                    >>> ## applying softmax shouldn't change which entry is biggest
                    >>> np.argmax(rand_array) == np.argmax(softmax(rand_array))
                    True
                    >>> ## if all the entries are the same, all outputs are the same
                    >>> np.allclose(softmax(three_ones), 1 / 3)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> three_ones = np.ones(3)
            >>> rand_array = np.random.rand(10)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
