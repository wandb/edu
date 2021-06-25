test = {
    "name": "softmax_crossentropy",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(softmax_crossentropy)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(softmax_crossentropy(rand_logits, rand_pmf), float)
                    True
                    >>> ## the crossentropy is always non-negative
                    >>> np.greater_equal(softmax_crossentropy(rand_logits, coin_pmf), 0.))
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> rand_logits = np.random.rand(2) + 0.05
            >>> rand_pmf = rand_array / np.sum(rand_array)
            >>> coin_pmf = np.array([0.5, 0.5])
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
