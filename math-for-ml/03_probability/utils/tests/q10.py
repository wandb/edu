test = {
    "name": "sum_squared_error",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(sum_squared_error)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(sum_squared_error(mean, rand_array), float)
                    True
                    >>> ## the surprise is always non-negative
                    >>> np.greater_equal(sum_squared_error(mean, rand_array), 0.)
                    True
                    >>> ## the squared error for x == mu is 0
                    >>> np.isclose(sum_squared_error(mean, constant_array), 0.)
                    True
                    >>> sse_with_mean = sum_squared_error(mean, rand_array)
                    >>> ## the mean minimizes the squared error
                    >>> np.greater_equal(sum_squared_error(rand_array[0], rand_array), sse_with_mean)
                    True
                    >>> ## the squared error of the mean is equal to the variance * N
                    >>> np.isclose(sum_squared_error(mean, rand_array), len(rand_array) * variance)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> rand_array = np.random.randn(5)
            >>> mean = np.mean(rand_array)
            >>> constant_array = np.zeros(5) + mean
            >>> variance = np.var(rand_array)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
