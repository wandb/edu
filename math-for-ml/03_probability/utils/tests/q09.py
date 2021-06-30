test = {
    "name": "gaussian_surprise",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(gaussian_surprise)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(gaussian_surprise(mean, rand_array), float)
                    True
                    >>> ## the surprise is always non-negative
                    >>> np.greater_equal(gaussian_surprise(mean, rand_array), 0.)
                    True
                    >>> ## the squared error for x == mu is 0, so surprise is N * 1/2 log Z
                    >>> np.isclose(gaussian_surprise(mean, constant_array), len(constant_array) * 0.5 * log_Z)
                    True
                    >>> ## N * 1/2 log Z is the minimum possible surprise
                    >>> np.greater_equal(gaussian_surprise(mean, rand_array), len(rand_array) * 0.5 * log_Z)
                    True
                    >>> np.greater_equal(gaussian_surprise(rand_array[0], rand_array), len(rand_array) * 0.5 * log_Z)
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
            >>> log_Z = np.log(2. * np.pi)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
