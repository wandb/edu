test = {
    "name": "array_is_pmf",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(array_is_pmf)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(array_is_pmf(np.array([0.])), bool)
                    True
                    >>> ## an array with negative entries is not a pmf
                    >>> array_is_pmf(np.array([-1.]))
                    False
                    >>> ## an array that doesn't sum to 1 is not a pmf
                    >>> array_is_pmf(np.array([0.5, 0.4]))
                    False
                    >>> ## the rand_array variable is never a valid pmf
                    >>> array_is_pmf(rand_array)
                    False
                    >>> ## some numerical error in the sum is tolerable
                    >>> ## so use np.isclose, not ==
                    >>> array_is_pmf(close_to_1)
                    True
                    >>> ## the rand_pmf variable is always a pmf
                    >>> array_is_pmf(rand_pmf)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> rand_array = 1 / 5 * np.random.rand(4)
            >>> rand_pmf = rand_array / np.sum(rand_array)
            >>> close_to_1 = np.array([1/100 for ii in range(100)])
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
