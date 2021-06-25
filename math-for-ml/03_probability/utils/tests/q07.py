test = {
    "name": "divergence",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(divergence)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(divergence(rand_pmf, rand_pmf), float)
                    True
                    >>> ## the divergence is always non-negative
                    >>> np.greater_equal(divergence(rand_pmf, coin_pmf), 0.)
                    True
                    >>> np.greater_equal(divergence(coin_pmf, rand_pmf), 0.)
                    True
                    >>> ## the order of arguments matters
                    >>> divergence(coin_pmf, rand_pmf) != divergence(rand_pmf, coin_pmf)
                    True
                    >>> ## the divergence between a pmf and itself is 0.
                    >>> np.isclose(divergence(coin_pmf, coin_pmf), 0.)
                    True
                    >>> np.isclose(divergence(rand_pmf, rand_pmf), 0.)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> rand_array = np.random.rand(2) + 0.05
            >>> rand_pmf = rand_array / np.sum(rand_array)
            >>> coin_pmf = np.array([0.5, 0.5])
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
