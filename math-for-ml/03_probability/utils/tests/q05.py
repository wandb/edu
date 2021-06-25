test = {
    "name": "entropy",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(entropy)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(entropy(rand_pmf), float)
                    True
                    >>> ## the entropy is always non-negative
                    >>> np.greater_equal(entropy(rand_pmf), 0.)
                    True
                    >>> ## the entropy of a fair coin is log(2)
                    >>> np.isclose(entropy(coin_pmf), log_2)
                    True
                    >>> ## the entropy of any other distribution on 2 states is lower
                    >>> entropy(coin_pmf) > entropy(rand_pmf)
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
            >>> log_2 = np.log(2)
            >>> rand_pmf_and_zero = np.append(rand_pmf, [0.])
            >>> one_hot_pmf = np.array([0., 0., 0., 1.])
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
