test = {
    "name": "crossentropy",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(crossentropy)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(crossentropy(rand_pmf, rand_pmf), float)
                    True
                    >>> ## the crossentropy is always non-negative
                    >>> np.greater_equal(crossentropy(rand_pmf, coin_pmf), 0.)
                    True
                    >>> np.greater_equal(crossentropy(coin_pmf, rand_pmf), 0.)
                    True
                    >>> ## the order of arguments matters
                    >>> crossentropy(coin_pmf, rand_pmf) != crossentropy(rand_pmf, coin_pmf)
                    True
                    >>> ## the entropy of any other distribution on 2 states is lower
                    >>> crossentropy(coin_pmf, coin_pmf) > crossentropy(rand_pmf, rand_pmf)
                    True
                    >>> ## the cross entropy is always higher than the entropy
                    >>> crossentropy(rand_pmf, coin_pmf) > crossentropy(rand_pmf, rand_pmf)
                    True
                    >>> ## the crossentropy of a fair coin with itself is log(2)
                    >>> np.isclose(crossentropy(coin_pmf, coin_pmf), log_2)
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
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
