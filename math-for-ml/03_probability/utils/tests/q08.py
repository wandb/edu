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
                    >>> np.greater_equal(softmax_crossentropy(rand_logits, coin_pmf), 0.)
                    True
                    >>> ## the crossentropy is the entropy when p == q
                    >>> np.isclose(softmax_crossentropy(coin_logits, coin_pmf), log_2)
                    True
                    >>> ## the crossentropy does not change
                    >>> ## if the logits are additively scaled
                    >>> np.isclose(softmax_crossentropy(coin_logits + 1., coin_pmf), log_2)
                    True
                    >>> ## the crossentropy does not change
                    >>> ## if the logits are multiplicatively scaled
                    >>> np.isclose(softmax_crossentropy(coin_logits * -1., coin_pmf), log_2)
                    True
                    >>> ## carefully check the argument order for the crossentropy!
                    >>> ## the crossentropy of the fair coin on any distribution is log 2
                    >>> np.isclose(softmax_crossentropy(coin_logits, rand_pmf), log_2)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> rand_logits = np.random.rand(2) + 0.05
            >>> rand_pmf = rand_logits / np.sum(rand_logits)
            >>> coin_logits = np.array([rand_logits[0]] * 2)
            >>> coin_pmf = np.array([0.5, 0.5])
            >>> log_2 = np.log(2.)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
