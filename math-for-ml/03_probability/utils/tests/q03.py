test = {
    "name": "surprise",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(surprise)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(surprise(simple_pmf, 0), float)
                    True
                    >>> ## for the same numerical input, should give the same output
                    >>> surprise(simple_pmf, 0) == surprise(simple_pmf, 1)
                    True
                    >>> ## the surprise for probability 1 is 0
                    >>> np.isclose(surprise(constant_pmf, 0), 0.)
                    True
                    >>> ## the inverse of the surprise is the negative exponent
                    >>> neg_exps = [np.exp(-1 * surprise(rand_pmf, ii)) for ii in range(4)]
                    >>> np.allclose(rand_pmf, neg_exps)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> constant_pmf = np.array([1.])
            >>> simple_pmf = np.array([0.5, 0.5])
            >>> rand_array = 1 / 5 * np.random.rand(4) + 0.01
            >>> rand_pmf = rand_array / np.sum(rand_array)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
