test = {
    "name": "Norm",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function called norm
                    >>> callable(norm)
                    True
                    >>> ## all-zeros vector has norm 0
                    >>> int(np.round(norm(all_zeros)))
                    0
                    >>> ## vector with one 1 has norm 1
                    >>> int(np.round(norm(one_hot)))
                    1
                    >>> ## all ones vector of length n
                    >>> ## has norm sqrt(n)
                    >>> np.isclose(norm(all_ones), all_ones_norm)
                    True
                    >>> ## we should get same answer as numpy
                    >>> np.isclose(np.linalg.norm(random_vector), norm(random_vector))
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            shape = 5
            all_zeros = np.zeros(shape)
            one_hot = np.array([1, 0, 0, 0, 0])
            all_ones = np.ones(shape)
            all_ones_norm = np.sqrt(shape)
            random_vector = np.random.randn(shape)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
