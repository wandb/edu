test = {
    "name": "Normalize",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function called normalize
                    >>> callable(normalize)
                    True
                    >>> ## testing on all_ones vector first
                    >>> int(np.round(np.linalg.norm(normalize(all_ones))))
                    1
                    >>> np.allclose(normalize(all_ones), 1 / np.sqrt(shape))
                    True
                    >>> ## and now on a random vector
                    >>> int(np.round(np.linalg.norm(normalize(random_vector))))
                    1
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            shape = 5
            all_ones = np.ones(shape)
            random_vector = np.random.randn(shape)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
