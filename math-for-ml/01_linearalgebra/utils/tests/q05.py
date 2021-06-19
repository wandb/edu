test = {
    "name": "Matrix Type Checking",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function called are_compatible
                    >>> callable(are_compatible)
                    True
                    >>> ## that function should return booleans
                    >>> isinstance(are_compatible(A, B), bool)
                    True
                    >>> ## when the inner shapes are the same, return True
                    >>> are_compatible(A, B)
                    True
                    >>> ## when the inner shapes differ, return False
                    >>> are_compatible(A, M)
                    False
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            random_outer_shapes = np.random.randint(1, 5, size=2)
            random_inner_shape = np.random.randint(1, 5)
            A = np.random.randn(random_outer_shapes[0], random_inner_shape)
            B = np.random.randn(random_inner_shape, random_outer_shapes[1])
            M = np.random.randn(random_inner_shape + 1, random_outer_shapes[1])
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
