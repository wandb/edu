test = {
    "name": "Refactoring",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define an array V
                    >>> type(V)
                    <class 'numpy.ndarray'>
                    >>> ## make sure you multiply in the right order!
                    >>> np.array_equal(WXYZ @ random_vec, V @ random_vec)
                    False
                    >>> ## result from their pipeline and yours should be (almost) same
                    >>> np.allclose(their_pipeline(random_vec), V @ random_vec)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            WXYZ = W @ X @ Y @ Z
            random_vec = np.random.standard_normal(size=(2, 1))
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
