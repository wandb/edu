test = {
    "name": "Repeat Thrice",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define an array with right name
                    >>> isinstance(repeat_3_2, np.ndarray)
                    True
                    >>> ## it should have two dimensions
                    >>> repeat_3_2.ndim
                    2
                    >>> ## it takes length-2 vectors as input
                    >>> repeat_3_2.shape[1]
                    2
                    >>> ## its output has 3 times the length of the input
                    >>> repeat_3_2.shape[0] // 3
                    2
                    >>> ## its output is three copies of the input
                    >>> repeat_3_2 @ [1., 2.]
                    array([1., 2., 1., 2., 1., 2.])
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
