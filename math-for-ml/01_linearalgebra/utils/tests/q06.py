test = {
    "name": "Zero Second and Repeat Thrice",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define an array with right name
                    >>> isinstance(set_second_to_zero_and_repeat_3, np.ndarray)
                    True
                    >>> ## it should have two dimensions
                    >>> set_second_to_zero_and_repeat_3.ndim
                    2
                    >>> ## it takes length-2 vectors as input
                    >>> set_second_to_zero_and_repeat_3.shape[1]
                    2
                    >>> ## its output has 3 times the length of the input
                    >>> set_second_to_zero_and_repeat_3.shape[0] // 3
                    2
                    >>> ## its output is three copies of the input
                    >>> set_second_to_zero_and_repeat_3 @ [1., 2.]
                    array([1., 0., 1., 0., 1., 0.])
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
