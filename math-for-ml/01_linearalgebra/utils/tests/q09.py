test = {
    "name": "make_repeater",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function called make_repeater
                    >>> callable(make_repeater)
                    True
                    >>> ## it should take two integer arguments
                    >>> out = make_repeater(2, 2)
                    >>> ## the return is of type ndarray
                    >>> type(out)
                    <class 'numpy.ndarray'>
                    """,
                    "hidden": False,
                    "locked": False
                },
                {
                    "code": r"""
                    >>> # make_repeater should be able to make repeat_3_2
                    >>> repeat_3_2 = make_repeater(3, 2)
                    >>> # the result should have two dimensions
                    >>> repeat_3_2.ndim
                    2
                    >>> # the result should have outputs of size 6 and inputs of size 2
                    >>> repeat_3_2.shape
                    (6, 2)
                    >>> # applying that repeater to the 0 vector should give a 0 vector
                    >>> np.allclose(np.zeros(6), repeat_3_2 @ zeros_2)
                    True
                    >>> # applying that repeater to the 1s vector should give a 1s vector
                    >>> np.allclose(np.ones(6), repeat_3_2 @ ones_2)
                    True
                    """
                }
            ],
            "setup": r"""
            zeros_2 = np.zeros(2)
            ones_2 = np.ones(2)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
