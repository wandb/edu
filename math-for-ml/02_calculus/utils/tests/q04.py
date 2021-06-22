test = {
    "name": "gd_step",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define an array with right name
                    >>> callable(gd_step)
                    True
                    >>> ## it should run on and return the right types
                    >>> isinstance(gd_step(0., np.square, 0.), float)
                    True
                    >>> ## if learning rate is zero, change is zero
                    >>> np.isclose(gd_step(0., constant_0, 0.), 0.)
                    True
                    >>> np.isclose(gd_step(0., constant, 0.), 0.)
                    True
                    >>> np.isclose(gd_step(val, identity, 0.), val)
                    True
                    >>> np.isclose(gd_step(val, np.square, 0.), val)
                    True
                    >>> ## gradient of identity is 1
                    >>> np.isclose(gd_step(1., identity, 1.), 0.)
                    True
                    >>> np.isclose(gd_step(-2., identity, 1.), -3.)
                    True
                    >>> ## gradient of square is 2 * x
                    >>> np.isclose(gd_step(val, np.square, 1/2), 0.)
                    True
                    >>> np.isclose(gd_step(val, np.square, 1), -val)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            val = np.random.randn()
            print(f"Testing on random value {val}")
            # identity function returns its inputs unchanged
            identity = lambda x: x
            # constant function always returns the same thing
            constant = lambda x: val
            constant_0 = lambda x: 0.
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
