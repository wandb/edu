test = {
    "name": "linear_approx",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function with right name
                    >>> callable(linear_approx)
                    True
                    >>> ## it should run on and return the right types
                    >>> isinstance(linear_approx(identity, 0., 0.), float)
                    True
                    >>> ## if epsilon is 0, should return f(input)
                    >>> np.isclose(linear_approx(identity, 0., 0.), 0.)
                    True
                    >>> np.isclose(linear_approx(constant, 0., 0.), constant(0.))
                    True
                    >>> np.isclose(linear_approx(np.square, 0., 0.), 0.)
                    True
                    >>> np.isclose(linear_approx(np.square, val, 0.), np.square(val))
                    True
                    >>> ## linear approximation of abs is line with slope +/-1
                    >>> np.isclose(linear_approx(np.abs, val, -val), 0.)
                    True
                    >>> ## linear approximation of square is 2 * x
                    >>> np.isclose(linear_approx(np.square, 1., -1.), -1.)
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
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
