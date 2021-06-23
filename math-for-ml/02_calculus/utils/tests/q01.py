test = {
    "name": "is_little_o_x",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## the symbol x should be defined
                    >>> isinstance(x, sympy.Symbol)
                    True
                    >>> ## the is_little_o_x dictionary should be defined
                    >>> isinstance(is_little_o_x, dict)
                    True
                    >>> ## o(x) is like a "stricly less than" symbol
                    >>> is_little_o_x[x]
                    False
                    >>> ## multiplying by a constant doesn't change anything
                    >>> is_little_o_x[1/1e6 * x]
                    False
                    >>> is_little_o_x[5 * x]
                    False
                    >>> ## close to 0, x is smaller than x ** 2
                    >>> is_little_o_x[x ** 2]
                    True
                    >>> ## the exponential function is not o(x^n) for any n
                    >>> is_little_o_x[sympy.exp(x)]
                    False
                    """,
                    "hidden": False,
                    "locked": False
                },
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## the symbol x should be defined
                    >>> isinstance(x, sympy.Symbol)
                    True
                    >>> ## the is_little_o_x dictionary should be defined
                    >>> isinstance(is_little_o_x, dict)
                    True
                    >>> ## the outputs should match the answers in the dictionary
                    >>> items = is_little_o_x.items()
                    >>> all(is_little_o(x, key, x) == val for key, val in items)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                },
            ],
            "setup": r"""
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
