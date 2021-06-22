test = {
    "name": "is_little_o_x2",
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
                    >>> ## the is_little_o_x2 dictionary should be defined
                    >>> isinstance(is_little_o_x2, dict)
                    True
                    >>> ## o(x) is like a "stricly less than" symbol
                    >>> is_little_o_x2[x ** 2]
                    False
                    >>> ## x^n is o(x^n+1), not the other way around
                    >>> is_little_o_x2[1e6 * x ** 3]
                    True
                    >>> ## x^n is o(x^n+1), not the other way around
                    >>> is_little_o_x2[x]
                    False
                    >>> ## multiplying by a constant doesn't change anything
                    >>> is_little_o_x2[1/1e6 * x]
                    False
                    >>> is_little_o_x2[5 * x]
                    False
                    >>> ## the exponential function is not o(x^n) for any n
                    >>> is_little_o_x2[sympy.exp(x)]
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
                    >>> ## the is_little_o_x2 dictionary should be defined
                    >>> isinstance(is_little_o_x2, dict)
                    True
                    >>> ## the outputs should match the answers in the dictionary
                    >>> items = is_little_o_x2.items()
                    >>> all(is_little_o(x ** 2, key, x) == val for key, val in items)
                    True
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
