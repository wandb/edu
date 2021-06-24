test = {
    "name": "my_pdf",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define a function with the right name
                    >>> callable(my_pdf)
                    True
                    >>> ## that function should take and return the right types
                    >>> isinstance(my_pdf(0.5), float)
                    True
                    >>> ## a pdf never takes on negative values
                    >>> all(my_pdf(x) >= 0 for x in test_values)
                    True
                    >>> ## a pdf integrates to 1
                    >>> integrates_to_one(my_pdf)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> test_values = np.random.rand(100)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
