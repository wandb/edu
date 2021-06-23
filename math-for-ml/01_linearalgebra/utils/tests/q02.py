test = {
    "name": "Shapes",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a dictionary called shapes
                    >>> type(shapes)
                    <class 'dict'>
                    >>> ## B is a row vector with two entries
                    >>> shapes["B"]
                    (1, 2)
                    >>> ## D is a column vector with two entries
                    >>> shapes["D"]
                    (2, 1)
                    >>> ## C is a matrix with two rows and two columns
                    >>> shapes["C"]
                    (2, 2)
                    >>> ## A is a vector with one entry
                    >>> ## Watch out! (1) == 1 !== (1,)
                    >>> shapes["A"]
                    (1,)
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
