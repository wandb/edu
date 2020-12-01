test = {
    "name": "Dimensions",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a dictionary called dimensions
                    >>> type(dimensions)
                    <class 'dict'>
                    >>> ## A is a vector, neither row nor column
                    >>> dimensions["A"]
                    1
                    >>> ## B and D are both vectors
                    >>> dimensions["B"] == dimensions["D"]
                    True
                    >>> ## B/D are explicitly row/column vectors
                    >>> ## so they have two dimensions
                    >>> dimensions["B"]
                    2
                    >>> ## C is a matrix, so it has two dimensions
                    >>> dimensions["C"]
                    2
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
