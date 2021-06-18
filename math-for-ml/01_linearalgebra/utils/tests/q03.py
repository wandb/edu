test = {
    "name": "Shape of Transpose",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function with right name
                    >>> callable(shape_of_transpose)
                    True
                    >>> ## shape is reversed shape of input matrix
                    >>> shape_of_transpose(random_matrix) == transposed_shape
                    array([ True,  True])
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            random_shape = np.random.randint(30, size=2) + 1
            print(f"Testing on matrix with shape {random_shape}")
            random_matrix = np.random.standard_normal(size=random_shape)
            transposed_shape = random_shape[::-1]
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
