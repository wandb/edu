test = {
    "name": "apply_to_batch",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function called apply_to_batch
                    >>> callable(apply_to_batch)
                    True
                    >>> ## it should run when applied to compatible inputs
                    >>> out = apply_to_batch(identity, vectors)
                    >>> ## the return is of type array
                    >>> type(out)
                    <class 'numpy.ndarray'>
                    """,
                    "hidden": False,
                    "locked": False
                },
                {
                    "code": r"""
                    >>> # applying the identity matrix shouldn't change inputs
                    >>> np.allclose(vectors, apply_to_batch(identity, vectors))
                    True
                    >>> # the result should be the same as normal matrix multiplication
                    >>> np.allclose(random_matrix @ vectors, apply_to_batch(random_matrix, vectors))
                    True
                    >>> # return_first should pull out the first entry in each vector
                    >>> np.allclose(vectors[0], apply_to_batch(return_first, vectors))
                    True
                    """
                }
            ],
            "setup": r"""
            shape, count = 5, 10
            identity = np.eye(shape)
            return_first = np.array([[1.] + [0.] * (shape - 1)])
            random_matrix = np.random.standard_normal((shape, shape))
            vectors = np.random.standard_normal((shape, count))
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
