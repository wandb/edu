test = {
    "name": "is_orthonormal",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function called normalize
                    >>> callable(is_orthonormal)
                    True
                    >>> ## it should take and return a single value
                    >>> out = is_orthonormal(np.eye(2))
                    >>> ## the return is of type bool
                    >>> type(out)
                    <class 'bool'>
                    """,
                    "hidden": False,
                    "locked": False
                },
                {
                    "code": r"""
                    >>> # the identity matrix is orthonormal
                    >>> is_orthonormal(np.eye(20))
                    True
                    >>> # a scaled version of the identity is not
                    >>> is_orthonormal(2 *  np.eye(20))
                    False
                    >>> # permuting the rows preserves orthonormality
                    >>> is_orthonormal(np.random.permutation(np.eye(20)))
                    True
                    """
                },
                {
                    "code": r"""
                    >>> # the U and V components of SVD are orthogonal
                    >>> is_orthonormal(U)
                    True
                    >>> is_orthonormal(V_T.T)
                    True
                    >>> # but the singular value matrix is generically not
                    >>> is_orthonormal(S)
                    False
                    >>> # a random gaussian matrix is not quite orthonormal
                    >>> is_orthonormal(random_values)
                    False
                    """
                }
            ],
            "setup": r"""
            shape = 20
            square_matrix = la.random_matrix.SymmetricWigner(shape).M
            U, S, V_T = la.svd.compact(square_matrix)
            random_values =  np.random.standard_normal((shape, shape))
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
