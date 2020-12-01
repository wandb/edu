test = {
    "name": "Dot Product",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## you must define a function called dot
                    >>> callable(dot)
                    True
                    >>> ## all-zeros vector should always return 0
                    >>> int(dot(all_zeros, all_zeros))
                    0
                    >>> int(dot(all_zeros, random_vector))
                    0
                    >>> ## dot product with all-ones vector is sum
                    >>> ## sum of numbers up to 5 is 15
                    >>> int(np.round(dot(range_vector, all_ones)))
                    15
                    >>> ## sum needs to also work on a random vector
                    >>> np.isclose(sum(random_vector), dot(random_vector, all_ones))
                    True
                    >>> ## and we should get same answer as numpy in general
                    >>> np.isclose(np.dot(random_vector, second_random_vector), dot(random_vector, second_random_vector))
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            shape = 5
            range_vector = np.array([1, 2, 3, 4, 5])
            all_zeros = np.zeros(shape)
            all_ones = np.ones(shape)
            random_vector = np.random.randn(shape)
            second_random_vector = np.random.randn(shape)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
