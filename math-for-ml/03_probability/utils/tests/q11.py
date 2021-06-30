test = {
    "name": "run_gd",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # TESTS BEGIN HERE
                    >>> ## must define functions with the right names
                    >>> callable(run_gd)
                    True
                    >>> callable(gaussian_surprise)
                    True
                    >>> callable(sum_squared_error)
                    True
                    >>> ## the run_gd function should take and return the right types
                    >>> isinstance(run_gd(mu_0, rand_array, sum_squared_error), float)
                    True
                    >>> ## if the learning rate is 0., the output should be mu_0
                    >>> np.isclose(run_gd(mu_0, rand_array, sum_squared_error, 0.), mu_0)
                    True
                    >>> ## with a reasonable learning rate, the squared error should go down
                    >>> sse_mu = run_gd(mu_0, rand_array, sum_squared_error)
                    >>> sse_result = sum_squared_error(sse_mu, rand_array)
                    >>> np.greater(sum_squared_error(mu_0, rand_array), sse_result)
                    True
                    >>> ## but not reach the mean, which minimizes the squared error
                    >>> np.less_equal(sum_squared_error(mean, rand_array), sse_result)
                    True
                    >>> ## if we double the learning rate,
                    >>> ## the result of minimizing gaussian_surprise should be the same
                    >>> gaussian_surprise_mu = run_gd(mu_0, rand_array, gaussian_surprise, learning_rate=0.2)
                    >>> np.isclose(sse_mu, gaussian_surprise_mu)
                    True
                    """,
                    "hidden": False,
                    "locked": False
                }
            ],
            "setup": r"""
            >>> rand_array = np.random.randn(5)
            >>> mu_0 = np.random.randn(1)[0]
            >>> mean = np.mean(rand_array)
            """,
            "teardown": r"""
            """,
            "type": "doctest"}]
        }
