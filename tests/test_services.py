import doctest


def test_doctests():
    import dimagi_clockify_cli.services

    results = doctest.testmod(dimagi_clockify_cli.services)
    assert results.failed == 0
