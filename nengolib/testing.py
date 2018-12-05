from nengo.version import version_info

if version_info >= (2, 7, 0):
    from pytest import warns  # noqa: F401
else:  # pragma: no cover
    # https://github.com/nengo/nengo/pull/1381
    from nengo.utils.testing import warns  # noqa: F401
