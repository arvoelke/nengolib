
from nengolib.utils.meta import ReuseUnderlying


class Example(object):
    __metaclass__ = ReuseUnderlying

    def __init__(self, cls):
        self._cls = cls


def test_reuse_underlying():

    obj1 = Example(1)
    obj2 = Example(obj1)
    obj3 = Example(obj2)
    obj4 = Example(1)

    assert obj1 is obj2
    assert obj2 is obj3
    assert obj3 is not obj4
