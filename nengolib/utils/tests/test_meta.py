from nengo.utils.compat import with_metaclass

from nengolib.utils.meta import ReuseUnderlying


class Example(with_metaclass(ReuseUnderlying)):

    def __init__(self, cls):
        self._cls = cls


def test_reuse_underlying():

    obj1 = Example(1)
    obj2 = Example(obj1)
    obj3 = Example(cls=obj2)
    obj4 = Example(1)

    assert obj1 is obj2
    assert obj2 is obj3
    assert obj3 is not obj4
