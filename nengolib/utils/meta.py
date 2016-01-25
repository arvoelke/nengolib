__all__ = ['ReuseUnderlying']


class ReuseUnderlying(type):
    """Metaclass for returning first argument if an instance of the class."""

    def __call__(self, cls, *args, **kwargs):
        if isinstance(cls, self):
            return cls
        return super(ReuseUnderlying, self).__call__(cls, *args, **kwargs)
