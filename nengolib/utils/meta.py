__all__ = ['ReuseUnderlying']


class ReuseUnderlying(type):
    """Metaclass for returning first argument if an instance of the class."""

    def __call__(self, *args, **kwargs):
        vals = tuple(args) + tuple(kwargs.values())
        if len(vals) == 1 and isinstance(vals[0], self):
            return vals[0]
        return super(ReuseUnderlying, self).__call__(*args, **kwargs)
