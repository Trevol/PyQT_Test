from enaml.application import deferred_call


def deferred(fn):
    def wrapped(*args, **kwargs):
        deferred_call(fn, *args, **kwargs)

    return wrapped
