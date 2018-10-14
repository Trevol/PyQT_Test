from threading import Timer
import colorsys
import math


def int_to_str(val):
    return f'{val}'


def debounce(wait):
    """ Decorator that will postpone a functions
        execution until after wait seconds
        have elapsed since the last time it was invoked. """

    def decorator(fn):
        def debounced(*args, **kwargs):
            def call_it():
                fn(*args, **kwargs)

            try:
                debounced.t.cancel()
            except(AttributeError):
                pass
            debounced.t = Timer(wait, call_it)
            debounced.t.start()

        return debounced

    return decorator


def make_n_colors(N, bright=True):
    """
    Generate random colors. To get visually distinct colors, generate them in HSV space then convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    return [
        [int(a * 255) for a in colorsys.hsv_to_rgb(i / N, 1, brightness)]
        for i in range(N)]


def rotate_around_point(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.

    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy
