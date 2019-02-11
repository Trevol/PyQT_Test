import numpy as np
import cv2
import utils

max_rads = np.radians(33.26)
rads = [3.09814176, 0.30704573, 0.19362199, 0.74837805, 0.56672922, 0.456072, 3.14159263, 0.41741723, 0.52807445,
        0.74541948, 0.19658056, 0.35049663]


def np_where_fn():
    indexes = np.where(rads >= max_rads)[0]
    return indexes


def np_nonzero_fn():
    indexes = (rads >= max_rads).nonzero()[0]
    return indexes


def np_flatnonzero_fn():
    indexes = np.flatnonzero(rads >= max_rads)
    return indexes


def loop_fn():
    indexes = []
    for i, r in enumerate(rads):
        if r >= max_rads:
            indexes.append(i)
    return indexes


def main():
    iters = 1000000
    print(utils.timeit(np_where_fn, iters))
    print(utils.timeit(np_nonzero_fn, iters))
    print(utils.timeit(np_flatnonzero_fn, iters))
    print(utils.timeit(loop_fn, iters))


if __name__ == '__main__':
    main()
