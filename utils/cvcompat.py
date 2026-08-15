"""OpenCV version-compatibility shims.

Several OpenCV Python functions return an array with a redundant middle axis —
(N, 1, K) — up to OpenCV 4.11, but a squeezed (N, K) in newer builds. Code
written against the old layout (`arr[:, 0]`, `arr[:, 0, 1]`) does not fail
loudly on the new one; it silently indexes the wrong axis and raises further
down with a confusing message:

    TypeError: cannot unpack non-iterable numpy.int32 object
    IndexError: too many indices for array: array is 2-dimensional...

Reshaping to (-1, K) is correct for BOTH layouts.
"""

import numpy as np

_EMPTY_SEGMENTS = np.empty((0, 4), dtype=np.int32)
_EMPTY_POINTS = np.empty((0, 2), dtype=np.int32)


def as_segments(lines):
    """cv2.HoughLinesP() -> (N, 4) array of x1, y1, x2, y2.

    Iterate it directly; None becomes an empty array, so no None check needed.
    """
    if lines is None:
        return _EMPTY_SEGMENTS
    arr = np.asarray(lines)
    if arr.size == 0 or arr.size % 4 != 0:
        return _EMPTY_SEGMENTS
    return arr.reshape(-1, 4)


def as_points(points):
    """cv2.findNonZero() -> (N, 2) array of x, y.

    None (an all-black mask) becomes an empty array.
    """
    if points is None:
        return _EMPTY_POINTS
    arr = np.asarray(points)
    if arr.size == 0 or arr.size % 2 != 0:
        return _EMPTY_POINTS
    return arr.reshape(-1, 2)
