"""OpenCV version-compatibility shims."""

import numpy as np

_EMPTY_SEGMENTS = np.empty((0, 4), dtype=np.int32)


def as_segments(lines):
    """Normalize a cv2.HoughLinesP() return value to an (N, 4) array of
    x1, y1, x2, y2 — iterate it directly, no None check needed.

    OpenCV's Python binding returns (N, 1, 4) up to 4.11 but (N, 4) in newer
    builds. The old `for x1, y1, x2, y2 in lines[:, 0]` idiom silently changes
    meaning between the two: on the newer shape it iterates int32 SCALARS and
    raises "cannot unpack non-iterable numpy.int32 object". Reshaping to (-1, 4)
    is correct for both layouts.
    """
    if lines is None:
        return _EMPTY_SEGMENTS
    arr = np.asarray(lines)
    if arr.size == 0 or arr.size % 4 != 0:
        return _EMPTY_SEGMENTS
    return arr.reshape(-1, 4)
