"""Camera intrinsics for the depth-based geometry paths.

Every depth consumer in this app back-projects pixels with

    X = (x - cx) * Z / focal_px

so focal_px sets the absolute LATERAL scale of the reconstruction: a focal that
is off by a factor k makes every measured width off by 1/k. Until now all three
depth routes assumed a hardcoded 0.8 * max(W, H) — a 64 deg field of view,
roughly a 29 mm-equivalent lens. Room photos are very often shot much wider
than that (phone ultrawide is ~13 mm equiv -> 0.36 * max(W, H)), and at that
focal the old guess reports the room barely half its true width.

This module recovers the real focal length from the photo's EXIF when it is
there, and keeps the old constant as the fallback when it is not.

Focal is carried around as a RATIO (focal_px / max(W, H)) rather than in
pixels, because the pipeline resizes constantly (upscale to 4500 px, depth at
1280 px, payload cap at 3600 px). The ratio is invariant under any
aspect-preserving resize, so it is computed once per photograph and multiplied
by the current max(W, H) at each point of use — the same convention as the
WALL_FOCAL_RATIO constant it replaces.
"""

import math
import os
import re
import threading
from urllib.parse import urlparse

from PIL import Image

# Legacy assumption, kept as the fallback: 0.8 * long side ~ 64 deg FOV.
FALLBACK_FOCAL_RATIO = 0.8

# 35mm equivalent focal length is defined against the FRAME DIAGONAL (36x24mm),
# which is what phone vendors quote ("26 mm main camera"). Converting via the
# long side instead would be ~4% off on a 4:3 sensor.
FRAME_35MM_DIAGONAL_MM = 43.266615

# Plausible focal ratios for interior photography. 0.28 ~ 121 deg across the
# long side (wider than any consumer ultrawide), 2.5 ~ 23 deg (a short tele).
# Anything outside means the EXIF is lying or the image was re-framed.
MIN_FOCAL_RATIO = 0.28
MAX_FOCAL_RATIO = 2.50

# EXIF tags. Focal-plane tags appear under both the EXIF (0xA2xx) and the older
# TIFF/EP (0x92xx) numbering depending on the vendor, so both are tried.
_TAG_FOCAL_35MM = 0xA405
_TAG_FOCAL_LENGTH = 0x920A
_TAGS_FP_X_RES = (0xA20E, 0x920E)
_TAGS_FP_RES_UNIT = (0xA210, 0x9210)
_TAG_PIXEL_X = 0xA002
_TAG_PIXEL_Y = 0xA003

_IFD_EXIF = 0x8769

# FocalPlaneResolutionUnit -> mm per unit (2 = inch is the default per spec).
_RES_UNIT_MM = {1: 25.4, 2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}

# Extensions /api/upload may have written the room photo under.
_UPLOAD_EXTS = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Resolved intrinsics per photograph: {cache_key: (ratio, source)}. Only real
# measurements are stored — never the fallback, so a cold cache can still be
# beaten by a later on-disk EXIF read.
_focal_cache: dict[str, tuple] = {}
_focal_cache_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def focal_px(width, height, ratio=None):
    """focal_px at the CURRENT image size for a stored focal ratio."""
    r = ratio if ratio else FALLBACK_FOCAL_RATIO
    return float(r) * float(max(width, height))


def fov_deg(ratio):
    """Field of view in degrees across the image's LONG side."""
    r = float(ratio) if ratio else FALLBACK_FOCAL_RATIO
    return math.degrees(2.0 * math.atan(0.5 / r))


def focal_35mm_equiv(ratio, width, height):
    """Inverse of the EXIF conversion — for logging / response payloads."""
    long_side = float(max(width, height))
    diag_px = math.hypot(float(width), float(height))
    if diag_px <= 0:
        return None
    return float(ratio) * long_side / diag_px * FRAME_35MM_DIAGONAL_MM


def camera_info(ratio, source, width, height):
    """Serializable summary of the intrinsics used for a response."""
    f35 = focal_35mm_equiv(ratio, width, height)
    return {
        "focal_ratio": round(float(ratio), 5),
        "focal_px": round(focal_px(width, height, ratio), 1),
        "focal_35mm_equiv": round(f35, 1) if f35 else None,
        "fov_deg": round(fov_deg(ratio), 1),
        "source": source,
    }


def describe(ratio, source, width, height):
    """One-line human summary for the server log."""
    return (f"focal {focal_px(width, height, ratio):.0f}px "
            f"({float(ratio):.3f} x long side, {fov_deg(ratio):.0f}deg FOV, "
            f"~{focal_35mm_equiv(ratio, width, height):.0f}mm equiv) via {source}")


# --------------------------------------------------------------------------- #
# EXIF parsing
# --------------------------------------------------------------------------- #

def _num(value):
    """EXIF numerics arrive as int, IFDRational or a 1-tuple. Normalize."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        value = value[0]
    try:
        out = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) else None


def _flat_exif(pil_image):
    """Flatten IFD0 + the EXIF sub-IFD into one tag dict. Focal tags live in
    the sub-IFD, which Image.getexif() does not expose directly."""
    try:
        exif = pil_image.getexif()
    except Exception:
        return {}
    if not exif:
        return {}
    flat = dict(exif)
    try:
        flat.update(exif.get_ifd(_IFD_EXIF) or {})
    except Exception:
        pass
    return flat


def _first_tag(tags, tag_ids):
    for tag_id in tag_ids:
        value = _num(tags.get(tag_id))
        if value:
            return value
    return None


def _frame_intact(tags, width, height, tol=0.06):
    """Reject EXIF whose recorded frame no longer matches the image we hold.

    A crop keeps the true focal_px but shrinks the pixel dimensions, so scaling
    the EXIF focal by the cropped size would understate it. Compared long/short
    rather than W/H so an exif_transpose rotation does not read as a crop.
    """
    px_x = _num(tags.get(_TAG_PIXEL_X))
    px_y = _num(tags.get(_TAG_PIXEL_Y))
    if not px_x or not px_y:
        return True  # nothing recorded to contradict us
    exif_ar = max(px_x, px_y) / min(px_x, px_y)
    actual_ar = float(max(width, height)) / float(min(width, height))
    if actual_ar <= 0:
        return False
    return abs(exif_ar / actual_ar - 1.0) <= tol


def _validated(ratio, source):
    if ratio is None or not math.isfinite(ratio):
        return None, "unusable"
    if not (MIN_FOCAL_RATIO <= ratio <= MAX_FOCAL_RATIO):
        return None, f"{source}-implausible({ratio:.3f})"
    return float(ratio), source


def focal_ratio_from_exif(pil_image, width=None, height=None):
    """Focal ratio from a PIL image's EXIF, or (None, reason).

    Two routes, in order of reliability:
      1. FocalLengthIn35mmFilm — present on essentially every phone photo.
      2. FocalLength + FocalPlaneXResolution — the DSLR/mirrorless route,
         where focal_px = focal_mm * focal_plane_res / mm_per_unit reduces to a
         pure pixel quantity on the original sensor raster.
    """
    if pil_image is None:
        return None, "no-image"
    if width is None or height is None:
        width, height = pil_image.size
    if not width or not height:
        return None, "bad-dims"

    tags = _flat_exif(pil_image)
    if not tags:
        return None, "no-exif"
    if not _frame_intact(tags, width, height):
        return None, "exif-reframed"

    long_side = float(max(width, height))
    diag_px = math.hypot(float(width), float(height))

    f35 = _num(tags.get(_TAG_FOCAL_35MM))
    if f35 and f35 > 0:
        return _validated((f35 / FRAME_35MM_DIAGONAL_MM) * diag_px / long_side,
                          f"exif-35mm({f35:.0f}mm)")

    focal_mm = _num(tags.get(_TAG_FOCAL_LENGTH))
    fp_res = _first_tag(tags, _TAGS_FP_X_RES)
    px_x = _num(tags.get(_TAG_PIXEL_X))
    px_y = _num(tags.get(_TAG_PIXEL_Y))
    if focal_mm and fp_res and px_x and px_y:
        unit = int(_first_tag(tags, _TAGS_FP_RES_UNIT) or 2)
        mm_per_unit = _RES_UNIT_MM.get(unit, 25.4)
        # focal in pixels of the ORIGINAL raster; / its long side -> ratio.
        focal_px_orig = focal_mm * fp_res / mm_per_unit
        return _validated(focal_px_orig / max(px_x, px_y),
                          f"exif-sensor({focal_mm:.1f}mm)")

    return None, "exif-no-focal"


def focal_ratio_from_file(path, width=None, height=None):
    """Focal ratio read straight off a file on disk."""
    try:
        with Image.open(path) as img:
            w, h = (width, height) if width and height else img.size
            return focal_ratio_from_exif(img, w, h)
    except Exception as e:
        return None, f"file-unreadable({type(e).__name__})"


# --------------------------------------------------------------------------- #
# Per-photograph cache
# --------------------------------------------------------------------------- #

def room_id_from_url(url):
    """Every artefact this server writes carries the room's UUID in its
    filename (upload_<id>.jpg, final_<id>_<ts>.jpg, mask_<id>_<hotspot>.png),
    so any of them identifies the source photograph."""
    if not url:
        return None
    match = _UUID_RE.search(str(url))
    return match.group(0).lower() if match else None


def _keys(room_id=None, url=None):
    keys = []
    if room_id:
        keys.append(f"room:{str(room_id).lower()}")
    if url:
        keys.append(f"url:{url}")
        derived = room_id_from_url(url)
        if derived:
            key = f"room:{derived}"
            if key not in keys:
                keys.append(key)
    return keys


def remember(ratio, source, room_id=None, url=None):
    """Cache a MEASURED ratio under every key that identifies this photo."""
    if not ratio:
        return
    keys = _keys(room_id=room_id, url=url)
    if not keys:
        return
    with _focal_cache_lock:
        for key in keys:
            _focal_cache[key] = (float(ratio), source)


def recall(room_id=None, url=None):
    with _focal_cache_lock:
        for key in _keys(room_id=room_id, url=url):
            hit = _focal_cache.get(key)
            if hit:
                return hit
    return None, None


def forget(room_id=None, url=None):
    with _focal_cache_lock:
        for key in _keys(room_id=room_id, url=url):
            _focal_cache.pop(key, None)


def inherit(source_url, room_id=None, url=None):
    """Carry intrinsics from one canvas to a re-render of the same photograph
    (e.g. the AI-generated curtain room, whose output has no EXIF of its own)."""
    ratio, source = recall(url=source_url)
    if ratio:
        remember(ratio, f"{source}+inherited", room_id=room_id, url=url)
    return ratio, source


def register_from_pil(pil_image, room_id=None, url=None):
    """Extract-and-cache at ingest. Returns the camera info dict for the
    response; falls back to the legacy ratio without caching it, so a later
    on-disk read can still supply the real value."""
    width, height = pil_image.size
    ratio, source = focal_ratio_from_exif(pil_image, width, height)
    if ratio:
        remember(ratio, source, room_id=room_id, url=url)
    else:
        ratio, source = FALLBACK_FOCAL_RATIO, f"fallback({source})"
    return camera_info(ratio, source, width, height)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def _from_payload(payload, width, height):
    """Caller-supplied intrinsics, validated like any other source."""
    if not isinstance(payload, dict):
        return None, None

    nested = payload.get("camera")
    if isinstance(nested, dict):
        ratio, source = _from_payload(nested, width, height)
        if ratio:
            return ratio, source

    ratio = _num(payload.get("focal_ratio"))
    if ratio:
        return _validated(ratio, "request-ratio")

    px = _num(payload.get("focal_px"))
    if px:
        return _validated(px / float(max(width, height)), "request-px")

    f35 = _num(payload.get("focal_35mm") or payload.get("focal_35mm_equiv"))
    if f35:
        diag_px = math.hypot(float(width), float(height))
        return _validated((f35 / FRAME_35MM_DIAGONAL_MM) * diag_px / max(width, height),
                          f"request-35mm({f35:.0f}mm)")

    return None, None


def _local_names(room_id, url):
    """Filenames on disk that could hold this photo's EXIF. basename only —
    never join a caller-supplied path segment."""
    names = []
    if url:
        name = os.path.basename(urlparse(str(url)).path)
        if name:
            names.append(name)
    rid = room_id or room_id_from_url(url)
    if rid:
        names.extend(f"upload_{str(rid).lower()}.{ext}" for ext in _UPLOAD_EXTS)
    return names


def resolve_focal_ratio(width, height, payload=None, room_id=None, url=None,
                        local_dirs=(), probe_files=()):
    """Best available focal ratio for an image, as (ratio, source).

    Priority: explicit request value -> cached measurement for this photo ->
    EXIF re-read from our own upload on disk -> EXIF from any extra file the
    caller names (the download cache, for a room hosted elsewhere) -> the legacy
    0.8 * max(W, H) guess. The two disk routes matter because the in-memory
    cache does not survive a process restart.
    """
    ratio, source = _from_payload(payload, width, height)
    if ratio:
        return ratio, source

    ratio, source = recall(room_id=room_id, url=url)
    if ratio:
        return ratio, f"cached:{source}"

    for name in _local_names(room_id, url):
        for folder in local_dirs:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            ratio, source = focal_ratio_from_file(path, width, height)
            if ratio:
                remember(ratio, source, room_id=room_id, url=url)
                return ratio, f"disk:{source}"
            break  # file found but unusable — no point trying other folders

    for path in probe_files:
        if not path or not os.path.isfile(path):
            continue
        ratio, source = focal_ratio_from_file(path, width, height)
        if ratio:
            remember(ratio, source, room_id=room_id, url=url)
            return ratio, f"probe:{source}"

    return FALLBACK_FOCAL_RATIO, "fallback(default)"
