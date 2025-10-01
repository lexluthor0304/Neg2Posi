import numpy as np
import cv2
import exifread
# ------------------------------------------------------------------
# macOS stability: disable OpenCL/Metal backend & limit OpenCV threads
# ------------------------------------------------------------------
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass
cv2.setNumThreads(2)  # reduce parallelism to avoid race conditions
from pathlib import Path
from PIL import Image

# If RAW support is needed, please run: pip install rawpy
try:
    import rawpy
    RAW_SUPPORTED = True
except ImportError:
    RAW_SUPPORTED = False

# Flat-field correction using RAW metadata shading map
def apply_raw_flat_field(path: str, img: np.ndarray) -> np.ndarray:
    """
    Apply flat-field correction using RAW metadata shading map if available.
    """
    try:
        raw = rawpy.imread(str(path))
        # shading_table returns a small calibration map per channel
        shading = raw.shading_table()  # shape: (Hs, Ws, 4) in RGGB pattern
        # Merge RGGB channels into 3-channel multiplier map:
        # Interpolate shading map to image size
        h, w = img.shape[:2]
        # Convert 4-channel table to 3-channel: average appropriate channels
        # For simplicity, average RGRG for red, GBGB for green, BRBR for blue
        red_sh = (shading[:,:,0] + shading[:,:,2]) * 0.5
        green_sh = (shading[:,:,1] + shading[:,:,3]) * 0.5
        blue_sh = (shading[:,:,1] + shading[:,:,3]) * 0.5
        sh_map = np.stack([
            cv2.resize(red_sh,   (w,h), interpolation=cv2.INTER_LINEAR),
            cv2.resize(green_sh, (w,h), interpolation=cv2.INTER_LINEAR),
            cv2.resize(blue_sh,  (w,h), interpolation=cv2.INTER_LINEAR),
        ], axis=2)
        # Normalize map to mean=1
        sh_map /= np.mean(sh_map, axis=(0,1), keepdims=True) + 1e-8
        return np.clip(img / (sh_map + 1e-8), 0.0, 1.0)
    except Exception:
        # If raw shading not available, return original
        return img

import os
import lensfunpy

def apply_lensfun_correction(path: str, img: np.ndarray) -> np.ndarray:
    """
    Apply lens shading (vignetting) and distortion correction using lensfunpy.
    """
    # Read EXIF metadata via exifread for lensfun
    camera_maker = camera_model = lens_model = ""
    focal_length = aperture = 0.0
    distance = 0.0  # assume infinity
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        camera_maker = str(tags.get('Image Make', "")) 
        camera_model = str(tags.get('Image Model', ""))
        lens_model = str(tags.get('EXIF LensModel', "")) or str(tags.get('Image LensModel', ""))
        # FocalLength tag
        fl_tag = tags.get('EXIF FocalLength')
        if hasattr(fl_tag, 'values') and len(fl_tag.values) == 1:
            num, den = fl_tag.values[0].num, fl_tag.values[0].den
            if den:
                focal_length = num / den
        # FNumber tag
        fn_tag = tags.get('EXIF FNumber')
        if hasattr(fn_tag, 'values') and len(fn_tag.values) == 1:
            num, den = fn_tag.values[0].num, fn_tag.values[0].den
            if den:
                aperture = num / den
    except Exception as e_exif:
        logging.warning(f"EXIF read failed for lensfun metadata: {e_exif}")
    logging.info(f"EXIF metadata: Make='{camera_maker}', Model='{camera_model}', Lens='{lens_model}', FocalLength={focal_length}, FNumber={aperture}")
    # Initialize the lensfun database and lookup camera & lens
    db = lensfunpy.Database()
    cams = db.find_cameras(camera_maker, camera_model)
    if not cams:
        logging.warning(f"No lensfun camera profile found for {camera_maker} {camera_model}")
        return img
    cam = cams[0]
    # Search by maker and lens name explicitly
    lenses = db.find_lenses(cam, maker=camera_maker, lens=lens_model, loose_search=False)
    if not lenses:
        # Fallback: list all lenses for this camera maker
        all_lenses = db.find_lenses(cam, maker=camera_maker)
        matched = [l for l in all_lenses if lens_model.lower() in getattr(l, 'model', '').lower()]
        if matched:
            lens = matched[0]
            logging.info(f"Using fallback lensfun profile model '{lens.model}' for EXIF lens '{lens_model}'")
        else:
            available = [getattr(l, 'model', '') for l in all_lenses]
            logging.warning(f"No lensfun lens profile found for '{lens_model}'. Available models: {available}")
            return img
    else:
        lens = lenses[0]
    corrector = lensfunpy.Corrector(db, cam, lens, focal_length, aperture, distance)
    # lensfun expects a byte image in linear space; convert float [0,1] to uint16
    h, w = img.shape[:2]
    img_u16 = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)
    # Apply color (vignetting) correction
    img_shaded = corrector.apply_color(img_u16)
    # Convert back to float32 [0,1]
    img_corr = img_shaded.astype(np.float32) / 65535.0
    return img_corr
import logging # Added for consistency, though not heavily used in original GUI feedback

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Constants for film type radio buttons keys (internal, non-localized)
FILM_TYPE_COLOR_INTERNAL_KEY = "color"
FILM_TYPE_BW_INTERNAL_KEY = "bw"



# Global storage for manual crop overrides
user_crops = {}
# Global storage for manual rotation overrides (degrees)
user_angles = {}
def load_image(path):
    # Coerce path from nested tuple/list to a string path
    while isinstance(path, (tuple, list)):
        if not path:
            raise ValueError("Empty path tuple/list received.")
        path = path[0]
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if RAW_SUPPORTED and p.suffix.lower() in ('.cr2', '.nef', '.arw', '.raf', '.dng'):
        raw = rawpy.imread(str(p))
        rgb = raw.postprocess(no_auto_bright=True, gamma=(1,1), output_bps=16)
        return rgb.astype(np.float32) / 65535.0
    bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if bgr is not None:
        # Handle images with alpha channel by converting to RGB
        if bgr.ndim == 3 and bgr.shape[2] == 4:
            return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB).astype(np.float32) / 255.0
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    with Image.open(str(p)) as im:
        im = im.convert('RGB')
        arr = np.array(im).astype(np.float32) / 255.0
        return arr

def save_image(rgb, path):
    out = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_obj.with_suffix(".png")
    success = cv2.imwrite(str(png_path), bgr)
    if not success:
        logging.warning(f"cv2.imwrite failed for {png_path}, attempting PIL fallback.")
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(bgr).save(str(png_path))
            logging.info(f"PIL fallback save succeeded: {png_path}")
        except Exception as e_fallback:
            raise IOError(f"Failed to write image PNG to {png_path} via cv2 and PIL fallback: {e_fallback}")
    logging.info(f"Saved: {png_path}")


def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)] # Smallest x-y
    bl = pts[np.argmax(diff)] # Largest x-y
    return np.array([tl, tr, br, bl], dtype='float32')


# REMOVED: _is_probably_bw_image function

def _clahe_enhance(gray_u8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray_u8)

def detect_film_region_bw(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    inv  = cv2.bitwise_not(gray)
    proc = _clahe_enhance(inv)
    _, thresh = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_close = max(15, int(min(img.shape[:2]) * 0.06))
    k_close |= 1
    k_open = max(7, k_close // 2)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        thresh_fallback = cv2.adaptiveThreshold(
            proc, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -15
        )
        closed_fb = cv2.morphologyEx(thresh_fallback, cv2.MORPH_CLOSE, kernel_close)
        opened_fb = cv2.morphologyEx(closed_fb, cv2.MORPH_OPEN, kernel_open)
        cnts, _ = cv2.findContours(opened_fb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < (img.shape[0] * img.shape[1] * 0.05):
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(c)
        box  = cv2.boxPoints(rect)
        pts  = np.array(box, dtype=np.float32)
    pts = order_points(pts)
    return pts if _is_valid_region(pts, img.shape) else None


def _is_valid_region(pts, img_shape, min_area_ratio=0.15):
    h_img, w_img = img_shape[:2]
    if pts is None or len(pts) != 4: return False # Added check for None pts
    tl, tr, br, bl = pts
    # Check for degenerate quadrilaterals (e.g. all points collinear)
    if cv2.contourArea(pts.astype(np.int32)) < 1: return False

    width  = np.linalg.norm(tr - tl)
    height = np.linalg.norm(bl - tl)
    area   = width * height
    if area < (min_area_ratio * w_img * h_img):
        return False
    margin = 1.0
    for x, y in pts:
        if (x < -margin) or (y < -margin) or (x > w_img + margin) or (y > h_img + margin):
            return False
    return True

def detect_film_region_refined(img):
    gray_u8 = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_u8, 50, 150, apertureSize=3)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=70,
        minLineLength=min(img.shape[:2])//8, maxLineGap=30
    )
    if lines is None: return None
    horiz, vert = [], []
    for line in lines: # Iterate directly over lines
        x1,y1,x2,y2 = line[0]
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if ang < 10: horiz.append((y1 + y2) * 0.5)
        elif abs(ang - 90) < 10: vert.append((x1 + x2) * 0.5)
    if len(horiz) < 2 or len(vert) < 2: return None
    top, bottom = min(horiz), max(horiz)
    left, right = min(vert),  max(vert)
    pts = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    corners = pts.reshape(-1,1,2)
    cv2.cornerSubPix(gray_u8, corners, (11,11), (-1,-1),
                     (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    refined = corners.reshape(-1, 2)
    refined = order_points(refined)
    return refined if _is_valid_region(refined, img.shape) else None

def detect_film_region_proj(img, smooth: int = 15, thresh_ratio: float = 0.15):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    inv  = cv2.bitwise_not(gray)
    if smooth > 0: inv = cv2.GaussianBlur(inv, (0, 0), smooth)
    row_mean = inv.mean(axis=1)
    col_mean = inv.mean(axis=0)
    r_min, r_max = row_mean.min(), row_mean.max()
    c_min, c_max = col_mean.min(), col_mean.max()
    # Add epsilon to prevent division by zero if r_max == r_min
    r_thresh = r_min + (r_max - r_min + 1e-6) * thresh_ratio
    c_thresh = c_min + (c_max - c_min + 1e-6) * thresh_ratio
    rows = np.where(row_mean > r_thresh)[0]
    cols = np.where(col_mean > c_thresh)[0]
    if len(rows) < 2 or len(cols) < 2: return None
    top, bottom = rows[0],  rows[-1]
    left, right = cols[0],  cols[-1]
    tl = np.array([left, top], dtype=np.float32)
    tr = np.array([right, top], dtype=np.float32)
    br = np.array([right, bottom], dtype=np.float32)
    bl = np.array([left, bottom], dtype=np.float32)
    pts = order_points(np.stack([tl, tr, br, bl], axis=0))
    return pts if _is_valid_region(pts, img.shape, min_area_ratio=0.1) else None

def detect_film_region_hough(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=80,
        minLineLength=min(img.shape[:2])//4, maxLineGap=30
    )
    if lines is None: return None
    horiz, vert = [], []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if ang < 15: horiz.append((y1 + y2) * 0.5)
        elif abs(ang - 90) < 15: vert.append((x1 + x2) * 0.5)
    if len(horiz) < 2 or len(vert) < 2: return None
    top, bottom = min(horiz), max(horiz)
    left, right = min(vert), max(vert)
    pts = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    pts = order_points(pts)
    return pts if _is_valid_region(pts, img.shape, min_area_ratio=0.1) else None

def detect_film_region_simple(img):
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4,2).astype(np.float32) # Ensure float32
    else:
        x,y,w,h = cv2.boundingRect(c)
        pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
    pts = order_points(pts)
    return pts if _is_valid_region(pts, img.shape) else None

def detect_film_region_color(img, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < (img.shape[0] * img.shape[1] * 0.1): return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        pts = np.array(box, dtype=np.float32)
    pts = order_points(pts)
    return pts if _is_valid_region(pts, img.shape) else None


# Helper function for initial crop/angle settings
def _determine_initial_crop_settings(path: str, film_type_key: str, original_img: np.ndarray, user_crop_pts=None, user_angle: float = 0.0):
    """
    Determine initial crop margins and angle based on user overrides or auto-detection.
    Returns (settings_dict, detected_pts).
    """
    # If user override exists, use it
    if user_crop_pts and isinstance(user_crop_pts, (list, tuple)) and len(user_crop_pts) == 4:
        pts = order_points(np.array(user_crop_pts, dtype=np.float32))
        # Compute margins relative to original image dimensions
        h, w = original_img.shape[:2]
        tl, tr, br, bl = pts
        left = int(max(0.0, tl[0]))
        top = int(max(0.0, tl[1]))
        right = int(max(0.0, w - 1 - tr[0]))
        bottom = int(max(0.0, h - 1 - bl[1]))
        return ({"left": left, "right": right, "top": top, "bottom": bottom, "angle": user_angle}, pts)

    # No user override: attempt auto-detection
    pts = None
    if film_type_key == FILM_TYPE_BW_INTERNAL_KEY:
        detectors = (detect_film_region_proj, detect_film_region_refined, detect_film_region_bw, detect_film_region_hough, detect_film_region_simple)
    else:
        LOWER_ORANGE = np.array([2,50,50]); UPPER_ORANGE = np.array([30,255,255])
        detectors = (
            lambda im: detect_film_region_color(im, LOWER_ORANGE, UPPER_ORANGE),
            detect_film_region_hough,
            detect_film_region_proj,
            detect_film_region_refined,
            detect_film_region_simple
        )
    for det in detectors:
        if pts is None:
            pts = det(original_img)

    # Fallback to full image if no region detected
    if pts is None:
        h, w = original_img.shape[:2]
        pts = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
        angle = 0.0
    else:
        # Estimate rotation angle from top edge
        tl, tr, _, _ = order_points(pts)
        angle = float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
        # Zero small angles
        if abs(angle) < 1.0:
            angle = 0.0

    # Compute margins based on pts
    pts = order_points(pts)
    tl, tr, br, bl = pts
    h, w = original_img.shape[:2]
    left = int(max(0.0, tl[0]))
    top = int(max(0.0, tl[1]))
    right = int(max(0.0, w - 1 - tr[0]))
    bottom = int(max(0.0, h - 1 - bl[1]))
    return ({"left": left, "right": right, "top": top, "bottom": bottom, "angle": angle}, pts)

def crop_and_warp(img, src_pts):
    if src_pts is None or len(src_pts) != 4 : return img # Return original if no valid points
    tl, tr, br, bl = src_pts
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    maxW    = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH    = int(max(heightA, heightB))

    if maxW <= 0 or maxH <=0 : return img # Prevent errors with zero dimensions

    dst = np.array([[0,0], [maxW-1,0], [maxW-1,maxH-1], [0,maxH-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(src_pts, dst)
    # Ensure image is uint8 for warpPerspective if it came in as float
    img_u8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
    warped = cv2.warpPerspective(img_u8, M, (maxW, maxH))
    return warped.astype(np.float32) / 255.0


def rotate_and_crop_bw(img: np.ndarray) -> np.ndarray:
    """
    Dedicated rotation and crop for B&W negatives using the color logic.
    """
    # Delegate to color cropping logic for robust detection
    return rotate_and_crop_color(img)


def rotate_and_crop_color(img: np.ndarray) -> np.ndarray:
    """
    Robust color negative crop & rotate using iterative detection and rotation.
    """
    LOWER_ORANGE = np.array([2,50,50])
    UPPER_ORANGE = np.array([30,255,255])
    pts = None
    # Initial detection sequence for color
    for det in (
        lambda im: detect_film_region_color(im, LOWER_ORANGE, UPPER_ORANGE),
        detect_film_region_refined,
        detect_film_region_simple,
        detect_film_region_bw
    ):
        if pts is None:
            pts = det(img)
    # Rotate image based on detected top edge angle
    if pts is not None:
        tl, tr, br, bl = pts
        angle = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
        if abs(angle) > 1.0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
            img_rot = cv2.warpAffine(img, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0,0,0))
            # Re-detect on rotated image
            new_pts = None
            for det in (
                lambda im: detect_film_region_color(im, LOWER_ORANGE, UPPER_ORANGE),
                detect_film_region_refined,
                detect_film_region_simple
            ):
                if new_pts is None:
                    new_pts = det(img_rot)
            if new_pts is not None:
                img = img_rot
                pts = new_pts
    # Final crop using updated image and points
    return crop_and_warp(img, pts) if pts is not None else img


def process_bw_pipeline(img: np.ndarray,
                        low_pct: float = 1.0,
                        high_pct: float = 99.0) -> np.ndarray:
    """
    Re‑designed B&W negative pipeline (v2):
    1. Optional geometric correction via rotate_and_crop_bw.
    2. Inversion.
    3. Convert to single‑channel luminance (ITU‑R BT.601 weighting).
    4. Contrast stretch using percentiles.
    5. Local contrast boost with CLAHE.
    6. Replicate to 3‑channel RGB for downstream consistency.
    """
    # ---------- 1.  geometric cleanup ----------
    img = rotate_and_crop_bw(img)            # falls back to original if detection fails
    if img.size == 0 or img.shape[0] < 4 or img.shape[1] < 4:
        # Return tiny placeholder to avoid downstream errors
        return np.zeros((120, 120, 3), dtype=np.float32)

    # ---------- 2.  invert negative ----------
    img_inv = 1.0 - img

    # ---------- 3.  luminance conversion ----------
    if img_inv.ndim == 3 and img_inv.shape[2] >= 3:
        gray = 0.299 * img_inv[:, :, 0] + 0.587 * img_inv[:, :, 1] + 0.114 * img_inv[:, :, 2]
    else:       # already single channel
        gray = img_inv if img_inv.ndim == 2 else img_inv[..., 0]

    # ---------- 4.  global contrast stretch ----------
    p_low  = np.percentile(gray, low_pct)
    p_high = np.percentile(gray, high_pct)
    denom  = max(p_high - p_low, 1e-6)
    gray_cs = np.clip((gray - p_low) / denom, 0.0, 1.0)

    # ---------- 5.  local contrast enhancement (CLAHE) ----------
    gray_u8 = (gray_cs * 255).astype(np.uint8)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_u8).astype(np.float32) / 255.0

    # ---------- 6.  replicate to 3‑channel ----------
    out = np.stack([gray_eq, gray_eq, gray_eq], axis=2)
    return out

def process_color_pipeline(img: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    img = rotate_and_crop_color(img)
    if img.size == 0 or img.shape[0] < 2 or img.shape[1] < 2: return np.zeros((100,100,3), dtype=np.float32) # Handle empty crop
    img_inv = 1.0 - img
    black_pts = np.percentile(img_inv, low_pct, axis=(0,1))
    white_pts = np.percentile(img_inv, high_pct, axis=(0,1))
    scale = np.where(np.abs(white_pts-black_pts)>1e-7, 1.0/(white_pts-black_pts + 1e-7), 1.0) # Added epsilon to denominator
    img_corr = np.clip((img_inv - black_pts) * scale, 0.0, 1.0)
    ch_means = np.mean(img_corr, axis=(0,1))
    g_mean = np.mean(ch_means)
    gain = np.where(ch_means>1e-4, g_mean/(ch_means + 1e-7), 1.0) # Added epsilon
    img_corr = np.clip(img_corr * gain, 0.0, 1.0)
    img_corr = _auto_crop_white_border(img_corr)
    return img_corr

# MODIFIED: rt_auto_cast_removal to accept film_type
def rt_auto_cast_removal(in_path, out_path, film_type: str, low_pct=0.5, high_pct=99.5, override_pts=None):
    img = load_image(in_path)

    # If RAW and shading metadata available, apply flat-field correction
    if RAW_SUPPORTED and Path(in_path).suffix.lower() in ('.dng', '.cr2', '.nef', '.arw', '.raf'):
        img = apply_raw_flat_field(in_path, img)

    # Lensfun automatic flat-field (vignetting & distortion) correction
    try:
        img = apply_lensfun_correction(in_path, img)
    except Exception as e_lf:
        logging.warning(f"Lensfun correction failed: {e_lf}")
    
    # User-defined crop takes precedence over auto-detection within pipelines
    if override_pts is not None and len(override_pts) == 4:
        angle = user_angles.get(in_path, 0.0)
        if abs(angle) > 0.001:
            h_orig, w_orig = img.shape[:2]
            M_rot = cv2.getRotationMatrix2D((w_orig/2, h_orig/2), -angle, 1.0) # Negative angle for correction
            img = cv2.warpAffine(img, M_rot, (w_orig, h_orig),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0))
        img = crop_and_warp(img, np.array(override_pts, dtype=np.float32))
        # After manual crop, the pipelines should not re-crop but only process
        # We can pass the already cropped image to simpler versions of pipelines
        # For now, the existing pipelines will try to re-crop; this is acceptable.
        # Or, modify pipelines to skip cropping if image is pre-cropped.

    # Film type selection determines the pipeline
    if film_type == FILM_TYPE_BW_INTERNAL_KEY:
        img_corr = process_bw_pipeline(img, low_pct, high_pct)
    else: # Default to color
        img_corr = process_color_pipeline(img, low_pct, high_pct)
        
    save_image(img_corr, out_path)


def _auto_crop_white_border(img_rgb: np.ndarray, white_thresh: float = 0.97, min_area_ratio: float = 0.2):
    if img_rgb.ndim < 3 or img_rgb.shape[2] < 3: return img_rgb # Ensure it's an RGB image
    gray = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    mask = gray < white_thresh
    if not np.any(mask): return img_rgb
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = img_rgb.shape[:2]
    if (y_max - y_min +1) * (x_max - x_min+1) < min_area_ratio * h * w: return img_rgb # +1 for inclusive
    pad = 2
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(h, y_max + pad +1) # +1 to make slice inclusive
    x_max = min(w, x_max + pad +1)
    return img_rgb[y_min:y_max, x_min:x_max]

ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                '.dng', '.cr2', '.nef', '.arw', '.raf')

def _preview_images(in_path: str, film_type: str):
    before = load_image(in_path)
    after = None # Initialize after
    
    override_pts = user_crops.get(in_path)
    img_to_process = before.copy() # Start with a copy of the original

    if override_pts:
        angle = user_angles.get(in_path, 0.0)
        if abs(angle) > 0.001:
            h, w = img_to_process.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
            img_to_process = cv2.warpAffine(img_to_process, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0,0,0))
        img_to_process = crop_and_warp(img_to_process, np.array(override_pts, dtype=np.float32))
    
    # Now process img_to_process (which is either original or manually cropped/rotated)
    # The individual pipelines (process_bw_pipeline, process_color_pipeline)
    # contain their own rotate_and_crop logic if override_pts was NOT provided.
    # If override_pts WAS provided, img_to_process is ALREADY cropped.
    # The pipelines should ideally take this into account or have a flag.
    # For now, they will re-run detection if no points are passed to crop_and_warp inside them.

    if film_type == FILM_TYPE_BW_INTERNAL_KEY:
        # If already manually cropped, pass the cropped version.
        # The pipelines currently always call rotate_and_crop_xx(img), which then calls detectors.
        # This needs refinement if we want to avoid re-detection on already user-cropped images.
        # A quick fix: if override_pts, the pipeline's internal rotate_and_crop will be on an already cropped image.
        after = process_bw_pipeline(img_to_process, low_pct=0.5, high_pct=99.5)
    else: # Color
        after = process_color_pipeline(img_to_process, low_pct=0.5, high_pct=99.5)
        
    return before, after  # 'before' is always the original loaded image for comparison


def _apply_crop_editor(
    path: str,
    margins: dict[str, float],
    angle: float,
    film_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply manual crop margins and rotation overrides for *path*.

    Returns an updated ``(before, after)`` preview tuple so callers can refresh the UI.
    """

    img_original = load_image(path)
    h_img, w_img = img_original.shape[:2]

    def _clamp_margin(value: float, maximum: float) -> float:
        value = max(0.0, float(value))
        return min(value, maximum)

    left = _clamp_margin(margins.get("left", 0.0), w_img - 1)
    top = _clamp_margin(margins.get("top", 0.0), h_img - 1)
    right = _clamp_margin(margins.get("right", 0.0), w_img - 1 - left)
    bottom = _clamp_margin(margins.get("bottom", 0.0), h_img - 1 - top)

    tl = (left, top)
    tr = (w_img - 1 - right, top)
    br = (w_img - 1 - right, h_img - 1 - bottom)
    bl = (left, h_img - 1 - bottom)

    new_pts_ordered = order_points(np.array([tl, tr, br, bl], dtype=np.float32))
    user_crops[path] = [(float(x), float(y)) for x, y in new_pts_ordered]
    user_angles[path] = float(angle)

    return _preview_images(path, film_type)


def get_manual_crop_settings(path: str) -> tuple[dict[str, float], float]:
    """Return stored crop margins and rotation for *path*.

    The margins are given as pixel offsets from each edge of the original image.
    """

    stored_pts = user_crops.get(path)
    angle = user_angles.get(path, 0.0)
    if not stored_pts:
        return {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}, angle

    img = load_image(path)
    h_img, w_img = img.shape[:2]
    tl, tr, br, bl = np.array(stored_pts, dtype=np.float32)
    margins = {
        "left": max(0.0, float(tl[0])),
        "top": max(0.0, float(tl[1])),
        "right": max(0.0, float(w_img - 1 - tr[0])),
        "bottom": max(0.0, float(h_img - 1 - bl[1])),
    }
    return margins, angle








def _process_path(in_path: str, out_path: str, film_type: str, recurse: bool = False, crop_dict: dict | None = None):
    if os.path.isfile(in_path):
        if os.path.isdir(out_path): # If output is a dir, create filename inside
            stem = Path(in_path).stem + "_out" # Suffix added by save_image
            out_file_path = os.path.join(out_path, stem)
        else: # Assume out_path is a full file path (or will be by save_image)
            out_file_path = out_path
        Path(out_file_path).parent.mkdir(parents=True, exist_ok=True) # Ensure dir for file exists
        logging.info(f"Processing: {in_path} → {out_file_path}.png")
        rt_auto_cast_removal(in_path, out_file_path, film_type, override_pts=crop_dict.get(in_path) if crop_dict else None)
    elif os.path.isdir(in_path):
        os.makedirs(out_path, exist_ok=True)
        for root, _, files in os.walk(in_path):
            for name in files:
                if name.lower().endswith(ALLOWED_EXTS):
                    src = os.path.join(root, name)
                    dst_stem = Path(name).stem + "_out"
                    dst = os.path.join(out_path, dst_stem) # save_image will add .png
                    logging.info(f"Processing: {src} → {dst}.png")
                    rt_auto_cast_removal(src, dst, film_type, override_pts=crop_dict.get(src) if crop_dict else None)
            if not recurse: # Process only top-level directory if not recursing
                break
    else:
        logging.error(f"Input path is not a file or directory: {in_path}")


def launch_qt_ui() -> None:
    from ui_qt.main_window import ProcessingAPI, run_qt_app

    api = ProcessingAPI(
        preview_images=_preview_images,
        process_path=_process_path,
        apply_crop_editor=_apply_crop_editor,
        get_crop_settings=get_manual_crop_settings,
        user_crops=user_crops,
        user_angles=user_angles,
        allowed_extensions=ALLOWED_EXTS,
    )
    run_qt_app(api)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        if len(sys.argv) < 5:  # python main.py --cli <input_path> <output_path> <film_type: color/bw>
            print("Usage: python main.py --cli <input_path> <output_path> <film_type (color or bw)>")
            sys.exit(1)
        cli_input_path = sys.argv[2]
        cli_output_path = sys.argv[3]
        cli_film_type = sys.argv[4].lower()
        if cli_film_type not in [FILM_TYPE_COLOR_INTERNAL_KEY, FILM_TYPE_BW_INTERNAL_KEY]:
            print(f"Invalid film_type: {cli_film_type}. Must be 'color' or 'bw'.")
            sys.exit(1)
        _process_path(cli_input_path, cli_output_path, cli_film_type, recurse=os.path.isdir(cli_input_path))
    else:
        launch_qt_ui()



