import numpy as np
import cv2
import webbrowser
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

import os
import dearpygui.dearpygui as dpg
import platform
import matplotlib.font_manager as fm
import time
import logging # Added for consistency, though not heavily used in original GUI feedback

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# UI_TEXT: English-only
UI_TEXT = {
    "input_label": "Input:",
    "output_label": "Output Directory:",
    "select_file": "Select File",
    "select_folder": "Select Folder",
    "choose_output": "Choose Output Dir",
    "preview": "Preview",
    "process": "Process",
    "status_default": "Preview will appear here.",
    "film_type_label": "Processing Mode:",
    "film_type_color": "Color Negative",
    "film_type_bw": "B&W Negative",
    "left_panel_header": "Controls",
    "right_panel_header": "Preview Area",
    "step1_header": "Step 1 > Choose Negatives",
    "step2_header": "Step 2 > Pick Save Location",
    "step3_header": "Step 3 > Preview & Convert",
    "edit_crop_button": "Edit Crop",
    "apply_crop_button": "Apply Crop",
    "crop_editor_title": "Edit Crop",
    "crop_editor_adjust_for": "Adjust crop for: {}",
    "crop_margin_left": "Left (px)",
    "crop_margin_right": "Right (px)",
    "crop_margin_top": "Top (px)",
    "crop_margin_bottom": "Bottom (px)",
    "crop_rotation_angle": "Rotation Angle (°)",
    "status_no_images": "No supported image files found.",
    "status_preview_generated": "Preview generated for {} image(s).",
    "status_processing": "Processing...",
    "status_completed": "✔ Completed conversion.",
    "status_error": "✘ Error: {}",
    "banner_welcome": "Welcome! Follow these 4 quick steps:",
    "banner_step1": "1. Choose your film negative file *or* a folder containing scans.",
    "banner_step2": "2. Select where to save the converted positives.",
    "banner_step3": "3. Click **Preview** to check the result.",
    "banner_step4": "4. Click **Process** to complete the conversion.",
    "tooltip_lang_combo": "Change the interface language.",
    "tooltip_select_file": "Pick a single scan (JPEG/PNG/TIFF/RAW).",
    "tooltip_select_folder": "Pick a folder to process all supported images inside.",
    "tooltip_choose_output": "Choose the folder where the converted positives will be saved.",
    "tooltip_preview_btn": "Shows a quick before/after view without saving any files.",
    "tooltip_process_btn": "Runs the full conversion and writes the results to disk.",
}

# Constants for film type radio buttons keys (internal, non-localized)
FILM_TYPE_COLOR_INTERNAL_KEY = "color"
FILM_TYPE_BW_INTERNAL_KEY = "bw"


# Global storage for manual crop overrides
user_crops = {}
# Global storage for manual rotation overrides (degrees)
user_angles = {}


# Globals for current crop editor context
editing_path = None
editing_idx = None


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
        raise IOError(f"Failed to write image PNG to {png_path}.")
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

def _numpy_to_rgba32f(img: np.ndarray):
    if img.dtype == np.uint8: img = img.astype(np.float32) / 255.0
    elif img.dtype == np.float32: img = np.clip(img, 0.0, 1.0)
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0: img /= (65535.0 if img.max() > 255 else 255.0) # Handle 16-bit or 8-bit scaling
        img = np.clip(img, 0.0, 1.0)

    if img.ndim == 2: rgba = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGBA).astype(np.float32) / 255.0
    elif img.shape[2] == 3: rgba = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2RGBA).astype(np.float32) / 255.0
    elif img.shape[2] == 4: rgba = img
    else: raise ValueError("Unsupported image format for texture")
    h, w, _ = rgba.shape
    return w, h, rgba.ravel() # ravel() is fine

def _create_texture(img: np.ndarray, tag: str):
    w, h, data = _numpy_to_rgba32f(img)
    if dpg.does_item_exist(tag): # If texture exists, update it
        # dpg.set_value(tag, data) # This is for value of texture, not texture data directly
        # For static textures, we might need to delete and recreate if size changes,
        # or use dynamic textures. For simplicity, let's try to update or recreate.
        # A robust way is to delete and add.
        dpg.delete_item(tag) 
    
    # Add new static texture
    # Ensure texture registry exists or is active, usually default one is fine
    # With DearPyGui 1.x, texture_registry is mostly implicit for add_static_texture
    try:
        return dpg.add_static_texture(width=w, height=h, default_value=data, tag=tag, parent="texture_registry_main")
    except Exception as e:
        logging.error(f"Failed to create/update texture {tag}: {e}")
        # Fallback: create a dummy texture or return None
        dummy_data = np.zeros((10 * 10 * 4), dtype=np.float32)
        if dpg.does_item_exist(tag): dpg.delete_item(tag) # cleanup if failed half-way
        return dpg.add_static_texture(width=10, height=10, default_value=dummy_data, tag=tag, parent="texture_registry_main")


# REMOVED: _process_image_array as logic moved to _preview_images and _update_crop_preview

# MODIFIED: _preview_images to use selected film_type
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
        
    return before, after # 'before' is always the original loaded image for comparison


# MODIFIED: _update_crop_preview to use selected film_type
def _update_crop_preview(sender, app_data):
    global current_lang, editing_path, editing_idx
    before_original = load_image(editing_path)
    img_for_preview = before_original.copy()

    angle = dpg.get_value("crop_angle") if dpg.does_item_exist("crop_angle") else 0.0
    if abs(angle) > 0.001:
        h, w = img_for_preview.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0) # Negative angle
        img_for_preview = cv2.warpAffine(img_for_preview, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0,0,0))
    
    margin_left   = dpg.get_value("crop_left")   if dpg.does_item_exist("crop_left") else 0
    margin_right  = dpg.get_value("crop_right")  if dpg.does_item_exist("crop_right") else 0
    margin_top    = dpg.get_value("crop_top")    if dpg.does_item_exist("crop_top") else 0
    margin_bottom = dpg.get_value("crop_bottom") if dpg.does_item_exist("crop_bottom") else 0

    h_img, w_img = before_original.shape[:2] # Use original for margin calculation relative to un-rotated
    margin_left   = int(np.clip(margin_left,   0, w_img-1))
    margin_right  = int(np.clip(margin_right,  0, w_img-1 - margin_left))
    margin_top    = int(np.clip(margin_top,    0, h_img-1))
    margin_bottom = int(np.clip(margin_bottom, 0, h_img-1 - margin_top))

    tl = (margin_left, margin_top)
    tr = (w_img - 1 - margin_right, margin_top)
    br = (w_img - 1 - margin_right, h_img - 1 - margin_bottom)
    bl = (margin_left, h_img - 1 - margin_bottom)
    pts = [tl, tr, br, bl]
    
    # The crop_and_warp should be applied to the (potentially) rotated image_for_preview
    # if the margins are relative to the rotated image.
    # If margins are relative to original image, then rotation should be applied AFTER defining pts based on original.
    # Current logic: angle applied, then margins define pts on original geom, then crop applied to rotated.
    # This means pts must be transformed if they are from original coordinate system.
    # Simpler: assume margins are applied on the `img_for_preview` which might already be rotated.
    
    h_preview, w_preview = img_for_preview.shape[:2]
    preview_tl = (margin_left, margin_top)
    preview_tr = (w_preview -1 - margin_right, margin_top)
    preview_br = (w_preview -1 - margin_right, h_preview -1 - margin_bottom)
    preview_bl = (margin_left, h_preview-1-margin_bottom)

    # Ensure points are within the bounds of img_for_preview
    preview_pts_np = np.array([preview_tl, preview_tr, preview_br, preview_bl], dtype=np.float32)
    preview_pts_np[:,0] = np.clip(preview_pts_np[:,0], 0, w_preview -1)
    preview_pts_np[:,1] = np.clip(preview_pts_np[:,1], 0, h_preview -1)

    img_cropped = crop_and_warp(img_for_preview, preview_pts_np)

    # Get selected film type
    selected_film_type_key = get_selected_film_type_key_from_gui()

    after = None
    if selected_film_type_key == FILM_TYPE_BW_INTERNAL_KEY:
        after = process_bw_pipeline(img_cropped, low_pct=0.5, high_pct=99.5)
    else:
        after = process_color_pipeline(img_cropped, low_pct=0.5, high_pct=99.5)

    tex_tag = f"crop_live_tex_{editing_idx}_{int(time.time()*1000)}" # Unique tag
    tex = _create_texture(after, tex_tag)
    
    if dpg.does_item_exist(f"after_view_{editing_idx}"):
        # Check if the image item itself exists to configure it
        if dpg.does_item_exist(f"img_display_after_{editing_idx}"):
             dpg.configure_item(f"img_display_after_{editing_idx}", texture_tag=tex)
        else: # If not, it might be the first time, try adding image to after_view container
             w_after, h_after, _ = _numpy_to_rgba32f(after) # Need w,h for add_image
             dpg.add_image(tex, width=w_after, height=h_after, tag=f"img_display_after_{editing_idx}", parent=f"after_view_container_{editing_idx}")


def _open_crop_editor(sender, app_data, user_data):
    global editing_path, editing_idx
    path, idx = user_data
    editing_path = path
    editing_idx = idx
    selected_film_type_key = get_selected_film_type_key_from_gui()
    pts_to_edit = user_crops.get(path, None)
    initial_angle = user_angles.get(path, 0.0)
    img_for_detection = load_image(path)
    h_img, w_img = img_for_detection.shape[:2]
    if pts_to_edit is None:
        temp_img_rotated_for_detection = img_for_detection
        detected_pts = None
        LOWER_ORANGE = np.array([2,50,50])
        UPPER_ORANGE = np.array([30,255,255])
        if selected_film_type_key == FILM_TYPE_BW_INTERNAL_KEY:
            dets = (detect_film_region_proj, detect_film_region_refined, detect_film_region_bw, detect_film_region_hough, detect_film_region_simple)
        else:
            dets = (lambda im: detect_film_region_color(im, LOWER_ORANGE, UPPER_ORANGE), detect_film_region_hough, detect_film_region_proj, detect_film_region_refined, detect_film_region_simple)
        for det_func in dets:
            if detected_pts is None:
                detected_pts = det_func(temp_img_rotated_for_detection)
        if detected_pts is not None:
            tl_det, tr_det, br_det, bl_det = detected_pts
            h_det_img, w_det_img = temp_img_rotated_for_detection.shape[:2]
            init_left = int(max(0.0, tl_det[0]))
            init_top = int(max(0.0, tl_det[1]))
            init_right = int(max(0.0, w_det_img - tr_det[0]))
            init_bottom = int(max(0.0, h_det_img - bl_det[1]))
            user_crops[path] = [(float(x), float(y)) for x,y in detected_pts]
            pts_to_edit = user_crops[path]
        else:
            logging.warning(f"No crop region auto-detected for: {Path(path).name}. Using full image.")
            init_left, init_top, init_right, init_bottom = 0,0,0,0
            pts_to_edit = [(0.0,0.0), (float(w_img-1), 0.0), (float(w_img-1), float(h_img-1)), (0.0, float(h_img-1))]
            user_crops[path] = pts_to_edit
    current_tl, current_tr, _, current_bl = order_points(np.array(pts_to_edit, dtype=np.float32))
    init_left   = int(max(0.0, current_tl[0]))
    init_top    = int(max(0.0, current_tl[1]))
    init_right  = int(max(0.0, w_img - 1 - current_tr[0]))
    init_bottom = int(max(0.0, h_img - 1 - current_bl[1]))
    if dpg.does_item_exist("CropEditor"): dpg.delete_item("CropEditor")
    with dpg.window(label=UI_TEXT["crop_editor_title"], tag="CropEditor", width=400, height=350, pos=[dpg.get_viewport_width() // 2 - 200, dpg.get_viewport_height() // 2 - 175]):
        with dpg.child_window(tag="CropEditorBody", autosize_x=True, border=False):
            dpg.add_text(UI_TEXT["crop_editor_adjust_for"].format(Path(path).name))
            dpg.add_text("Note: 'Top' = upper edge, 'Bottom' = lower edge, 'Left' = left edge, 'Right' = right edge.", wrap=350)
            # Orientation guide diagram
            dpg.add_text("Orientation Guide:", wrap=350)
            with dpg.drawlist(width=250, height=150):
                # Outer rectangle
                dpg.draw_rectangle((10, 10), (240, 140))
                # Side labels
                dpg.draw_text((120, 12), "Top")
                dpg.draw_text((120, 130), "Bottom")
                dpg.draw_text((12, 75), "Left")
                dpg.draw_text((200, 75), "Right")
            dpg.add_input_int(
                label=UI_TEXT["crop_margin_left"],
                default_value=init_left,
                tag="crop_left",
                callback=_update_crop_preview,
                min_value=0, step=1, width=-1
            )
            dpg.add_input_int(
                label=UI_TEXT["crop_margin_right"],
                default_value=init_right,
                tag="crop_right",
                callback=_update_crop_preview,
                min_value=0, step=1, width=-1
            )
            dpg.add_input_int(
                label=UI_TEXT["crop_margin_top"],
                default_value=init_top,
                tag="crop_top",
                callback=_update_crop_preview,
                min_value=0, step=1, width=-1
            )
            dpg.add_input_int(
                label=UI_TEXT["crop_margin_bottom"],
                default_value=init_bottom,
                tag="crop_bottom",
                callback=_update_crop_preview,
                min_value=0, step=1, width=-1
            )
            dpg.add_input_float(
                label=UI_TEXT["crop_rotation_angle"],
                default_value=initial_angle,
                tag="crop_angle",
                callback=_update_crop_preview,
                step=0.5, format="%.1f deg", width=-1
            )
            dpg.add_spacer(height=10)
            dpg.add_button(
                label=UI_TEXT["apply_crop_button"],
                callback=_apply_crop_editor,
                width=-1, height=30
            )
        _update_crop_preview(None, None)


def _apply_crop_editor(sender, app_data):
    global editing_path, editing_idx
    img_original_geom = load_image(editing_path)
    h_img, w_img = img_original_geom.shape[:2]
    margin_left   = dpg.get_value("crop_left")
    margin_right  = dpg.get_value("crop_right")
    margin_top    = dpg.get_value("crop_top")
    margin_bottom = dpg.get_value("crop_bottom")
    # Ensure margin values are not None to avoid TypeError
    if margin_left is None:
        margin_left = 0
    if margin_right is None:
        margin_right = 0
    if margin_top is None:
        margin_top = 0
    if margin_bottom is None:
        margin_bottom = 0
    tl = (float(margin_left), float(margin_top))
    tr = (float(w_img - 1 - margin_right), float(margin_top))
    br = (float(w_img - 1 - margin_right), float(h_img - 1 - margin_bottom))
    bl = (float(margin_left), float(h_img - 1 - margin_bottom))
    new_pts_ordered = order_points(np.array([tl, tr, br, bl], dtype=np.float32))
    user_crops[editing_path] = [(x,y) for x,y in new_pts_ordered]
    angle = dpg.get_value("crop_angle")
    user_angles[editing_path] = float(angle)
    selected_film_type_key = get_selected_film_type_key_from_gui()
    before_img, after_img = _preview_images(editing_path, selected_film_type_key)
    after_view_container_tag = f"after_view_container_{editing_idx}"
    img_display_tag = f"img_display_after_{editing_idx}"
    preview_area_width = dpg.get_item_width("preview_area_content")
    if preview_area_width is None or preview_area_width <=0: preview_area_width = 600
    panel_w = max(100, int(preview_area_width / 2.0 - 20))
    after_ratio = after_img.shape[0] / (after_img.shape[1] + 1e-6)
    img_h_preview = int(panel_w * after_ratio)
    img_w_preview = panel_w
    tex_tag = f"tex_after_{editing_idx}_{int(time.time()*1000)}"
    new_tex = _create_texture(after_img, tex_tag)
    if dpg.does_item_exist(img_display_tag):
        dpg.configure_item(img_display_tag, texture_tag=new_tex, width=img_w_preview, height=img_h_preview)
    elif dpg.does_item_exist(after_view_container_tag):
        dpg.add_image(new_tex, tag=img_display_tag, parent=after_view_container_tag, width=img_w_preview, height=img_h_preview)
    if dpg.does_item_exist("CropEditor"):
        dpg.delete_item("CropEditor")


# MODIFIED: _process_path to accept and use film_type
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


# Helper to get the currently selected film type's internal key
def get_selected_film_type_key_from_gui():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist("film_type_selector"):
        return FILM_TYPE_COLOR_INTERNAL_KEY
    selected_displayed_text = dpg.get_value("film_type_selector")
    if selected_displayed_text == UI_TEXT["film_type_bw"]:
        return FILM_TYPE_BW_INTERNAL_KEY
    return FILM_TYPE_COLOR_INTERNAL_KEY

# =========================== GUI (DearPyGui) ===============================
def build_gui():
    dpg.create_context()

    # ----- Font registration -----
    # Use system font for all text, since only English is needed now
    from matplotlib import font_manager as fm_manager
    default_font_path = fm_manager.findfont(fm_manager.FontProperties())
    with dpg.font_registry():
        main_font = dpg.add_font(default_font_path, 20, tag="main_font")
    dpg.bind_font("main_font")

    dpg.create_viewport(title='Film Negative Converter Pro (Retro)', width=1024, height=768)

    # Texture registry
    with dpg.texture_registry(tag="texture_registry_main"):
        pass

    preview_textures_tags = []


    def _select_file_cb(sender, app_data):
        # app_data contains 'file_path_name'
        if "file_path_name" in app_data:
            dpg.set_value("input_path", app_data["file_path_name"])
            dpg.enable_item("preview_btn")
            dpg.enable_item("process_btn")
        else: # Handle older DPG or different structure if any
            if len(app_data['selections']) > 0:
                 dpg.set_value("input_path", list(app_data['selections'].values())[0])
                 dpg.enable_item("preview_btn")
                 dpg.enable_item("process_btn")


    def _select_folder_cb(sender, app_data):
        # app_data contains 'file_path_name' for directory
        if "file_path_name" in app_data: # DPG 1.x style
            dpg.set_value("input_path", app_data["file_path_name"])
        elif "current_path" in app_data: # Older style
            dpg.set_value("input_path", app_data["current_path"])
        else: # Fallback for selections if new DPG versions change dialog app_data
             if len(app_data['selections']) > 0:
                 dpg.set_value("input_path", list(app_data['selections'].values())[0])

        dpg.enable_item("preview_btn")
        dpg.enable_item("process_btn")

    def _select_output_cb(sender, app_data):
        if "file_path_name" in app_data:
            dpg.set_value("output_path", app_data["file_path_name"])
        elif "current_path" in app_data:
            dpg.set_value("output_path", app_data["current_path"])
        else:
             if len(app_data['selections']) > 0:
                 dpg.set_value("output_path", list(app_data['selections'].values())[0])


    def _preview_cb():
        nonlocal preview_textures_tags
        in_path = dpg.get_value("input_path")
        if not in_path or not os.path.exists(in_path):
            dpg.set_value("status_text", UI_TEXT["status_no_images"])
            return
        ui_controls = ["btn_select_file", "btn_select_folder", "btn_choose_output", "preview_btn", "process_btn", "film_type_selector"]
        for tag_ in ui_controls: dpg.disable_item(tag_)
        dpg.set_value("status_text", UI_TEXT["status_processing"])
        paths = []
        if os.path.isdir(in_path):
            for root, _, files in os.walk(in_path):
                for fname in sorted(files):
                    if fname.lower().endswith(ALLOWED_EXTS):
                        paths.append(os.path.join(root, fname))
                break
        elif os.path.isfile(in_path) and in_path.lower().endswith(ALLOWED_EXTS):
            paths.append(in_path)
        if not paths:
            dpg.set_value("status_text", UI_TEXT["status_no_images"])
            for tag_ in ui_controls: dpg.enable_item(tag_)
            return
        # Clear existing preview items first
        if dpg.does_item_exist("preview_area_content"):
            dpg.delete_item("preview_area_content", children_only=True)
        else:
            logging.error("preview_area_content not found for clearing.")
            for tag_ in ui_controls: dpg.enable_item(tag_)
            return
        # Now delete old textures
        for tex_tag_to_del in preview_textures_tags:
            if dpg.does_item_exist(tex_tag_to_del):
                dpg.delete_item(tex_tag_to_del)
        preview_textures_tags.clear()
        dpg.add_progress_bar(default_value=0.0, tag="preview_progress", width=-1, parent="preview_area_content")
        selected_film_type_key = get_selected_film_type_key_from_gui()
        preview_area_width = dpg.get_item_width("preview_area_content")
        if preview_area_width is None or preview_area_width <= 0 : preview_area_width = 600
        panel_w = max(100, int(preview_area_width / 2.0 - 24))
        for i, pth_str in enumerate(paths):
            try:
                before_img, after_img = _preview_images(pth_str, selected_film_type_key)
                dpg.set_value("preview_progress", float(i + 1) / len(paths))
                before_ratio = (before_img.shape[0] / (before_img.shape[1] + 1e-6)) if before_img.size >0 else 1
                after_ratio  = (after_img.shape[0]  / (after_img.shape[1] + 1e-6)) if after_img.size > 0 else 1
                before_h = int(panel_w * before_ratio)
                after_h = int(panel_w * after_ratio) if after_img.size > 0 and after_ratio > 0 else before_h
                before_tex_tag = f"tex_before_{i}_{int(time.time()*10000)}"
                after_tex_tag  = f"tex_after_{i}_{int(time.time()*10000)}"
                before_tex = _create_texture(before_img, before_tex_tag)
                after_tex  = _create_texture(after_img,  after_tex_tag)
                preview_textures_tags.extend([before_tex_tag, after_tex_tag])
                with dpg.group(parent="preview_area_content", horizontal=True):
                    with dpg.child_window(width=panel_w, height=max(before_h, after_h) + 60, border=False):
                        dpg.add_text(f"Before ({Path(pth_str).name})")
                        dpg.add_image(before_tex_tag, width=panel_w, height=before_h)
                    after_container_tag = f"after_view_container_{i}"
                    with dpg.child_window(width=panel_w, height=max(before_h, after_h) + 60, tag=after_container_tag, border=False):
                        dpg.add_text("After")
                        dpg.add_image(after_tex_tag, tag=f"img_display_after_{i}", width=panel_w, height=after_h)
                        dpg.add_button(label=UI_TEXT["edit_crop_button"], callback=_open_crop_editor, user_data=(pth_str, i))
            except Exception as e_preview:
                logging.error(f"Error previewing {pth_str}: {e_preview}")
                dpg.add_text(f"Error previewing {Path(pth_str).name}: {e_preview}", parent="preview_area_content", color=[255,0,0])
        if dpg.does_item_exist("preview_progress"): dpg.delete_item("preview_progress")
        for tag_ in ui_controls: dpg.enable_item(tag_)
        dpg.set_value("status_text", UI_TEXT["status_preview_generated"].format(len(paths)))


    def _process_cb():
        in_path  = dpg.get_value("input_path")
        parent_out = dpg.get_value("output_path")
        if not in_path or not os.path.exists(in_path):
            dpg.set_value("status_text", f"Error: Input path '{in_path}' does not exist or is invalid.")
            return
        if not parent_out:
            parent_out = os.getcwd()
            dpg.set_value("output_path", parent_out)
        out_path = os.path.join(parent_out, "converted_output")
        os.makedirs(out_path, exist_ok=True)
        dpg.set_value("status_text", UI_TEXT["status_processing"])
        dpg.disable_item("process_btn")
        dpg.disable_item("preview_btn")
        selected_film_type_key = get_selected_film_type_key_from_gui()
        try:
            crop_dict_to_use = user_crops
            is_recursive = True if os.path.isdir(in_path) else False
            _process_path(in_path, out_path, selected_film_type_key,
                          recurse=is_recursive,
                          crop_dict=crop_dict_to_use)
            dpg.set_value("status_text", UI_TEXT["status_completed"])
        except Exception as e:
            dpg.set_value("status_text", UI_TEXT["status_error"].format(e))
            logging.exception("Processing failed")
        finally:
            dpg.enable_item("process_btn")
            dpg.enable_item("preview_btn")

    # ---------- Main UI Layout ----------
    with dpg.window(tag="MainWindow"):
        # Copyright and website at the very top
        dpg.add_button(label="© 2025 Lex - Visit https://tokugai.com", tag="copyright_btn",
                       callback=lambda: webbrowser.open("https://tokugai.com"))
        with dpg.group(horizontal=True):
            # ---------- Left Panel (Controls) ----------
            with dpg.child_window(tag="left_panel", width=380):
                dpg.add_text(UI_TEXT["left_panel_header"], tag="left_panel_header_text")
                dpg.add_separator()
                dpg.add_text(UI_TEXT["film_type_label"], tag="film_type_label_text")
                film_type_radio_items = [UI_TEXT["film_type_color"], UI_TEXT["film_type_bw"]]
                dpg.add_radio_button(items=film_type_radio_items, horizontal=True,
                                     tag="film_type_selector",
                                     default_value=film_type_radio_items[0])
                dpg.add_spacer(height=10)
                # --- Banner ---
                with dpg.collapsing_header(label="Quick Start Guide", default_open=False):
                    dpg.add_text("Welcome! Here's how to convert your film negatives:", wrap=350)
                    dpg.add_text("1. Select a film type (Color Negative or B&W Negative).", wrap=350)
                    dpg.add_text("2. Use 'Select File' or 'Select Folder' to choose input scans.", wrap=350)
                    dpg.add_text("3. Set an output directory using 'Choose Output Dir'.", wrap=350)
                    dpg.add_text("4. Click 'Preview' to view before/after conversions.", wrap=350)
                    dpg.add_text("5. Optionally, adjust crop or rotation via 'Edit Crop'.", wrap=350)
                    dpg.add_text("6. Click 'Process' to save all converted images.", wrap=350)
                dpg.add_separator()
                # --- Workflow Sections ---
                with dpg.collapsing_header(label=UI_TEXT["step1_header"], default_open=True, tag="collapsing_header_step1"):
                    dpg.add_text(UI_TEXT["input_label"], tag="input_label_text")
                    dpg.add_input_text(tag="input_path", readonly=True, width=-1)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=UI_TEXT["select_file"], tag="btn_select_file", callback=lambda: dpg.show_item("file_dialog"))
                        with dpg.tooltip("btn_select_file"): dpg.add_text(UI_TEXT["tooltip_select_file"], tag="btn_select_file_tooltip_text")
                        dpg.add_button(label=UI_TEXT["select_folder"], tag="btn_select_folder", callback=lambda: dpg.show_item("folder_dialog"))
                        with dpg.tooltip("btn_select_folder"): dpg.add_text(UI_TEXT["tooltip_select_folder"], tag="btn_select_folder_tooltip_text")
                with dpg.collapsing_header(label=UI_TEXT["step2_header"], default_open=True, tag="collapsing_header_step2"):
                    dpg.add_text(UI_TEXT["output_label"], tag="output_label_text")
                    dpg.add_input_text(tag="output_path", readonly=True, width=-1)
                    dpg.add_button(label=UI_TEXT["choose_output"], tag="btn_choose_output", callback=lambda: dpg.show_item("output_dialog"))
                    with dpg.tooltip("btn_choose_output"): dpg.add_text(UI_TEXT["tooltip_choose_output"], tag="btn_choose_output_tooltip_text")
                with dpg.collapsing_header(label=UI_TEXT["step3_header"], default_open=True, tag="collapsing_header_step3"):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=UI_TEXT["preview"], tag="preview_btn", callback=_preview_cb, enabled=False)
                        with dpg.tooltip("preview_btn"): dpg.add_text(UI_TEXT["tooltip_preview_btn"], tag="preview_btn_tooltip_text")
                        dpg.add_button(label=UI_TEXT["process"], tag="process_btn", callback=_process_cb, enabled=False)
                        with dpg.tooltip("process_btn"): dpg.add_text(UI_TEXT["tooltip_process_btn"], tag="process_btn_tooltip_text")
                    dpg.add_spacer(height=4)
                    dpg.add_text(UI_TEXT["status_default"], tag="status_text")
            # ---------- Right Panel (Preview) ----------
            with dpg.child_window(tag="right_panel", width=-1):
                with dpg.child_window(tag="preview_area_content", width=-1, height=-1, menubar=False):
                    dpg.add_text(UI_TEXT["status_default"])

    # ---------- File Dialogs (defined once) ----------
    img_ext_string = (
        "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff "
        "*.dng *.cr2 *.nef *.arw *.raf){"
        ".jpg,.jpeg,.png,.bmp,.tif,.tiff,"
        ".dng,.cr2,.nef,.arw,.raf,"
        ".JPG,.JPEG,.PNG,.BMP,.TIF,.TIFF,"
        ".DNG,.CR2,.NEF,.ARW,.RAF},.*"
    )
    with dpg.file_dialog(directory_selector=False, show=False, callback=_select_file_cb, tag="file_dialog", width=700, height=400, modal=True):
        dpg.add_file_extension(img_ext_string)
        dpg.add_file_extension(".*")
    with dpg.file_dialog(directory_selector=True, show=False, callback=_select_folder_cb, tag="folder_dialog", width=700, height=400, modal=True):
        pass
    with dpg.file_dialog(directory_selector=True, show=False, callback=_select_output_cb, tag="output_dialog", width=700, height=400, modal=True):
        pass
    # ---------- Styling (Retro Theme Attempt) ----------
    with dpg.theme(tag="global_retro_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 30, 240))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (40, 40, 40, 240))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (20, 20, 20, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 120, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 150, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 100, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (0, 80, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (0, 100, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (0, 60, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (20,20,20,255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (0,100,0,255))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 4)
        with dpg.theme_component(dpg.mvProgressBar):
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (0, 200, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50,50,50,255))
    dpg.bind_theme("global_retro_theme")
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("MainWindow", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

# =========================== End GUI =======================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        if len(sys.argv) < 5: # python test.py --cli <input_path> <output_path> <film_type: color/bw>
            print("Usage: python test.py --cli <input_path> <output_path> <film_type (color or bw)>")
            sys.exit(1)
        cli_input_path = sys.argv[2]
        cli_output_path = sys.argv[3]
        cli_film_type = sys.argv[4].lower()
        if cli_film_type not in [FILM_TYPE_COLOR_INTERNAL_KEY, FILM_TYPE_BW_INTERNAL_KEY]:
            print(f"Invalid film_type: {cli_film_type}. Must be 'color' or 'bw'.")
            sys.exit(1)
        _process_path(cli_input_path, cli_output_path, cli_film_type, recurse=os.path.isdir(cli_input_path))
    else:
        build_gui()