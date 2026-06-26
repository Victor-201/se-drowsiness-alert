import numpy as np
import logging
from math import atan2, degrees

logger = logging.getLogger(__name__)


def euclidean_distance(a, b):
    diff = a - b
    return np.sqrt(np.sum(diff * diff))


def tao_kernel_gaussian(sigma):
    r = int(np.ceil(3 * sigma))
    truc = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(truc, truc)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def tich_chap_2d(anh, kernel):
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    anh_pad = np.pad(anh.astype(np.float64), ((pH, pH), (pW, pW)), mode='reflect')
    H, W = anh.shape
    shape = (H, W, kH, kW)
    strides = (anh_pad.strides[0], anh_pad.strides[1],
               anh_pad.strides[0], anh_pad.strides[1])
    cua_so = np.lib.stride_tricks.as_strided(anh_pad, shape=shape, strides=strides)
    return np.einsum('ijkl,kl->ij', cua_so, kernel)


KERN_X = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)

KERN_Y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float64)


def tinh_gradient(anh_mo):
    Gx = tich_chap_2d(anh_mo, KERN_X)
    Gy = tich_chap_2d(anh_mo, KERN_Y)
    M = np.hypot(Gx, Gy)
    theta = np.degrees(np.arctan2(Gy, Gx)) % 180
    return Gx, Gy, M, theta


def non_max_suppression(M, theta):
    H, W = M.shape
    ket_qua = np.zeros_like(M)
    huong = np.zeros_like(theta, dtype=np.int8)
    huong[(theta >= 22.5) & (theta < 67.5)] = 1
    huong[(theta >= 67.5) & (theta < 112.5)] = 2
    huong[(theta >= 112.5) & (theta < 157.5)] = 3
    cac_huong = {0: (0, 1), 1: (-1, 1), 2: (-1, 0), 3: (-1, -1)}
    for d, (dy, dx) in cac_huong.items():
        mat_na = (huong == d)
        r1 = np.clip(np.arange(H)[:, None] + dy, 0, H - 1)
        c1 = np.clip(np.arange(W)[None, :] + dx, 0, W - 1)
        r2 = np.clip(np.arange(H)[:, None] - dy, 0, H - 1)
        c2 = np.clip(np.arange(W)[None, :] - dx, 0, W - 1)
        giu = mat_na & (M >= M[r1, c1]) & (M >= M[r2, c2])
        ket_qua[giu] = M[giu]
    return ket_qua


def hysteresis(nms_map, tau_low, tau_high):
    manh = nms_map >= tau_high
    yeu = (nms_map >= tau_low) & ~manh
    ket_qua = manh.copy()
    H, W = nms_map.shape
    from collections import deque
    q = deque(zip(*np.where(manh)))
    while q:
        y, x = q.popleft()
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        for ny in range(y0, y1):
            for nx in range(x0, x1):
                if (ny == y and nx == x) or not yeu[ny, nx] or ket_qua[ny, nx]:
                    continue
                ket_qua[ny, nx] = True
                q.append((ny, nx))
    return ket_qua.astype(np.uint8) * 255


def otsu_threshold(nms_map):
    pixel_khac0 = nms_map[nms_map > 0]
    if len(pixel_khac0) == 0:
        return 50, 25
    max_val = nms_map.max()
    if max_val == 0:
        return 50, 25
    pixel_u8 = (pixel_khac0 / max_val * 255).astype(np.uint8)
    hist = np.bincount(pixel_u8, minlength=256).astype(np.float64)
    tong = hist.sum()
    bins = np.arange(256, dtype=np.float64)
    cum_w = np.cumsum(hist) / tong
    cum_mu = np.cumsum(hist * bins) / tong
    mu_all = cum_mu[-1]
    w0 = cum_w
    w1 = 1.0 - w0
    mu0 = np.where(w0 > 0, cum_mu / (w0 + 1e-12), 0)
    mu1 = np.where(w1 > 0, (mu_all - cum_mu) / (w1 + 1e-12), 0)
    phuong_sai = w0 * w1 * (mu0 - mu1) ** 2
    t_opt = int(np.argmax(phuong_sai))
    tau_h = t_opt / 255.0 * max_val
    tau_l = tau_h / 2.0
    return max(tau_h, 10), max(tau_l, 5)


def manual_canny(gray, sigma=0.8, low=None, high=None):
    anh_mo = tich_chap_2d(gray, tao_kernel_gaussian(sigma))
    _, _, M, theta = tinh_gradient(anh_mo)
    nms = non_max_suppression(M, theta)
    if low is None or high is None:
        high, low = otsu_threshold(nms)
    edges = hysteresis(nms, low, high)
    return edges


def manual_adaptive_histogram_equalization(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = np.asarray(gray, dtype=np.float64)
    h, w = gray.shape
    t_h = h // tile_grid_size[0]
    t_w = w // tile_grid_size[1]
    bins = 256

    mappings = np.zeros((tile_grid_size[0], tile_grid_size[1], bins), dtype=np.float64)
    for ti in range(tile_grid_size[0]):
        for tj in range(tile_grid_size[1]):
            y0, y1 = ti * t_h, (ti + 1) * t_h
            x0, x1 = tj * t_w, (tj + 1) * t_w
            tile = gray[y0:y1, x0:x1].ravel().astype(np.intp)
            hist = np.bincount(tile, minlength=bins).astype(np.float64)
            clip_val = clip_limit * (tile.size / bins) if clip_limit > 0 else 0
            if clip_val > 0:
                excess = np.sum(np.maximum(hist - clip_val, 0))
                hist = np.minimum(hist, clip_val)
                hist += excess / bins
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1] * 255.0
            mappings[ti, tj] = cdf

    ti_f = (np.arange(h, dtype=np.float64) + 0.5) / t_h - 0.5
    tj_f = (np.arange(w, dtype=np.float64) + 0.5) / t_w - 0.5
    ti_f = np.clip(ti_f, 0, tile_grid_size[0] - 1)
    tj_f = np.clip(tj_f, 0, tile_grid_size[1] - 1)
    ti0 = np.floor(ti_f).astype(np.intp)
    ti1 = np.minimum(ti0 + 1, tile_grid_size[0] - 1).astype(np.intp)
    tj0 = np.floor(tj_f).astype(np.intp)
    tj1 = np.minimum(tj0 + 1, tile_grid_size[1] - 1).astype(np.intp)
    dy = (ti_f - ti0)[:, np.newaxis]
    dx = (tj_f - tj0)[np.newaxis, :]
    idx = np.clip(np.round(gray), 0, bins - 1).astype(np.intp)

    shape = (h, w)
    v00 = mappings[ti0[:, np.newaxis], tj0[np.newaxis, :], idx]
    v10 = mappings[ti1[:, np.newaxis], tj0[np.newaxis, :], idx]
    v01 = mappings[ti0[:, np.newaxis], tj1[np.newaxis, :], idx]
    v11 = mappings[ti1[:, np.newaxis], tj1[np.newaxis, :], idx]
    result = (v00 * (1 - dx) + v01 * dx) * (1 - dy) + (v10 * (1 - dx) + v11 * dx) * dy
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def manual_bounding_rect(pts):
    xs = pts[:, 0]
    ys = pts[:, 1]
    x = int(xs.min())
    y = int(ys.min())
    w = int(xs.max() - x)
    h = int(ys.max() - y)
    return x, y, w, h


def manual_largest_contour_area(binary):
    labeled, _ = label_connected_components(binary)
    if labeled is None:
        return 0
    labels = labeled[labeled > 0]
    if len(labels) == 0:
        return 0
    counts = np.bincount(labels)
    largest_area = counts.max()
    return int(largest_area)


def label_connected_components(binary):
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 1
    equivalences = {}

    def find(x):
        while equivalences.get(x, x) != x:
            equivalences[x] = equivalences[equivalences[x]]
            x = equivalences[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            equivalences[ra] = rb

    for i in range(h):
        for j in range(w):
            if binary[i, j] == 0:
                continue
            neighbors = []
            if i > 0 and labeled[i - 1, j] > 0:
                neighbors.append(labeled[i - 1, j])
            if j > 0 and labeled[i, j - 1] > 0:
                neighbors.append(labeled[i, j - 1])
            if not neighbors:
                labeled[i, j] = current_label
                equivalences[current_label] = current_label
                current_label += 1
            else:
                labeled[i, j] = min(neighbors)
                for n in neighbors:
                    union(labeled[i, j], n)

    for i in range(h):
        for j in range(w):
            if labeled[i, j] > 0:
                labeled[i, j] = find(labeled[i, j])

    unique = np.unique(labeled[labeled > 0])
    if len(unique) == 0:
        return None, None
    mapping = {old: new + 1 for new, old in enumerate(unique)}
    remapped = np.zeros_like(labeled)
    for i in range(h):
        for j in range(w):
            if labeled[i, j] > 0:
                remapped[i, j] = mapping[labeled[i, j]]

    return remapped, len(unique)


def manual_sobel_edges(gray):
    gray_f = gray.astype(np.float64)
    Gx = tich_chap_2d(gray_f, KERN_X)
    Gy = tich_chap_2d(gray_f, KERN_Y)
    M = np.hypot(Gx, Gy)
    return np.clip(M, 0, 255).astype(np.uint8)


class FacialAnalyzer:
    def __init__(self):
        self.min_ear = 0.15
        self.max_ear = 0.40

    def apply_clahe(self, gray_frame):
        return manual_adaptive_histogram_equalization(gray_frame, clip_limit=2.0, tile_grid_size=(8, 8))

    def calculate_ear(self, eye_points):
        points = np.array(eye_points, dtype=np.float32)
        A = euclidean_distance(points[1], points[5])
        B = euclidean_distance(points[2], points[4])
        C = euclidean_distance(points[0], points[3])
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return np.clip(ear, self.min_ear, self.max_ear)

    def calculate_mar(self, mouth_points):
        A = euclidean_distance(mouth_points[13], mouth_points[19])
        B = euclidean_distance(mouth_points[14], mouth_points[18])
        C = euclidean_distance(mouth_points[15], mouth_points[17])
        D = euclidean_distance(mouth_points[12], mouth_points[16])
        return (A + B + C) / (3.0 * D) if D > 0 else 0.0

    def eye_aspect_ratio_variance(self, ear_history, window_size=30):
        if len(ear_history) < window_size:
            return 0.0
        return float(np.var(ear_history[-window_size:]))

    def extract_eye_roi(self, gray, eye_points, margin=10):
        pts = np.array(eye_points, dtype=np.int32)
        x, y, w, h = manual_bounding_rect(pts)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2 * margin)
        h = min(gray.shape[0] - y, h + 2 * margin)
        return gray[y:y + h, x:x + w]

    def apply_canny_on_eye(self, gray, eye_points, low=None, high=None):
        roi = self.extract_eye_roi(gray, eye_points)
        if roi.size == 0:
            return np.zeros((10, 10), dtype=np.uint8)
        edges = manual_canny(roi, sigma=0.8, low=low, high=high)
        return edges

    def detect_iris_by_contour(self, eye_edges):
        if eye_edges.size == 0:
            return None
        area = manual_largest_contour_area(eye_edges)
        return area

    def calculate_head_pose(self, shape_np):
        left_eye_center = np.mean(shape_np[36:42], axis=0)
        right_eye_center = np.mean(shape_np[42:48], axis=0)
        nose_bridge = shape_np[27]
        nose_tip = shape_np[30]
        left_mouth = shape_np[48]
        right_mouth = shape_np[54]

        eye_center = (left_eye_center + right_eye_center) / 2.0
        mouth_center = (left_mouth + right_mouth) / 2.0

        eye_vector = right_eye_center - left_eye_center
        roll_angle = degrees(atan2(eye_vector[1], eye_vector[0]))

        face_height = euclidean_distance(eye_center, mouth_center)
        nose_height = euclidean_distance(nose_bridge, nose_tip)

        pitch_ratio = nose_height / face_height if face_height > 0 else 0
        pitch_angle = (pitch_ratio - 0.35) * 180

        return abs(roll_angle), abs(pitch_angle), pitch_ratio

    def reset_display(self):
        import cv2
        cv2.destroyAllWindows()
        logger.info("Reset hiển thị")
