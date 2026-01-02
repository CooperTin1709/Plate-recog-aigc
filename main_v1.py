import cv2
import numpy as np
import easyocr
import re


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    将四个点按：左上、右上、右下、左下 的顺序排序，便于透视矫正
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    对检测到的车牌四边形做透视变换，尽量拉正车牌
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    maxW = max(maxW, 1)
    maxH = max(maxH, 1)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped


def preprocess_for_plate_detection(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    基础预处理：灰度化、去噪、边缘检测
    返回：灰度图、边缘图
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 双边滤波：在保留边缘的同时降噪（比高斯更适合这种场景）
    gray_blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Canny 边缘
    edges = cv2.Canny(gray_blur, 50, 150)

    # 形态学闭运算：连接断裂边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, edges


def locate_plate_roi(bgr: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    通过轮廓寻找“可能是车牌的矩形区域”
    返回：车牌 ROI（彩色）、车牌四边形轮廓点（4x1x2）
    """
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    h, w = bgr.shape[:2]
    img_area = h * w

    # 按面积从大到小排序，优先检查更大的轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    best_quad = None
    for c in cnts[:30]:  # 只看前 30 个，减少误检和耗时
        area = cv2.contourArea(c)
        if area < img_area * 0.001:  # 太小的噪声轮廓直接跳过
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 车牌大多接近矩形：approx 点数为 4 先当作候选
        if len(approx) == 4:
            x, y, ww, hh = cv2.boundingRect(approx)
            aspect = ww / float(hh + 1e-6)

            # 简单的长宽比过滤（车牌通常较“扁”）
            if 2.0 <= aspect <= 6.0:
                best_quad = approx
                break

    if best_quad is None:
        return None, None

    # 透视矫正后裁剪 ROI（更利于 OCR）
    plate_roi = four_point_transform(bgr, best_quad)
    return plate_roi, best_quad


def preprocess_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    """
    给 OCR 用的预处理：增强对比度、二值化等
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # 放大一点有助于 OCR
    scale = 2.0
    gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

    # 轻微降噪
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 自适应阈值二值化（光照不均时更稳）
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    return bin_img


def normalize_plate_text(s: str) -> str:
    """
    对 OCR 输出做清洗：去空格/符号，转大写
    """
    s = s.strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", s)  # 保留：数字/大写字母/中文
    return s


def pick_best_plate(candidates: list[str]) -> str:
    """
    从若干候选字符串里挑一个最像“中文车牌”的
    常见：
    - 普通车牌：1个汉字 + 1个字母 + 5个字母/数字（总长度 7）
    - 新能源：1个汉字 + 1个字母 + 6个字母/数字（总长度 8）
    """
    # 典型车牌正则（宽松一点，避免误删）
    pattern = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")

    # 先找完全匹配的
    for t in candidates:
        if pattern.match(t):
            return t

    # 否则选一个“看起来最接近”的：包含中文 + 字母 + 数字，且长度在 6~10
    scored = []
    for t in candidates:
        if not (6 <= len(t) <= 10):
            continue
        has_cn = bool(re.search(r"[\u4e00-\u9fff]", t))
        has_letter = bool(re.search(r"[A-Z]", t))
        has_digit = bool(re.search(r"[0-9]", t))
        score = 0
        score += 2 if has_cn else 0
        score += 1 if has_letter else 0
        score += 1 if has_digit else 0
        score += (10 - abs(len(t) - 7)) * 0.1  # 更偏好长度接近 7/8
        scored.append((score, t))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1] if scored else ""


def main():
    img_path = "images/car1.jpg"

    # 1) 读取图片
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"读取失败：{img_path}")

    # 可选：把大图缩小一点，处理更快（不影响整体流程）
    h, w = bgr.shape[:2]
    if max(h, w) > 1200:
        ratio = 1200 / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

    # 2) 基本预处理（灰度/去噪/边缘）
    gray, edges = preprocess_for_plate_detection(bgr)

    # 3) 轮廓定位车牌 ROI
    plate_roi, plate_quad = locate_plate_roi(bgr, edges)

    # 如果没定位到，就退化为“全图 OCR”（效果可能较差，但保证程序能跑通）
    if plate_roi is None:
        plate_roi = bgr.copy()

    # 4) OCR 前预处理
    ocr_img = preprocess_for_ocr(plate_roi)

    # 5) EasyOCR 识别
    # 说明：加入 'en' 是为了更好识别字母数字；gpu=False 让没有显卡的环境也能跑
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

    # detail=1 时会返回 [bbox, text, conf]，这里保留置信度以便挑选
    results = reader.readtext(ocr_img, detail=1)

    # 6) 整理候选文本
    texts = []
    for bbox, text, conf in results:
        t = normalize_plate_text(text)
        if t:
            texts.append(t)

    plate_text = pick_best_plate(texts)

    print("OCR 原始候选：", texts)
    print
