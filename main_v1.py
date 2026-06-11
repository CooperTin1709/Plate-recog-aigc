import cv2
import numpy as np
import easyocr
import re
import os


def order_points(pts: np.ndarray) -> np.ndarray:
    """将四个点按：左上、右上、右下、左下 排序"""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """透视变换：尽量把车牌拉正"""
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

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def preprocess_for_plate_detection(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """基础预处理：灰度化、去噪、边缘检测"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(gray_blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gray, edges


def locate_plate_roi(bgr: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """轮廓法定位车牌 ROI"""
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    h, w = bgr.shape[:2]
    img_area = h * w
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    best_quad = None
    for c in cnts[:30]:
        area = cv2.contourArea(c)
        if area < img_area * 0.001:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, ww, hh = cv2.boundingRect(approx)
            aspect = ww / float(hh + 1e-6)
            if 2.0 <= aspect <= 6.0:
                best_quad = approx
                break

    if best_quad is None:
        return None, None

    plate_roi = four_point_transform(bgr, best_quad)
    return plate_roi, best_quad


def preprocess_for_ocr(plate_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OCR 预处理：灰度、放大、降噪、Otsu 二值化 + 反色"""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    scale = 2.0
    gray = cv2.resize(
        gray,
        (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)
    return th, th_inv


def normalize_plate_text(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", s)
    return s


def extract_texts(results) -> list[str]:
    texts = []
    for bbox, text, conf in results:
        t = normalize_plate_text(text)
        if t:
            texts.append(t)
    return texts


def pick_best_plate(candidates: list[str]) -> str:
    """宽松规则：5~8 位字母数字（先把后半段识别稳）"""
    pattern = re.compile(r"^[A-Z0-9]{5,8}$")
    for t in candidates:
        if pattern.match(t):
            return t
    return max(candidates, key=len, default="")


def crop_alnum_region(plate_bgr: np.ndarray) -> np.ndarray:
    """裁掉左侧部分（尽量避开中文省份字），保留字母数字区域"""
    h, w = plate_bgr.shape[:2]
    x1 = int(w * 0.10)  # 不准就微调：0.08~0.15
    roi = plate_bgr[:, x1:w]

    # 加白边，避免字符贴边被漏检
    roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return roi


def tight_crop_binary(img_bin: np.ndarray, pad: int = 25) -> np.ndarray:
    """对二值图做自动紧裁剪 + padding，减少边框/噪声干扰和贴边漏字"""
    ys, xs = np.where(img_bin > 0)  # 白色前景（th 是白字黑底）
    if len(xs) == 0 or len(ys) == 0:
        return img_bin

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = img_bin[y1:y2 + 1, x1:x2 + 1]

    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return crop


def ocr_by_recognize(reader, img_bin: np.ndarray, allow: str) -> str:
    """用 recognize 直接识别整行（车牌这种单行文本更稳）"""
    results = reader.recognize(
        img_bin,
        detail=1,
        allowlist=allow,
        decoder="beamsearch",
        beamWidth=5,
        contrast_ths=0.1,
        adjust_contrast=0.5,
    )

    texts = []
    for bbox, text, conf in results:
        t = normalize_plate_text(text)
        if t:
            texts.append(t)
    return max(texts, key=len, default="")


def main():
    print("程序已启动")
    print("当前工作目录：", os.getcwd())

    img_path = "images/test (1).jpg"

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"读取失败：{img_path}")

    h, w = bgr.shape[:2]
    if max(h, w) > 1200:
        ratio = 1200 / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

    gray, edges = preprocess_for_plate_detection(bgr)
    plate_roi, plate_quad = locate_plate_roi(bgr, edges)
    if plate_roi is None:
        plate_roi = bgr.copy()

    # 只取字母数字区域
    alnum_roi = crop_alnum_region(plate_roi)
    th, th_inv = preprocess_for_ocr(alnum_roi)

    # 二值图自动去边 + padding（降低边框干扰 + 减少漏字）
    th = tight_crop_binary(th, pad=25)
    th_inv = tight_crop_binary(th_inv, pad=25)

    # 保存调试图
    cv2.imwrite("debug_edges.jpg", edges)
    cv2.imwrite("debug_plate_roi.jpg", plate_roi)
    cv2.imwrite("debug_alnum_roi.jpg", alnum_roi)
    cv2.imwrite("debug_ocr_th_tight.jpg", th)
    cv2.imwrite("debug_ocr_th_inv_tight.jpg", th_inv)

    # 初始化一次 Reader
    reader = easyocr.Reader(["ch_sim", "en"], gpu=False)

    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # A：readtext（带检测）
    def run_ocr_readtext(img_bin: np.ndarray):
        return reader.readtext(
            img_bin,
            detail=1,
            allowlist=allow,
            decoder="beamsearch",
            beamWidth=5,
            rotation_info=[90, 180, 270],
            text_threshold=0.4,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=2.0,
            canvas_size=2560,
            add_margin=0.3,
        )

    results1 = run_ocr_readtext(th)
    results2 = run_ocr_readtext(th_inv)

    texts1 = extract_texts(results1)
    texts2 = extract_texts(results2)

    plate1 = pick_best_plate(texts1)
    plate2 = pick_best_plate(texts2)

    # B：recognize（不做检测，整行识别）
    rec1 = ocr_by_recognize(reader, th, allow)
    rec2 = ocr_by_recognize(reader, th_inv, allow)

    candidates = [plate1, plate2, rec1, rec2]
    final_plate = max(candidates, key=len, default="")

    print("OCR(th) 候选：", texts1, "=>", plate1)
    print("OCR(th_inv) 候选：", texts2, "=>", plate2)
    print("recognize(th)：", rec1)
    print("recognize(th_inv)：", rec2)
    print("最终车牌号：", final_plate if final_plate else "(未识别到)")

    if plate_quad is not None:
        vis = bgr.copy()
        cv2.drawContours(vis, [plate_quad], -1, (0, 255, 0), 2)
        cv2.imwrite("debug_plate_on_image.jpg", vis)


if __name__ == "__main__":
    main()
