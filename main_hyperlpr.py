import os
import re
import csv
import glob
import cv2
import hyperlpr3 as lpr3


def extract_plate_text(hyperlpr_result) -> str:
    """
    尽量兼容不同版本/不同返回结构：
    - 可能返回 list（里面是 dict / tuple / list）
    - 也可能直接包含字符串
    目标：从结果里提取最像车牌的一段字符串
    """
    # 常见车牌（含新能源）宽松正则：1中文 + 1字母 + 5~6位字母数字
    plate_re = re.compile(r"[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}")

    # 1) 如果本身就是字符串
    if isinstance(hyperlpr_result, str):
        m = plate_re.search(hyperlpr_result)
        return m.group(0) if m else hyperlpr_result.strip()

    # 2) 如果是列表，遍历里面的元素找字符串
    if isinstance(hyperlpr_result, list):
        candidates = []

        for item in hyperlpr_result:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                # 常见字段名猜测：code/plate/text
                for k in ("code", "plate", "text"):
                    if k in item and isinstance(item[k], str):
                        candidates.append(item[k])
            elif isinstance(item, (tuple, list)):
                for x in item:
                    if isinstance(x, str):
                        candidates.append(x)

        # 先找最像车牌的
        for s in candidates:
            m = plate_re.search(s)
            if m:
                return m.group(0)

        # 再退化：取最长的字符串
        candidates = [c.strip() for c in candidates if c and isinstance(c, str)]
        return max(candidates, key=len, default="")

    # 3) 其他类型：直接转字符串
    return str(hyperlpr_result)


def recognize_one_image(catcher, img_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        return ""

    # HyperLPR3：直接把整张图丢进去做检测+识别
    result = catcher(img)
    return extract_plate_text(result)


def main():
    # 你可以改成单张：img_list = ["images/test01.jpg"]
    img_list = []
    img_list += glob.glob("images/*.jpg")
    img_list += glob.glob("images/*.jpeg")
    img_list += glob.glob("images/*.png")

    if not img_list:
        print("images/ 目录下没有找到 jpg/jpeg/png 图片")
        return

    # 初始化一次（不要每张图都初始化）
    catcher = lpr3.LicensePlateCatcher()

    os.makedirs("outputs", exist_ok=True)
    out_csv = "outputs/hyperlpr_results.csv"

    rows = [("image_path", "plate")]
    for p in sorted(img_list):
        plate = recognize_one_image(catcher, p)
        print(p, "=>", plate if plate else "(未识别到)")
        rows.append((p, plate))

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

    print("完成，结果已保存：", out_csv)


if __name__ == "__main__":
    main()
