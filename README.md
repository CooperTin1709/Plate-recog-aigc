# Plate Recognition (车牌识别)

使用传统计算机视觉方法实现的车牌检测与识别系统，包含两种方案。

## 两种方案

### 方案一：`main_v1.py`（手动实现）

基于 OpenCV 图像处理 + EasyOCR 识别，完整流程：

1. **预处理**：双边滤波去噪 → Canny 边缘检测 → 形态学闭运算
2. **定位**：轮廓检测 → 四边形近似 → 长宽比过滤 → 透视矫正
3. **OCR**：灰度化 → 2x 放大 → Otsu 二值化 → 紧裁剪 → EasyOCR 识别（readtext + recognize 双通道投票）

### 方案二：`main_hyperlpr.py`（基于 HyperLPR3）

使用 [HyperLPR3](https://github.com/szad670401/HyperLPR) 高性能中文车牌识别库，支持批量处理

结果导出至 `outputs/hyperlpr_results.csv`。

## 依赖

pip install opencv-python numpy easyocr hyperlpr3

## 测试结果

对 12 张不同省份、不同颜色车牌图片进行测试，方案二（HyperLPR3）准确率达 100%。

完整结果见 `outputs/hyperlpr_results.csv`。
