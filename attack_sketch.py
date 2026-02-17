import os
import cv2
import numpy as np
from pathlib import Path

def load_image(image_path, grayscale=True):
    """读取指定位置的图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件未找到: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法加载图像，请检查文件路径或格式是否正确。")
    return image.astype(np.float32) - 128  # OpenCV 的 DCT 要求输入为浮点型

def save_image(image, base_path, suffix):
    """保存图像到源图像同级目录"""
    dir_name, file_name = os.path.split(base_path)
    name, ext = os.path.splitext(file_name)
    output_path = os.path.join(dir_name, f"{name}_{suffix}{ext}")
    cv2.imwrite(output_path, image + 128)
    print(f"图像已保存到: {output_path}")

def compute_dct_blocks(image, block_size=8):
    """将图像分割为不重叠的块并计算 DCT 系数"""
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.shape != (block_size, block_size):  # 忽略非完整块
                continue
            dct_block = cv2.dct(block)  # 使用 OpenCV 的 DCT
            blocks.append(dct_block)
    return np.array(blocks)

def quantize_blocks(blocks, quantization_matrix):
    """对 DCT 块进行量化"""
    quantized_blocks = [np.round(block / quantization_matrix).astype(np.int32) for block in blocks]
    return np.array(quantized_blocks)

def dequantize_blocks(blocks, quantization_matrix):
    """对量化后的 DCT 块进行反量化"""
    dequantized_blocks = [block * quantization_matrix for block in blocks]
    return np.array(dequantized_blocks)

def ncc(blocks):
    """实现 NCC 方法"""
    ncc_map = [np.count_nonzero(np.round(block)) for block in blocks]
    ncc_map = np.array(ncc_map)
    ncc_max = np.max(ncc_map)
    ncc_map_normalized = (255 * ncc_map / ncc_max).reshape(-1, 1)
    return ncc_map_normalized


def eac(blocks):
    """实现 EAC 方法"""
    eac_map = [np.sum(np.abs(block)) - np.abs(block[0, 0]) for block in blocks]
    eac_map = np.array(eac_map).astype(np.int32)
    mean_energy = np.mean(eac_map)

    # 保护机制：避免分母为零
    if mean_energy == 0:
        mean_energy = 1  # 防止除零

    eac_map_normalized = (255 * eac_map / mean_energy).reshape(-1, 1)
    eac_map_normalized = np.nan_to_num(eac_map_normalized, nan=0)  # 替换非法值
    return eac_map_normalized


def plz(blocks):
    """实现 PLZ 方法"""
    plz_map = [np.max(np.nonzero(block.flatten())) if np.any(block) else 0 for block in blocks]
    plz_map = np.array(plz_map).astype(np.int32)
    max_pos = np.max(plz_map)

    # 保护机制：避免分母为零
    if max_pos == 0:
        max_pos = 1  # 防止除零

    plz_map_normalized = (255 * plz_map / max_pos).reshape(-1, 1)
    plz_map_normalized = np.nan_to_num(plz_map_normalized, nan=0)  # 替换非法值
    return plz_map_normalized


def reshape_to_image(map_data, image_shape, block_size):
    """将计算结果扩展为原始图像大小，每个块的值填充对应的 8x8 区域"""
    h, w = image_shape
    small_h, small_w = h // block_size, w // block_size

    # 检查 map_data 的长度是否匹配块数量
    expected_length = small_h * small_w
    if map_data.size != expected_length:
        raise ValueError(f"数据大小与目标形状不匹配：期望 {expected_length}，实际 {map_data.size}")

    # 将 map_data 重塑为 (small_h, small_w)
    map_data = np.nan_to_num(map_data, nan=0).reshape(small_h, small_w)

    # 将每个像素值扩展为 block_size x block_size 大小的块
    expanded_image = np.repeat(np.repeat(map_data, block_size, axis=0), block_size, axis=1)

    # 确保输出大小与原始图像一致
    if expanded_image.shape != image_shape:
        raise ValueError(f"扩展后图像大小 {expanded_image.shape} 与目标大小 {image_shape} 不匹配")

    return expanded_image.astype(np.uint8)


# 主流程
if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent
    image_path = str(root_path / "Result" / "OutputImage" / "DcEncryption" / "QF=90" / "Cameraman.bmp" / "36.jpg")
    image = load_image(image_path, grayscale=True)

    # 计算 DCT 块
    block_size = 8
    dct_blocks = compute_dct_blocks(image, block_size)

    # 使用标准 JPEG 量化矩阵（亮度）
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # 对 DCT 块进行量化
    quantized_blocks = quantize_blocks(dct_blocks, quantization_matrix)

    # 反量化（可选，若需要恢复为原始 DCT 值）
    dequantized_blocks = dequantize_blocks(quantized_blocks, quantization_matrix)

    # NCC、EAC 和 PLZ（使用量化后的块）
    ncc_map = ncc(quantized_blocks)
    eac_map = eac(quantized_blocks)
    plz_map = plz(quantized_blocks)

    # 重塑结果为原图大小
    ncc_image = reshape_to_image(ncc_map, image.shape, block_size)
    eac_image = reshape_to_image(eac_map, image.shape, block_size)
    plz_image = reshape_to_image(plz_map, image.shape, block_size)

    # 保存结果到源图像同级目录
    save_image(ncc_image, image_path, "NCC")
    save_image(eac_image, image_path, "EAC")
    save_image(plz_image, image_path, "PLZ")
