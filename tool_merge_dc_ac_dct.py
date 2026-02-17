import os
import cv2
import numpy as np
from pathlib import Path

def add_images_from_paths(strDcPath, strAcPath, strOutputPath):
    # 确保输出目录存在
    if not os.path.exists(strOutputPath):
        os.makedirs(strOutputPath)

    # 支持的图片扩展名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 遍历strDcPath中的所有文件
    for dc_filename in os.listdir(strDcPath):
        # 检查是否是图片文件
        if not dc_filename.lower().endswith(valid_extensions):
            continue

        # 构建完整的文件路径
        dc_image_path = os.path.join(strDcPath, dc_filename)
        ac_image_path = os.path.join(strAcPath, dc_filename)

        # 检查对应的AC图片是否存在
        if not os.path.exists(ac_image_path):
            print(f"Warning: No matching AC image found for {dc_filename}")
            continue

        # 读取DC和AC图片（灰度图）
        dc_image = cv2.imread(dc_image_path, cv2.IMREAD_GRAYSCALE)
        ac_image = cv2.imread(ac_image_path, cv2.IMREAD_GRAYSCALE)

        # 检查图片是否成功加载
        if dc_image is None or ac_image is None:
            print(f"Error: Failed to load images for {dc_filename}")
            continue

        # 检查图片尺寸是否匹配
        if dc_image.shape != ac_image.shape:
            print(f"Error: Image sizes do not match for {dc_filename}")
            continue

        # 确保图像尺寸是8的倍数（DCT需要）
        h, w = dc_image.shape
        if h % 8 != 0 or w % 8 != 0:
            print(f"Error: Image dimensions must be multiples of 8 for {dc_filename}")
            continue

        # 将图像转换为浮点型以进行DCT计算
        dc_image_float = dc_image.astype(np.float32)
        ac_image_float = ac_image.astype(np.float32)

        # 初始化DCT系数矩阵
        dct_dc = np.zeros_like(dc_image_float)
        dct_ac = np.zeros_like(ac_image_float)
        combined_dct = np.zeros_like(dc_image_float)

        # 分块进行DCT变换 (8x8)
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                # 对DC图像进行DCT
                block_dc = dc_image_float[i:i+8, j:j+8]
                dct_block_dc = cv2.dct(block_dc)
                dct_dc[i:i+8, j:j+8] = dct_block_dc

                # 对AC图像进行DCT
                block_ac = ac_image_float[i:i+8, j:j+8]
                dct_block_ac = cv2.dct(block_ac)
                dct_ac[i:i+8, j:j+8] = dct_block_ac

                # 提取DC图像的DC系数（左上角元素）和AC图像的AC系数
                combined_block = dct_block_ac.copy()  # 先复制AC的DCT系数
                combined_block[0, 0] = dct_block_dc[0, 0]  # 用DC图像的DC系数替换
                combined_dct[i:i+8, j:j+8] = combined_block

        # 进行逆DCT变换
        result_float = np.zeros_like(dc_image_float)
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = combined_dct[i:i+8, j:j+8]
                idct_block = cv2.idct(block)
                result_float[i:i+8, j:j+8] = idct_block

        # 将结果裁剪到0-255并转换为uint8
        result = np.clip(result_float, 0, 255).astype(np.uint8)

        # 构建输出路径
        output_path = os.path.join(strOutputPath, dc_filename)

        # 保存结果
        success = cv2.imwrite(output_path, result)
        if success:
            print(f"Successfully processed and saved: {dc_filename}")
        else:
            print(f"Error: Failed to save output for {dc_filename}")

# 使用示例
if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent
    strDcPath = str(root_path / "RelatedWrok" / "SHC" / "Result" / "output_dc")
    strAcPath = str(root_path / "RelatedWrok" / "SHC" / "Result" / "output_ac")
    strOutputPath = str(root_path / "RelatedWrok" / "SHC" / "Result" / "output")

    add_images_from_paths(strDcPath, strAcPath, strOutputPath)
