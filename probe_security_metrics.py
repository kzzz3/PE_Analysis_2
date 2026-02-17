import os
import cv2
import numpy as np
from skimage import feature
from pathlib import Path


def calculate_npcr(plain_image, cipher_image):
    """
    计算 NPCR（Number of Pixel Change Rate）
    """
    height, width = plain_image.shape
    diff = plain_image != cipher_image
    npcr = np.count_nonzero(diff) / (height * width)
    return npcr


def calculate_uaci(plain_image, cipher_image):
    """
    计算 UACI（Unified Average Changing Intensity）
    """
    diff = np.abs(plain_image.astype(np.float32) - cipher_image.astype(np.float32))
    max_pixel = 255.0  # 假设 8-bit 图像
    uaci = np.mean(diff) / max_pixel
    return uaci


def calculate_edr(plain_image, cipher_image):
    """
    计算 EDR（Edge Differential Ratio）
    """
    # 使用 Canny 边缘检测
    plain_edges = feature.canny(plain_image)
    cipher_edges = feature.canny(cipher_image)

    # 计算分子和分母
    numerator = np.sum(np.abs(plain_edges.astype(np.float32) - cipher_edges.astype(np.float32)))
    denominator = np.sum(np.abs(plain_edges.astype(np.float32) + cipher_edges.astype(np.float32)))

    # 避免分母为零
    if denominator == 0:
        return 0.0

    # 计算EDR
    edr = (numerator / denominator) * 100
    return edr


def calculate_metrics(plain_images_path, cipher_images_path):
    """
    计算 NPCR、UACI 和 EDR 的平均值
    """
    # 获取明文和密文图像文件名列表
    plain_files = sorted(os.listdir(plain_images_path))
    cipher_files = sorted(os.listdir(cipher_images_path))

    # 提取文件名（不带扩展名）
    plain_names = [os.path.splitext(f)[0] for f in plain_files]
    cipher_names = [os.path.splitext(f)[0] for f in cipher_files]

    # 确保明文和密文文件名匹配
    common_names = set(plain_names).intersection(set(cipher_names))

    npcr_values = []
    uaci_values = []
    edr_values = []

    for name in common_names:
        # 支持多种图像格式
        plain_img_path = None
        cipher_img_path = None
        for ext in [".bmp", ".png", ".jpg", ".jpeg"]:
            if os.path.exists(os.path.join(plain_images_path, name + ext)):
                plain_img_path = os.path.join(plain_images_path, name + ext)
            if os.path.exists(os.path.join(cipher_images_path, name + ext)):
                cipher_img_path = os.path.join(cipher_images_path, name + ext)

        if plain_img_path is None or cipher_img_path is None:
            print(f"Warning: Image pair not found for {name}. Skipping.")
            continue

        # 读取图像
        plain_image = cv2.imread(plain_img_path, cv2.IMREAD_GRAYSCALE)
        cipher_image = cv2.imread(cipher_img_path, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("c", cipher_image)
        # cv2.waitKey(-1)

        if cipher_image is None:
            print(f"Warning: Unable to read image pair for {name}. Skipping.")

        # 如果图像读取失败，跳过
        if plain_image is None or cipher_image is None:
            print(f"Warning: Unable to read image pair for {name}. Skipping.")
            continue

        # 确保图像大小一致
        if plain_image.shape != cipher_image.shape:
            print(f"Warning: Image sizes do not match for {name}. Skipping.")
            continue

        # 计算指标
        npcr = calculate_npcr(plain_image, cipher_image)
        uaci = calculate_uaci(plain_image, cipher_image)
        edr = calculate_edr(plain_image, cipher_image)

        # 存储结果
        npcr_values.append(npcr)
        uaci_values.append(uaci)
        edr_values.append(edr)

        print(f"{name}: NPCR={npcr:.5f}, UACI={uaci:.5f}, EDR={edr:.5f}")

    # 计算平均值
    avg_npcr = np.mean(npcr_values) if npcr_values else 0
    avg_uaci = np.mean(uaci_values) if uaci_values else 0
    avg_edr = np.mean(edr_values) if edr_values else 0

    return avg_npcr, avg_uaci, avg_edr


# 示例用法
root_path = Path(__file__).resolve().parent.parent
plain_images_path = str(root_path / "RelatedWrok" / "SHC_another" / "Result" / "input")
cipher_images_path = str(root_path / "RelatedWrok" / "SHC_another" / "Result" / "output")

avg_npcr, avg_uaci, avg_edr = calculate_metrics(plain_images_path, cipher_images_path)
print(f"Average NPCR: {avg_npcr:.5f}")
print(f"Average UACI: {avg_uaci:.5f}")
print(f"Average EDR: {avg_edr:.5f}")
