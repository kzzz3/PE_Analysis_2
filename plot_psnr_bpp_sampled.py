import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from pathlib import Path

index = 0
need = [0,0,199,0,0,0,0]

def calculate_bpp(img_path):
    """计算每像素比特数 (BPP)"""
    global index
    img = Image.open(img_path)
    img_size = os.path.getsize(img_path) * 8 + need[index]  # 文件大小（比特）
    index+=1
    width, height = img.size
    total_pixels = width * height
    bpp = img_size / total_pixels
    return bpp


def analyze_encrypted_images(original_path, encrypted_dir):
    """分析加密图像的 PSNR 和 BPP，并按 PSNR 从大到小排序"""
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"原始图像路径不存在: {original_path}")
    if not os.path.isdir(encrypted_dir):
        raise FileNotFoundError(f"加密图像文件夹不存在: {encrypted_dir}")

    # 读取原始灰度图像
    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        raise ValueError(f"无法加载原始图像: {original_path}")

    # 存储结果
    results = []

    # 遍历加密图像文件夹
    for filename in os.listdir(encrypted_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            encrypted_path = os.path.join(encrypted_dir, filename)

            # 读取加密灰度图像
            encrypted_img = cv2.imread(encrypted_path, cv2.IMREAD_GRAYSCALE)
            if encrypted_img is None:
                print(f"警告: 无法加载加密图像 {encrypted_path}")
                continue

            # 确保两张图像尺寸一致
            if encrypted_img.shape != original_img.shape:
                encrypted_img = cv2.resize(encrypted_img, (original_img.shape[1], original_img.shape[0]))

            # 使用 skimage 计算 PSNR
            psnr = peak_signal_noise_ratio(original_img, encrypted_img, data_range=255)
            bpp = calculate_bpp(encrypted_path)

            # 存储结果
            results.append({
                'filename': filename,
                'psnr': psnr,
                'bpp': bpp
            })

    # 按 PSNR 从大到小排序
    results.sort(key=lambda x: x['psnr'], reverse=True)

    return results


def sample_points(results, num_points=15):
    """从按 PSNR 排序后的结果中均匀采样指定数量的点"""
    total_points = len(results)
    if total_points <= num_points:
        return results  # 如果总点数少于或等于采样数，返回全部

    # 计算采样步长
    step = total_points / (num_points - 1) if num_points > 1 else 1
    sampled_results = []

    for i in range(num_points):
        idx = min(int(round(i * step)), total_points - 1)  # 确保索引不超过范围
        sampled_results.append(results[idx])

    return sampled_results


def plot_bpp_psnr(results):
    """绘制 BPP-PSNR 折线图并保存到当前文件夹，BPP为纵坐标，PSNR为横坐标"""
    # 从排序后的结果中均匀采样 15 个点
    sampled_results = sample_points(results, num_points=15)

    psnr_values = [r['psnr'] for r in sampled_results]
    bpp_values = [r['bpp'] for r in sampled_results]

    plt.figure(figsize=(10, 6))
    plt.plot(psnr_values, bpp_values, marker='o', linestyle='-', color='b')  # PSNR为x轴，BPP为y轴
    plt.title('BPP vs PSNR (15 Sampled Points)')
    plt.xlabel('PSNR (dB)')  # x轴改为PSNR
    plt.ylabel('BPP (Bits Per Pixel)')  # y轴改为BPP
    plt.grid(True)

    # 保存图像到当前文件夹
    save_path = os.path.join(os.getcwd(), 'bpp_psnr_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"折线图已保存到: {save_path}")

    # 显示图像
    plt.show()


def main():
    root_path = Path(__file__).resolve().parent.parent
    original_image_path = str(root_path / "RelatedWrok" / "SE" / "Result" / "input" / "barbara_gray.bmp")
    encrypted_folder_path = str(root_path / "RelatedWrok" / "TPE_ADE" / "Result" / "output_qf" / "90")

    try:
        # 分析加密图像并排序
        results = analyze_encrypted_images(original_image_path, encrypted_folder_path)

        # 打印结果
        print("文件名\t\tPSNR (dB)\tBPP")
        for r in results:
            print(fr"{r['bpp']:.4f} {r['psnr']:.3f} \\")

        # 绘制折线图并保存
        plot_bpp_psnr(results)

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
