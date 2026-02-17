import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体，解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体（Windows常用中文字体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def process_images_and_plot(input_image_path, output_folder, max_strength):
    """主函数：处理已有加密图像并绘制折线图"""
    # 初始化结果列表
    psnr_values = []
    ssim_values = []
    strengths = []

    # 读取原始图像
    original_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        raise ValueError("无法读取原始图像文件")

    # 对不同强度的加密图像进行处理
    for strength in range(0,max_strength):
        if not os.path.exists(os.path.join(output_folder, f"{strength}.jpg")):
            break
        try:
            # 读取已有的加密图像
            encrypted_path = os.path.join(output_folder, f"{strength}.jpg")
            encrypted_img = cv2.imread(encrypted_path, cv2.IMREAD_GRAYSCALE)

            if encrypted_img is None:
                print(f"无法读取加密图像: {encrypted_path}")
                continue

            # 确保两张图像尺寸一致
            if original_img.shape != encrypted_img.shape:
                encrypted_img = cv2.resize(encrypted_img, (original_img.shape[1], original_img.shape[0]))

            # 使用库函数计算PSNR和SSIM
            psnr_value = psnr(original_img, encrypted_img, data_range=255)
            ssim_value = ssim(original_img, encrypted_img, data_range=255)

            # 存储结果
            strengths.append(strength)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

            print(f"强度 {strength}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")

        except Exception as e:
            print(f"处理强度 {strength} 时出错: {str(e)}")
            continue

    # 绘制折线图
    plt.figure(figsize=(12, 5))

    # PSNR折线图
    plt.subplot(1, 2, 1)
    plt.plot(strengths, psnr_values, 'b-o', label='PSNR')
    plt.xlabel('加密强度')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs 加密强度')
    plt.grid(True)
    plt.legend()

    # SSIM折线图
    plt.subplot(1, 2, 2)
    plt.plot(strengths, ssim_values, 'r-o', label='SSIM')
    plt.xlabel('加密强度')
    plt.ylabel('SSIM')
    plt.title('SSIM vs 加密强度')
    plt.grid(True)
    plt.legend()

    # 调整布局
    plt.tight_layout()

    # 保存折线图到当前文件夹
    output_file = os.path.join(os.getcwd(), 'psnr_ssim_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"折线图已保存至: {output_file}")

    # 显示折线图
    plt.show()


def main():
    root_path = Path(__file__).resolve().parent.parent
    input_image_path = str(root_path / "Result" / "InputImage" / "barbara_gray.bmp")
    output_folder = str(root_path / "Result" / "OutputImage" / "DcEncryption" / "QF=90" / "barbara_gray.bmp")

    try:
        process_images_and_plot(input_image_path, output_folder, max_strength=1000)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()
