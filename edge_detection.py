import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


# 方法1：Canny边缘检测
def canny_edge_detection(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 检查是否成功加载图像
    if img is None:
        print("Error: Could not load image")
        return

    # 应用高斯模糊去噪
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny边缘检测
    # 参数：图像、最小阈值、最大阈值
    edges = cv2.Canny(blurred, 100, 200)

    # 显示结果
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')

    plt.show()

    # 保存结果
    cv2.imwrite('canny_edges.jpg', edges)


# 方法2：Sobel边缘检测
def sobel_edge_detection(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not load image")
        return

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Sobel边缘检测
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # x方向
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # y方向

    # 计算梯度幅度
    sobel_combined = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_combined = np.uint8(sobel_combined)

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')

    plt.show()

    # 保存结果
    cv2.imwrite('sobel_edges.jpg', sobel_combined)


# 主函数
if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent
    image_path = str(root_path / "RelatedWrok" / "TPE_ADE" / "Result" / "output" / "barbara_gray.jpg")

    print("Performing Canny Edge Detection...")
    canny_edge_detection(image_path)

    print("Performing Sobel Edge Detection...")
    sobel_edge_detection(image_path)
