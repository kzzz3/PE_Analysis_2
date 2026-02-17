import os
from pathlib import Path

from matplotlib import pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def PSNRandSSIM(strImgPath1, strImgPath2):
    imgDst = io.imread(strImgPath1)
    imgSrc = io.imread(strImgPath2)
    psnr = peak_signal_noise_ratio(imgDst, imgSrc)
    ssim = structural_similarity(imgDst, imgSrc, multichannel=True)
    return psnr, ssim

strSrcName = 'barbara_gray.bmp'
root_path = Path(__file__).resolve().parent.parent
strSrcPath = str(root_path / "Result" / "InputImage" / strSrcName)
strCipherDirPath = str(root_path / "Result" / "OutputImage" / "DcEncryption")

for QF in [30,50,70,90]:
    print(f'QF={QF}')

    listPSNR_cipher = []
    listSSIM_cipher = []
    for ST in range(0,1000):
        strCipherPath = os.path.join(strCipherDirPath, 'QF={}'.format(QF), strSrcName, f'{ST}.jpg')
        if not os.path.exists(strCipherPath):
            break

        psnr, ssim = PSNRandSSIM(strSrcPath, strCipherPath)
        listPSNR_cipher.append([ST, psnr])
        listSSIM_cipher.append([ST, ssim])

    print('PSNR:')
    for PSNR in listPSNR_cipher:
        print(PSNR[0], round(PSNR[1],3),r'\\')
    print('SSIM:')
    for SSIM in listSSIM_cipher:
        print(SSIM[0], round(SSIM[1],3),r'\\')




