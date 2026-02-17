import os
from pathlib import Path

import cv2 as cv
import gurobipy as gp
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

arrZigZag = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                      [2, 4, 7, 13, 16, 26, 29, 42],
                      [3, 8, 12, 17, 25, 30, 41, 43],
                      [9, 11, 18, 24, 31, 40, 44, 53],
                      [10, 19, 23, 32, 39, 45, 52, 54],
                      [20, 22, 33, 38, 46, 51, 55, 60],
                      [21, 34, 37, 47, 50, 56, 59, 61],
                      [35, 36, 48, 49, 57, 58, 62, 63]])

def ZigzagScan(arrMatrix: np.ndarray) -> np.ndarray:
    nHeight, nWidth = arrMatrix.shape[:2]
    arrZigzagScaned = np.zeros((nHeight, nWidth, 8 * 8), dtype=np.float64)
    for nRow in range(nHeight):
        for nCol in range(nWidth):
            for i in range(8):
                for j in range(8):
                    arrZigzagScaned[nRow, nCol, arrZigZag[i, j]] = arrMatrix[nRow, nCol, i, j]
    return arrZigzagScaned


def ComputeContributionMatrix():
    a = np.zeros([8, 8, 8, 8], dtype=np.float64)
    c = np.zeros(8, dtype=np.float64)
    c[0], c[1:] = (1 / 8) ** 0.5, (2 / 8) ** 0.5
    c = c[:, None] * c[None, :]
    i = np.arange(8).astype(np.float64)
    cos = np.cos((i[:, None] + 0.5) * i[None, :] * np.pi / 8)
    a = c[None, None] * cos[:, None, :, None] * cos[None, :, None, :]
    return a

def PreProcess(strInputImagePath,nST):
    arrImgData = cv.imread(strInputImagePath, cv.IMREAD_GRAYSCALE).astype(np.float32) - 128
    nHeight, nWidth = arrImgData.shape

    arrDCT = np.zeros_like(arrImgData, dtype=np.float32)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrDCT[nRow:nRow + 8, nCol:nCol + 8] = cv.dct(arrImgData[nRow:nRow + 8, nCol:nCol + 8])

    arrMask = np.zeros_like(arrDCT, dtype=bool)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrZigZag[i, j] >= nST and arrZigZag[i, j] <= 40:
                        arrMask[nRow + i, nCol + j] = True
    return arrDCT,arrMask

def OptimizationBasedAttack(
        arrDct: np.ndarray,
        arrMask: np.ndarray
) -> np.ndarray:
    nHeight, nWidth = arrDct.shape

    # 计算像素贡献矩阵并分离已知和未知部分
    arrContributionMatrix = ComputeContributionMatrix().reshape(8 * 8, 8 * 8)

    # 构造模型
    modelOptimization = gp.Model("Optimization-based Attack")

    # 构造变量
    arrDiffValueHorizontal = modelOptimization.addMVar((nHeight, nWidth - 1), vtype=gp.GRB.CONTINUOUS)
    arrDiffValueVertical = modelOptimization.addMVar((nHeight - 1, nWidth), vtype=gp.GRB.CONTINUOUS)
    arrVarMissingAc = modelOptimization.addMVar(arrMask.sum(), vtype=gp.GRB.CONTINUOUS, name="MissingAc")
    arrVarMissingAc = np.asarray(arrVarMissingAc.tolist())

    arrFullPixel = np.zeros((nHeight, nWidth), dtype=gp.LinExpr)

    nIndex = 0
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrMask[nRow + i, nCol + j]:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                    arrVarMissingAc[np.newaxis, nIndex:nIndex + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
                        nIndex += 1
                    else:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                    arrDct[nRow + i:nRow + i + 1, nCol + j:nCol + j + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)

    # 构造约束
    modelOptimization.addConstr(arrDiffValueHorizontal >= arrFullPixel[:, :-1] - arrFullPixel[:, 1:])
    modelOptimization.addConstr(arrDiffValueHorizontal >= arrFullPixel[:, 1:] - arrFullPixel[:, :-1])
    modelOptimization.addConstr(arrDiffValueVertical >= arrFullPixel[:-1, :] - arrFullPixel[1:, :])
    modelOptimization.addConstr(arrDiffValueVertical >= arrFullPixel[1:, :] - arrFullPixel[:-1, :])

    # 构造目标函数
    modelOptimization.setObjective(arrDiffValueHorizontal.sum() + arrDiffValueVertical.sum(), gp.GRB.MINIMIZE)

    modelOptimization.update()
    modelOptimization.optimize()

    arrRecoveredDct = np.zeros_like(arrVarMissingAc, dtype=np.float32)
    nIndex = 0
    for i in arrVarMissingAc:
        arrRecoveredDct[nIndex] = i.X
        nIndex += 1

    nIndex = 0
    arrFullPixel = np.zeros((nHeight, nWidth), dtype=np.float32)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrMask[nRow + i, nCol + j]:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                arrRecoveredDct[np.newaxis, nIndex:nIndex + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
                        nIndex += 1
                    else:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                arrDct[nRow + i:nRow + i + 1, nCol + j:nCol + j + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
    return (arrFullPixel + 128).clip(0, 255)

# AC Encryption
strImageName = "Peppers.bmp"
root_path = Path(__file__).resolve().parent.parent
strSrcImgPath = str(root_path / "Result" / "InputImage")
strDstImgPath = str(root_path / "Result" / "OutputImage" / "AcEncryption")

dictPSNR = {}
dictSSIM = {}
dictCipherPSNR = {}
dictCipherSSIM = {}
strRoot = strDstImgPath
arrQFs = os.listdir(strDstImgPath)
for QF in tqdm(arrQFs):
    dictPSNR[int(QF[QF.find("=") + 1:])] = {}
    dictSSIM[int(QF[QF.find("=") + 1:])] = {}
    dictCipherPSNR[int(QF[QF.find("=") + 1:])] = {}
    dictCipherSSIM[int(QF[QF.find("=") + 1:])] = {}

    strQFRoot = os.path.join(strRoot, QF)
    arrSTs = os.listdir(strQFRoot)
    for ST in tqdm(arrSTs):
        nQF = int(QF[QF.find("=") + 1:])
        nST = int(ST[ST.find("=") + 1:])

        strSTRoot = os.path.join(strQFRoot, ST)

        strPlainImagePath = os.path.join(strSrcImgPath, strImageName)
        strCipherImagePath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + ".jpg")
        strOutputImagePath = os.path.join(strSTRoot, strImageName[:strImageName.rfind(".")] + "_OA.bmp")

        if not os.path.exists(strOutputImagePath):
            #针对CipherImage 进行优化攻击
            arrDCT,arrMask = PreProcess(strCipherImagePath,nST)
            arrRecoveredImage = OptimizationBasedAttack(arrDCT,arrMask)
            cv.imwrite(strOutputImagePath, arrRecoveredImage)


        # 计算图像psnr 和 ssim
        imgSrc = io.imread(strPlainImagePath)
        imgDst = io.imread(strOutputImagePath)
        imgCipher = io.imread(strCipherImagePath)
        psnr = peak_signal_noise_ratio(imgDst, imgSrc)
        ssim = structural_similarity(imgDst, imgSrc)

        cipher_psnr = peak_signal_noise_ratio(imgCipher, imgSrc)
        cipher_ssim = structural_similarity(imgCipher, imgSrc)

        dictPSNR[int(QF[QF.find("=")+1:])][int(ST[ST.find("=")+1:])] = psnr
        dictSSIM[int(QF[QF.find("=")+1:])][int(ST[ST.find("=")+1:])] = ssim

        dictCipherPSNR[int(QF[QF.find("=")+1:])][int(ST[ST.find("=")+1:])] = cipher_psnr
        dictCipherSSIM[int(QF[QF.find("=")+1:])][int(ST[ST.find("=")+1:])] = cipher_ssim

#draw the result
arrQFs = [ QF for QF in list(dictPSNR.keys())]
arrSTs = [ ST for ST in list(dictPSNR.values())[0]]
arrSTs.sort()

# for QF in arrQFs:
#     print("QF="+str(QF))
#     print("PSNR:")
#     for ST in arrSTs:
#         print(ST, round(dictCipherPSNR[QF][ST],6),"&",round(dictPSNR[QF][ST],6),r"\\")
#     print("SSIM:")
#     for ST in arrSTs:
#         print(ST, round(dictCipherSSIM[QF][ST],6),"&",round(dictSSIM[QF][ST],6),r"\\")

for QF in arrQFs:
    print("QF="+str(QF))
    for ST in arrSTs:
        print(ST, round(dictCipherPSNR[QF][ST],6),"&",round(dictCipherSSIM[QF][ST],6),"&",round(dictPSNR[QF][ST],6),"&",round(dictSSIM[QF][ST],6),r"\\")
