import os

import cv2 as cv
import gurobipy as gp
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

arrZigZag_8 = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                        [2, 4, 7, 13, 16, 26, 29, 42],
                        [3, 8, 12, 17, 25, 30, 41, 43],
                        [9, 11, 18, 24, 31, 40, 44, 53],
                        [10, 19, 23, 32, 39, 45, 52, 54],
                        [20, 22, 33, 38, 46, 51, 55, 60],
                        [21, 34, 37, 47, 50, 56, 59, 61],
                        [35, 36, 48, 49, 57, 58, 62, 63]])

def ComputeContributionMatrix():
    a = np.zeros([8, 8, 8, 8], dtype=np.float64)
    c = np.zeros(8, dtype=np.float64)
    c[0], c[1:] = (1 / 8) ** 0.5, (2 / 8) ** 0.5
    c = c[:, None] * c[None, :]
    i = np.arange(8).astype(np.float64)
    cos = np.cos((i[:, None] + 0.5) * i[None, :] * np.pi / 8)
    a = c[None, None] * cos[:, None, :, None] * cos[None, :, None, :]
    return a

def wavelet_5_3_1d(data):
    length = len(data)
    if length <= 1:
        return data.copy()

    half = length // 2  # 这里可以用 //，因为 length 是正整数
    temp = np.zeros(length, dtype=np.int16)

    # 计算高频系数 d[n]
    for n in range(half):
        left = data[2 * n]
        right = data[2 * n + 2] if (2 * n + 2 < length) else data[2 * n]  # 对称扩展
        pred = int((left + right) / 2)  # 改为截断除法
        temp[n + half] = data[2 * n + 1] - pred  # d[n]

    # 计算低频系数 a[n]
    for n in range(half):
        if n == 0:
            update = int(temp[half] / 2)  # 改为截断除法
        else:
            update = int((temp[half + n - 1] + temp[half + n]) / 4)  # 改为截断除法
        temp[n] = data[2 * n] + update  # a[n]

    return temp


def inverse_wavelet_5_3_1d_numeric(data):
    length = len(data)
    if length <= 1:
        return data.copy()

    half = length // 2
    temp = np.zeros(length, dtype=np.float64)  # 使用 float64 保持精度

    # 恢复偶数位置样本
    for n in range(half):
        if n == 0:
            update = data[half] / 2
        else:
            update = (data[half + n - 1] + data[half + n]) / 4
        temp[2 * n] = data[n] - update

    # 恢复奇数位置样本
    for n in range(half):
        left = temp[2 * n]
        right = temp[2 * n + 2] if (2 * n + 2 < length) else temp[2 * n]
        pred = (left + right) / 2
        temp[2 * n + 1] = data[n + half] + pred

    return temp

def inverse_wavelet_5_3_1d_gurobi(data):
    length = len(data)
    if length <= 1:
        return data.copy()

    half = length // 2
    temp = np.zeros(length, dtype=object)  # 使用 object 类型存储 gp.LinExpr

    # 恢复偶数位置样本
    for n in range(half):
        if n == 0:
            update = data[half] * 0.5  # 使用乘法避免类型问题
        else:
            update = (data[half + n - 1] + data[half + n]) * 0.25
        temp[2 * n] = data[n] - update

    # 恢复奇数位置样本
    for n in range(half):
        left = temp[2 * n]
        right = temp[2 * n + 2] if (2 * n + 2 < length) else temp[2 * n]
        pred = (left + right) * 0.5
        temp[2 * n + 1] = data[n + half] + pred

    return temp

def integer_53_wavelet_transform(image, nLevel):
    if image.dtype != np.int16:
        result = image.astype(np.int16)
    else:
        result = image.copy()

    rows, cols = result.shape

    for level in range(1, nLevel + 1):
        sub_rows = rows >> (level - 1)
        sub_cols = cols >> (level - 1)
        sub_rows = int(sub_rows / 2) * 2  # 改为截断除法并确保为偶数
        sub_cols = int(sub_cols / 2) * 2
        if sub_rows < 2 or sub_cols < 2:
            break

        # 行变换
        for i in range(sub_rows):
            result[i, :sub_cols] = wavelet_5_3_1d(result[i, :sub_cols])

        # 列变换
        for j in range(sub_cols):
            result[:sub_rows, j] = wavelet_5_3_1d(result[:sub_rows, j])

    return result


def inverse_integer_53_wavelet_transform(image, nLevel, use_gurobi=False):
    result = image.copy()
    rows, cols = result.shape

    for level in range(nLevel, 0, -1):
        sub_rows = rows >> (level - 1)
        sub_cols = cols >> (level - 1)
        sub_rows = int(sub_rows / 2) * 2
        sub_cols = int(sub_cols / 2) * 2
        if sub_rows < 2 or sub_cols < 2:
            continue

        # 根据 use_gurobi 参数选择逆变换函数
        inverse_func = inverse_wavelet_5_3_1d_gurobi if use_gurobi else inverse_wavelet_5_3_1d_numeric

        # 列逆变换
        for j in range(sub_cols):
            result[:sub_rows, j] = inverse_func(result[:sub_rows, j])

        # 行逆变换
        for i in range(sub_rows):
            result[i, :sub_cols] = inverse_func(result[i, :sub_cols])

    return result

def PreProcess(strDcIwtPath: str, strInputImagePath: str, nST: int, nIwtLevel: int):
    arrImgData = cv.imread(strInputImagePath, cv.IMREAD_GRAYSCALE).astype(np.float64) - 128
    nHeight, nWidth = arrImgData.shape

    arrDCT = np.zeros_like(arrImgData, dtype=np.float64)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrDCT[nRow:nRow + 8, nCol:nCol + 8] = cv.dct(arrImgData[nRow:nRow + 8, nCol:nCol + 8])

    arrMask = np.zeros_like(arrDCT, dtype=bool)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrZigZag_8[i, j] >= 1 and arrZigZag_8[i, j] <= 25:
                        arrMask[nRow + i, nCol + j] = True

    npIwtInfo = np.load(strDcIwtPath)
    arrDcIwt, nDcQuantStep = npIwtInfo['matrix'], npIwtInfo['integer']
    arrDcMask = np.zeros_like(arrDcIwt, dtype=bool)

    nDcHeight, nDcWidth = arrDcIwt.shape
    for level in range(2,nIwtLevel,1):
        block_size = nDcWidth >> level

        bWholeEncrypted = True
        # LH
        for nRow in range(0, block_size):
            for nCol in range(block_size, block_size*2):
                arrDcMask[nRow, nCol] = True
                if arrDcIwt[nRow, nCol]:
                    bWholeEncrypted = False

        # HL
        for nRow in range(block_size, block_size*2):
            for nCol in range(0, block_size):
                arrDcMask[nRow, nCol] = True
                if arrDcIwt[nRow, nCol]:
                    bWholeEncrypted = False

        # HH
        for nRow in range(block_size, block_size*2):
            for nCol in range(block_size, block_size*2):
                arrDcMask[nRow, nCol] = True
                if arrDcIwt[nRow, nCol]:
                    bWholeEncrypted = False

        if not bWholeEncrypted:
            break

    return arrDCT, arrMask, arrDcIwt, arrDcMask, nDcQuantStep


def OptimizationBasedAttack(
        nIWTLevel: int,
        arrDct: np.ndarray,
        arrAcMask: np.ndarray,
        arrDcIwt: np.ndarray,
        arrDcAcMask: np.ndarray,
        nDcQuantStep: int
) -> np.ndarray:
    nHeight, nWidth = arrDct.shape

    # 计算像素贡献矩阵并分离已知和未知部分
    arrContributionMatrix = ComputeContributionMatrix().reshape(8 * 8, 8 * 8)

    # 构造模型
    modelOptimization = gp.Model("Optimization-based Attack")
    modelOptimization.setParam('Threads', os.cpu_count())

    # 构造变量
    arrDiffValueHorizontal = modelOptimization.addMVar((nHeight, nWidth - 1), vtype=gp.GRB.CONTINUOUS)
    arrDiffValueVertical = modelOptimization.addMVar((nHeight - 1, nWidth), vtype=gp.GRB.CONTINUOUS)
    arrVarMissingAc = modelOptimization.addMVar(arrAcMask.sum(), vtype=gp.GRB.CONTINUOUS, name="MissingAc")
    arrVarMissingDcAc = modelOptimization.addMVar(arrDcAcMask.sum(), vtype=gp.GRB.CONTINUOUS, name="MissingDcAc")
    arrVarMissingAc = np.asarray(arrVarMissingAc.tolist())
    arrVarMissingDcAc = np.asarray(arrVarMissingDcAc.tolist())

    arrFullPixel = np.zeros((nHeight, nWidth), dtype=object)
    
    nIndex = 0
    nDcHeight, nDcWidth = arrDcIwt.shape
    arrFullDc = np.zeros(arrDcIwt.shape, dtype=object)
    arrRecoveredDcIwt = np.zeros(arrDcIwt.shape, dtype=object)
    for nRow in range(0, nDcHeight):
        for nCol in range(0, nDcWidth):
            arrRecoveredDcIwt[nRow, nCol] = arrDcIwt[nRow, nCol]
            if arrDcAcMask[nRow, nCol]:
                arrRecoveredDcIwt[nRow, nCol] += arrVarMissingDcAc[nIndex]
                nIndex += 1
    arrFullDc = inverse_integer_53_wavelet_transform(arrRecoveredDcIwt, nIWTLevel, use_gurobi=True)

    # # 将DcDct还原至Dct
    # for nRow in range(0, nHeight, 8):
    #     for nCol in range(0, nWidth, 8):
    #         arrFullPixel[nRow, nCol] = arrFullDc[nRow // 8, nCol // 8] * nDcQuantStep

    # 构造DCAC表达式
    arrDct = arrDct.astype(object)

    # 构造AC表达式
    nIndex = 0
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrDct[nRow, nCol] = arrFullDc[nRow // 8, nCol // 8] * nDcQuantStep
            for i in range(8):
                for j in range(8):
                    if arrAcMask[nRow + i, nCol + j]:
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

    arrRecoveredDcDct = np.zeros_like(arrVarMissingDcAc, dtype=np.float64)
    nIndex = 0
    for i in arrVarMissingDcAc:
        arrRecoveredDcDct[nIndex] = i.X
        nIndex += 1

    nIndex = 0
    nDcHeight, nDcWidth = arrDcIwt.shape
    arrFullDc = np.zeros(arrDcIwt.shape, dtype=np.float64)
    arrRecoveredDcIwt = np.zeros(arrDcIwt.shape, dtype=np.float64)
    for nRow in range(0, nDcHeight):
        for nCol in range(0, nDcWidth):
            arrRecoveredDcIwt[nRow, nCol] = arrDcIwt[nRow, nCol]
            if arrDcAcMask[nRow, nCol]:
                arrRecoveredDcIwt[nRow, nCol] += arrRecoveredDcDct[nIndex]
                nIndex += 1
    arrFullDc = inverse_integer_53_wavelet_transform(arrRecoveredDcIwt, nIWTLevel, use_gurobi=False)

    # 将DcDct还原至Dct
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            arrDct[nRow, nCol] = arrFullDc[nRow // 8, nCol // 8] * nDcQuantStep

    arrRecoveredDct = np.zeros_like(arrVarMissingAc, dtype=np.float64)
    nIndex = 0
    for i in arrVarMissingAc:
        arrRecoveredDct[nIndex] = i.X
        nIndex += 1

    
    arrDct = arrDct.astype(np.float64)
    nIndex = 0
    arrFullPixel = np.zeros((nHeight, nWidth), dtype=np.float64)
    for nRow in range(0, nHeight, 8):
        for nCol in range(0, nWidth, 8):
            for i in range(8):
                for j in range(8):
                    if arrAcMask[nRow + i, nCol + j]:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                arrRecoveredDct[np.newaxis, nIndex:nIndex + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
                        nIndex += 1
                    else:
                        arrFullPixel[nRow:nRow + 8, nCol:nCol + 8] += (
                                arrDct[nRow + i:nRow + i + 1, nCol + j:nCol + j + 1] @ arrContributionMatrix[:, i * 8 + j:i * 8 + j + 1].T).flatten().reshape(8, 8)
    return (arrFullPixel + 128).clip(0, 255)


# DC Encryption
# strImageName = "Peppers.bmp"
nIWTLevel = 4
strSrcImgPath = r"/home/zhouke/Result/InputImage"
strDstImgPath = r"/home/zhouke/Result/OutputImage/NewDcEncryption"

dictPSNR = {}
dictSSIM = {}
dictCipherPSNR = {}
dictCipherSSIM = {}
strRoot = strDstImgPath

arrQFs = sorted(os.listdir(strRoot), reverse=True)
for QF in tqdm(arrQFs):
    dictPSNR[int(QF[QF.find("=") + 1:])] = {}
    dictSSIM[int(QF[QF.find("=") + 1:])] = {}
    dictCipherPSNR[int(QF[QF.find("=") + 1:])] = {}
    dictCipherSSIM[int(QF[QF.find("=") + 1:])] = {}
    strQFRoot = os.path.join(strRoot, QF)

    for strImageName in os.listdir(strSrcImgPath):
        strImageRoot = os.path.join(strQFRoot, strImageName)

        dictPSNR[int(QF[QF.find("=") + 1:])][strImageName] = {}
        dictSSIM[int(QF[QF.find("=") + 1:])][strImageName] = {}
        dictCipherPSNR[int(QF[QF.find("=") + 1:])][strImageName] = {}
        dictCipherSSIM[int(QF[QF.find("=") + 1:])][strImageName] = {}

        arrSTImgs = os.listdir(strImageRoot)
        arrSTs = sorted([int(filename.split(".")[0]) for filename in arrSTImgs if filename.split(".")[0].isdigit() and filename.endswith(".jpg")])
        arrSTs = arrSTs[1:-1]
        
        # Select 5 evenly distributed STs plus the last one
        if len(arrSTs) > 5:
            selected_indices = np.linspace(0, len(arrSTs)-1, 5, dtype=int)  # 5 evenly spaced indices (excluding last)
            selected_STs = [arrSTs[i] for i in selected_indices]  # Add the last ST
        else:
            selected_STs = arrSTs  # If <=6 STs, process all
        
        for ST in selected_STs:
            strPlainImagePath = os.path.join(strSrcImgPath, strImageName)
            strDcIwtPath = os.path.join(strImageRoot, str(ST) + '.npy')
            strCipherImagePath = os.path.join(strImageRoot, str(ST) + '.jpg')
            strOutputImagePath = os.path.join(strImageRoot, str(ST) + "_OA.bmp")

            print(f"Processing {strImageName} with QF={QF[QF.find('=')+1:]} and ST={ST}...")
            if not os.path.exists(strOutputImagePath):
                # 针对CipherImage 进行优化攻击
                arrDCT, arrMask, arrDcIwt, arrDcMask, nDcQuantStep = PreProcess(strDcIwtPath, strCipherImagePath, ST, 4)
                arrRecoveredImage = OptimizationBasedAttack(4, arrDCT, arrMask, arrDcIwt, arrDcMask, nDcQuantStep)
                cv.imwrite(strOutputImagePath, arrRecoveredImage)

            # 计算图像psnr 和 ssim
            imgSrc = io.imread(strPlainImagePath)
            imgDst = io.imread(strOutputImagePath) if os.path.exists(strOutputImagePath) else None
            imgCipher = io.imread(strCipherImagePath)
            
            if imgDst is not None:
                psnr = peak_signal_noise_ratio(imgDst, imgSrc)
                ssim = structural_similarity(imgDst, imgSrc)
            else:
                psnr = ssim = 0  # or some default value
                
            cipher_psnr = peak_signal_noise_ratio(imgCipher, imgSrc)
            cipher_ssim = structural_similarity(imgCipher, imgSrc)

            dictPSNR[int(QF[QF.find("=") + 1:])][strImageName][ST] = psnr
            dictSSIM[int(QF[QF.find("=") + 1:])][strImageName][ST] = ssim
            dictCipherPSNR[int(QF[QF.find("=") + 1:])][strImageName][ST] = cipher_psnr
            dictCipherSSIM[int(QF[QF.find("=") + 1:])][strImageName][ST] = cipher_ssim

# Output the results
arrQFs = sorted(dictPSNR.keys())  # QF values are the top-level keys in dictPSNR
for QF in arrQFs:
    print(f"QF={QF}")
    for strImageName in dictPSNR[QF]:  # Image names are the second-level keys
        print(f"Image: {strImageName}")
        # Get all ST values and sort them in descending order
        arrSTs = sorted(dictPSNR[QF][strImageName].keys())
        
        # print("PSNR values:")
        # for ST in arrSTs:
        #     print(f"ST_r={ST}: {round(dictPSNR[QF][strImageName][ST], 6)} dB")
        
        # print("\nSSIM values:")
        # for ST in arrSTs:
        #     print(f"ST_r={ST}: {round(dictSSIM[QF][strImageName][ST], 6)}")
        
        print("\nComparison Table (Cipher vs Recovered):")
        print("ST_r | Cipher PSNR | Cipher SSIM | Recovered PSNR | Recovered SSIM")
        print("--------------------------------------------------------------")
        for ST in arrSTs:
            cipher_psnr = round(dictCipherPSNR[QF][strImageName][ST], 6)
            cipher_ssim = round(dictCipherSSIM[QF][strImageName][ST], 6)
            recovered_psnr = round(dictPSNR[QF][strImageName][ST], 6)
            recovered_ssim = round(dictSSIM[QF][strImageName][ST], 6)
            print(f"{ST} | {cipher_psnr} dB | {cipher_ssim} | {recovered_psnr} dB | {recovered_ssim}")
        
        print("\n" + "="*80 + "\n")