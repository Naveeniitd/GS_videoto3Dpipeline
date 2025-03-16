import joblib
from PIL import Image, ImageStat
import cv2
import math
import os
import numpy as np
import pathlib
import pywt
from skimage.color import rgb2gray, rgba2rgb

model = joblib.load(os.path.join(pathlib.Path(__file__).parent.resolve(), "model.joblib"))


def brightness_calculation(img: np.ndarray):
    levels = np.linspace(0, 255, num=100)
    image_stats = ImageStat.Stat(Image.fromarray(img))
    red_mean, green_mean, blue_mean = image_stats.mean
    image_bright_value = math.sqrt(0.299 * (red_mean**2) + 0.587 * (green_mean**2) + 0.114 * (blue_mean**2))
    return np.digitize(image_bright_value, levels, right=True) / 100


def contrast_func(image, lower_percentile=1, upper_percentile=99):
    image = np.asanyarray(image)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = rgba2rgb(image)
        if image.shape[2] == 3:
            image = rgb2gray(image)
    dlimits = np.percentile(image, [0, 100])
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    return (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])


def blur_detect(img, threshold=35):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M, N = gray_img.shape
    gray_img = gray_img[: int(M / 16) * 16, : int(N / 16) * 16]

    LL1, (LH1, HL1, HH1) = pywt.dwt2(gray_img, "haar")
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, "haar")
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, "haar")

    E1 = np.sqrt(LH1**2 + HL1**2 + HH1**2)
    E2 = np.sqrt(LH2**2 + HL2**2 + HH2**2)
    E3 = np.sqrt(LH3**2 + HL3**2 + HH3**2)

    sizeM1, sizeN1 = 8, 8
    sizeM2, sizeN2 = sizeM1 // 2, sizeN1 // 2
    sizeM3, sizeN3 = sizeM2 // 2, sizeN2 // 2

    M1, N1 = E1.shape
    N_iter = (M1 // sizeM1) * (N1 // sizeN1)

    Emax1, Emax2, Emax3 = np.zeros(N_iter), np.zeros(N_iter), np.zeros(N_iter)

    count, x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0, 0
    Y_limit = N1 - sizeN1

    while count < N_iter:
        Emax1[count] = E1[x1 : x1 + sizeM1, y1 : y1 + sizeN1].max()
        Emax2[count] = E2[x2 : x2 + sizeM2, y2 : y2 + sizeN2].max()
        Emax3[count] = E3[x3 : x3 + sizeM3, y3 : y3 + sizeN3].max()

        if y1 == Y_limit:
            x1, y1 = x1 + sizeM1, 0
            x2, y2 = x2 + sizeM2, 0
            x3, y3 = x3 + sizeM3, 0
        else:
            y1, y2, y3 = y1 + sizeN1, y2 + sizeN2, y3 + sizeN3
        count += 1

    EdgePoint = (Emax1 > threshold) + (Emax2 > threshold) + (Emax3 > threshold)
    DAstructure = (Emax1 > Emax2) & (Emax2 > Emax3)
    RGstructure = (Emax1 < Emax2) & (Emax2 < Emax3)
    RSstructure = (Emax2 > Emax1) & (Emax2 > Emax3)
    BlurC = ((RGstructure | RSstructure) & (Emax1 < threshold)).astype(int)

    Per = np.sum(DAstructure) / np.sum(EdgePoint)
    BlurExtent = np.sum(BlurC) / max(np.sum(RGstructure) + np.sum(RSstructure), 1)

    return Per, BlurExtent


def varMaxLaplacian(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return lap.var(), lap.max()


def varMaxSobel(image, kernel=5):
    sob = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    return sob.var(), sob.max()


def predict(img_array=None, img_file=None):
    img = Image.open(img_file) if img_file else Image.fromarray(img_array)
    img_rgb = np.asarray(img.convert("RGB"))
    per, _ = blur_detect(img_rgb)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lapvar, lapmax = varMaxLaplacian(gray)
    sobvar, sobmax = varMaxSobel(gray)

    features = [lapmax, lapvar, sobmax, sobvar]
    blur_label = "Blur" if model.predict([features])[0] == 1 else "Not Blur"
    blur_score = round(model.predict_proba([features])[0][1], 2)

    brightness_score = brightness_calculation(img_rgb)
    contrast_score = round(contrast_func(gray), 2)

    return blur_label, blur_score, brightness_score, contrast_score, per
