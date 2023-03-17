import cv2
import numpy as np
from PIL import Image, ImageFilter

def add_frame(image, thickness=10):
    image = cv2.copyMakeBorder(image, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    image = cv2.copyMakeBorder(image, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image

def profile0(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    canny = cv2.Canny(blur, 100, 100)
    dillation = cv2.dilate(canny, kernel, iterations=2)
    erode = cv2.erode(dillation, kernel, iterations=1)
    _, thresh = cv2.threshold(erode, 210, 240, cv2.THRESH_BINARY_INV)
    thresh = cv2.GaussianBlur(thresh, (5,5), 0)
    image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=2)
    image = cv2.Canny(image, 100, 100)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (7,7), 0)
    image = cv2.filter2D(image, -1, kernel_sharpen)

    return image

def profile1(image):
    close = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(close, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(close, cv2.CV_16S, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    close = cv2.add(sobelx, sobely)

    return close

def profile2(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    close = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma = 4
    exp_table = np.array([255 * pow(i/255.0, gamma) for i in np.arange(0, 256)]).astype("uint8")

    # Apply exponential transform using LUT
    close = cv2.LUT(close, exp_table)
    close = cv2.medianBlur(close, 5)

    close = cv2.Canny(close, 200, 255)
    close = cv2.dilate(close, kernel, iterations=1)
    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations=1)

    return close

def profile3(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    image = Image.fromarray(image)
    image = image.filter(ImageFilter.BLUR)
    image = image.filter(ImageFilter.FIND_EDGES)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.Canny(image, 200, 255)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=1)

    return image

def profile4(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma = 2
    exp_table = np.array([255 * pow(i/255.0, gamma) for i in np.arange(0, 256)]).astype("uint8")

    # Apply exponential transform using LUT
    close = cv2.LUT(close, exp_table)

    sobelx = cv2.Sobel(close, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(close, cv2.CV_16S, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    close = cv2.add(sobelx, sobely)

    close = cv2.GaussianBlur(close, (3, 3), 0)
    close = cv2.erode(close, kernel, iterations=1)
    close = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=1)

    return close

def profile5(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    close = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(close, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(close, cv2.CV_16S, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    close = cv2.add(sobelx, sobely)
    _, close = cv2.threshold(close, 129, 255, cv2.THRESH_BINARY_INV)
    close = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.dilate(close, kernel, iterations=1)
    sobelx = cv2.Sobel(close, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(close, cv2.CV_16S, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    close = cv2.add(sobelx, sobely)

    return close

def profile6(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    close = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(close, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(close, cv2.CV_16S, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    close = cv2.add(sobelx, sobely)

    _, close = cv2.threshold(close, 30, 255, cv2.THRESH_BINARY_INV)
    close = cv2.GaussianBlur(close, (3, 3), 0)
    close = cv2.morphologyEx(close, cv2.MORPH_GRADIENT, kernel, iterations=1)

    return close

def pixelate1(image):
    image = Image.fromarray(image)
    w, h = image.size
    block_size = w // 10
    image = image.filter(ImageFilter.MedianFilter)

    image = image.resize((w // block_size, h // block_size), Image.NEAREST)
    image = image.resize((w, h), Image.NEAREST)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def pixelate2(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    height, width = image.shape[:2]
    # Desired "pixelated" size
    w, h = (64, 64)

    close = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gamma = 4
    exp_table = np.array([255 * pow(i/255.0, gamma) for i in np.arange(0, 256)]).astype("uint8")
    close = cv2.LUT(close, exp_table)

    close = cv2.medianBlur(close, 5)

    # Resize input to "pixelated" size
    temp = cv2.resize(close, (w, h), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    close = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    # close = cv2.equalizeHist(close)
    close = cv2.GaussianBlur(close, (3, 3), 0)
    # close = cv2.filter2D(close, -1, kernel_sharpen)
    close = cv2.Canny(close, 100, 100)
    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations=1)
    # close = cv2.dilate(close, kernel, iterations=1)

    return close

def save_image(image, name):
    cv2.imwrite('opencv/' + name + '.png', image)

PROFILES = [
    profile0,
    profile1,
    profile2,
    profile3,
    profile4,
    profile5,
    profile6,
]

# For testing
if __name__ == '__main__':
    image = cv2.imread('opencv/1b1B3r-6k1-8-5P1p-1K6-8-8-3n1B2.jpeg')
    assert image is not None
    image_frame = add_frame(image)
    image_copy = profile6(image_frame)
    # boards = find_chessboard(image)
    cv2.imshow('orig', image_frame)
    # cv2.imshow('board', boards[8])
    # save_image(boards[8], 'board')
    cv2.imshow('profile', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
