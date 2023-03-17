import cv2
import numpy as np
import logging

def load_image(img: str) -> np.ndarray:
    """
    Load the image from the given path or from the given numpy array
    """
    if isinstance(img, np.ndarray):
        image = img
        return image

    image = cv2.imread(img)
    if image is None:
        raise Exception('Image not found')
    return image

def to_opencv2_image(img):
    """
    Convert an image to OpenCV2 format, regardless of input type.

    :param img: The image to convert. Can be a path to an image file, a bytes object, or a numpy array.
    :return: The OpenCV2 image.
    """
    # logging.debug(f'Loading image from {type(img)}')
    if isinstance(img, str):
        # Path to image file
        image = cv2.imread(img)
        if image is None:
            raise Exception('Image not found')
        return image
    elif isinstance(img, bytes):
        # Bytes object
        nparr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(img, np.ndarray):
        # Numpy array
        logging.debug(f'Image shape: {img.shape}')
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported number of dimensions for numpy array")
    else:
        raise TypeError("Unsupported image type")

def add_frame(image: np.ndarray, thickness=2) -> np.ndarray:
    """
    Add a frame to the image
    :param image: The image to add the frame to
    :param thickness: The thickness of the frame
    :return: The image with the frame
    """
    image = cv2.copyMakeBorder(image, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    image = cv2.copyMakeBorder(image, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image

def crop_image(image: np.ndarray, zoom_size: int) -> list[np.ndarray]:
    """
    Crop the image into 64 squares
    :param image: The image to crop
    :param zoom_size: The size of the zoom
    :return: A list of the cropped squares
    """
    height, width = image.shape[:2]
    crop_size = image.shape[0] // 8

    cropped_squares = []
    for row in range(8):
        for col in range(8):
            x = col * crop_size - zoom_size
            y = row * crop_size - zoom_size
            w = crop_size + zoom_size * 2
            h = crop_size + zoom_size * 2

            if x < 0:
                w += x
                x = 0
            if y < 0:
                h += y
                y = 0
            if x + w > width:
                w = width - x
            if y + h > height:
                h = height - y

            cropped_square = image[y:y+h, x:x+w]
            cropped_squares.append(cropped_square)

    return cropped_squares

def crop_squares(image, squares: list[tuple[int, int]]) -> list[np.ndarray]:
    """
    Crop the squares from the image
    :param image: The image to crop the squares from
    :param squares: The list of corners of the squares to crop from the image (upper left and lower right)
    :return: A list of the cropped squares
    """
    cropped_squares = []
    for square in squares:
        cropped_squares.append(image[square[0][1]:square[1][1], square[0][0]:square[1][0]])
    return cropped_squares

def show_image(image):
    cv2.imshow('image', image)

def show_lines(image, lines):
    for plane in lines.values():
        for line in plane:
            cv2.line(image, line[0], line[1], (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('lines', image)

def show_points(image, points):
    for i, point in enumerate(points):
        cv2.circle(image, point, 3, (i*4, 0, 255), -1)
    cv2.imshow('points', image)

def show_squares(image, squares):
    for i, square in enumerate(squares):
        cv2.rectangle(image, square[0], square[1], (255, i*3, 0), 2)
    cv2.imshow('squares', image)