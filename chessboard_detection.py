import cv2
import numpy as np
import logging
import chessboard_utils as cbu
import image_utils as iu
from image_profiles import PROFILES


def find_chessboards(image: str, show: bool = False) -> list[np.ndarray]:
    """
    Tries to find chessboards in an image and returns a list of the cropped chessboards.

    :param image: image to find chessboards in
    :return: list of chessboards
    """
    image = iu.load_image(image)
    # image = iu.to_opencv2_image(image)
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boards = []
    areas = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,cv2.arcLength(contour, True) * 0.02, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 5000:
            board = cv2.boundingRect(approx)
            x, y, w, h = board
            board_image =  image[y:y+h, x:x+w]
            # Boards must be square so check the ratio of width and height
            # Ratio must be approximately 1
            h, w, _ = board_image.shape
            if abs(w/h - 1) < 0.2:
                areas.append(w * h)
                boards.append(board_image)

    # Check area of boards
    # Boards must be approximately the same size
    # Get max area and remove boards that are less than 85% of that
    max_area = max(areas)
    boards = [board for board in boards if board.shape[0] * board.shape[1] > max_area * 0.85]

    if show:
        for board in boards:
            cv2.imshow('board', board)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return boards

def slice_squares(image: str, **kwargs) -> list[np.ndarray]:
    """
    Slice the image of chessboard into squares without preprocessing and return them as a list of numpy arrays.

    :param image_path: The path to the image to slice
    :param kwargs: The parameters to use for the slicing
    :return: A list of numpy arrays containing the cropped squares
    """
    # Set default values for the parameters
    if 'frame' not in kwargs:
        kwargs['frame'] = True
    if 'frame_thickness' not in kwargs:
        kwargs['frame_thickness'] = 5
    if 'zoom_size' not in kwargs:
        kwargs['zoom_size'] = 5

    logging.debug(f'Loading image from {type(image)}')
    image = iu.load_image(image)
    # image = iu.to_opencv2_image(image)

    if kwargs['frame']:
        logging.debug(f'Adding frame with thickness {kwargs["frame_thickness"]}')
        image = iu.add_frame(image, kwargs['frame_thickness'])

    logging.debug(f'Cropping image to squares with zoom size {kwargs["zoom_size"]}')
    squares = iu.crop_image(image, kwargs['zoom_size'])

    logging.debug(f'Successfully sliced image into {len(squares)} squares')
    return squares

def detect_squares(image: str, show: bool = False, **kwargs) -> list[np.ndarray]:
    """
    Detect the squares in the image of chessboard and return them as a list of numpy arrays.

    :param image_path: The path to the image to detect the squares in
    :param show: Whether to show the image with the detected squares
    :param kwargs: The parameters to use for the detection
    :return: A list of numpy arrays containing the cropped squares
    """
    # Set default values for the parameters
    if 'vertical_error' not in kwargs:
        kwargs['vertical_error'] = 0.5
    if 'horizontal_error' not in kwargs:
        kwargs['horizontal_error'] = 0.5
    if 'threshold_x' not in kwargs:
        kwargs['threshold_x'] = 15
    if 'threshold_y' not in kwargs:
        kwargs['threshold_y'] = 15
    if 'lines_threshold' not in kwargs:
        kwargs['lines_threshold'] = 400
    if 'frame' not in kwargs:
        kwargs['frame'] = True
    if 'frame_thickness' not in kwargs:
        kwargs['frame_thickness'] = 10
    if 'profile' not in kwargs or kwargs['profile'] is None:
        profiles = PROFILES
    elif 'profile' in kwargs and isinstance(kwargs['profile'], int):
        profiles = PROFILES[kwargs['profile']]
    elif 'profile' in kwargs and isinstance(kwargs['profile'], list):
        profiles = kwargs['profile']
    else:
        raise ValueError(f'Invalid profile: {kwargs["profile"]}')

    logging.debug(f'Using parameters: {kwargs}')
    logging.debug(f'Loading image from {type(image)}')
    image = iu.load_image(image)
    # image = iu.to_opencv2_image(image)

    if kwargs['frame']:
        logging.debug(f'Adding frame with thickness {kwargs["frame_thickness"]}')
        image = iu.add_frame(image, kwargs['frame_thickness'])

    for profile in profiles:
        lines_threshold = kwargs['lines_threshold']
        threshold_x = kwargs['threshold_x']
        threshold_y = kwargs['threshold_y']

        logging.debug(f"Using profile '{profile.__name__}'")
        image_processed = profile(image)

        for i in range(20):
            logging.debug(f'{profile.__name__} Attempt {i+1} of 20')
            try:
                lines = cbu.detect_lines(image_processed, lines_threshold )
                logging.debug(f'Found {len(lines["horizontal"])} | {len(lines["vertical"])} lines with threshold {lines_threshold}')
                if lines is None:
                    logging.warning(f'No lines found with threshold {lines_threshold}')
                    logging.debug(f'Decrease threshold by 10')
                    lines_threshold -= 10
                elif len(lines['horizontal']) < 9 or len(lines['vertical']) < 9:
                    logging.debug(f'Found less than 9 lines, decrease threshold by 10')
                    lines_threshold  -= 10
                elif len(lines['horizontal']) + len(lines['vertical']) > 34:
                    logging.debug(f'Found more than 34 lines, increase threshold by 10')
                    lines_threshold  += 10
                elif len(lines['horizontal']) + len(lines['vertical']) == 18:
                    break
                else:
                    break
            except Exception as e:
                logging.error(f'Error: {e} \nWhile detecting lines with threshold: {lines_threshold}')
                lines_threshold -= 10

        logging.debug(f'Calculate intersection points')
        points_1 = cbu.get_intersection_points(lines)
        logging.debug(f'Found {len(points_1)} intersection points')

        if len(points_1) < 81:
            logging.debug(f"Can't find 81 points with profile '{profile.__name__}'")
            continue

        # If the number of points above 81, try to increase the threshold otherwise decrease it
        # Make 5 attempts
        for i in range(5):
            logging.debug(f'Merge instersection points. Attempt {i+1} of 5')
            points = cbu.merge_points(points_1, threshold_x, threshold_y)
            if len(points) > 81:
                logging.debug(f'Get more than 81 points, increase threshold by 1')
                threshold_x += 1
                threshold_y += 1
            elif len(points) < 81:
                logging.debug(f'Get less than 81 points, decrease threshold by 1')
                threshold_x -= 1
                threshold_y -= 1
            elif len(points) == 81:
                break
        if len(points) == 81:
            logging.debug(f'Successfully found 81 points')
            break
        logging.debug(f"Can't get 81 points with profile '{profile.__name__}'")

    if len(points) != 81:
        if show:
            iu.show_image(image_processed)
            iu.show_lines(image.copy(), lines)
            iu.show_points(image.copy(), points)
            cv2.waitKey(0)
        logging.error(f'Number of points is {len(points)} instead of 81')
        raise Point81(f'Number of points is {len(points)} instead of 81')

    logging.debug(f'Get corners of chessboard squares')
    squares = cbu.get_chessboard_corners(points)

    logging.debug(f'Crop squares from image')
    cropped_squares = iu.crop_squares(image, squares)

    if show:
        iu.show_image(image_processed)
        iu.show_lines(image.copy(), lines)
        iu.show_points(image.copy(), points_1)
        iu.show_squares(image.copy(), squares)
        cv2.waitKey(0)

    logging.debug(f'Successfully detected 64 squares')
    return cropped_squares

class Point81(Exception):
    pass

# For testing
if __name__ == '__main__':
    from nn_utils import load_model, ChessCNNv3
    from chessboard_utils import generate_fen

    logging.basicConfig(level=logging.DEBUG)

    boards = find_chessboards('opencv/boards2.jpg', show=True)
    for i, board in enumerate(boards):
        detect_squares(board,
                       show=True,
                       vertical_error=0.5,
                       horizontal_error=0.5,
                       threshold_x=20,
                       threshold_y=20,
                       lines_threshold=400)
    

    # squares = detect_squares('opencv/1b1B3r-6k1-8-5P1p-1K6-8-8-3n1B2.jpeg', 
    #                          show=True,
    #                          vertical_error=0.5,
    #                          horizontal_error=0.5,
    #                          threshold_x=20,
    #                          threshold_y=20,
    #                          lines_threshold=400)

    # model, model_transform = load_model('model/model85.pth')

    # print(squares[-1].shape)
    # print(generate_fen(squares, model, model_transform))

    # for i, square in enumerate(squares):
    #     cv2.imwrite(f'opencv/squares_new/{i}.png', square)