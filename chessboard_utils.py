import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple

# Type aliases
Point = Tuple[int, int]
Line = Tuple[Point, Point]

LABELS_MAP = {
    '0': 0, 0: '0',
    'k': 1, 1: 'k',
    'q': 2, 2: 'q',
    'r': 3, 3: 'r',
    'b': 4, 4: 'b',
    'n': 5, 5: 'n',
    'p': 6, 6: 'p',
    'K': 7, 7: 'K',
    'Q': 8, 8: 'Q',
    'R': 9, 9: 'R',
    'B': 10, 10: 'B',
    'N': 11, 11: 'N',
    'P': 12 , 12: 'P',
}

def detect_lines(image: np.ndarray, lines_threshold: int = 600, vertical_error: float = 0.5,
                 horizontal_error: float = 0.5) -> dict[str, list[Line]]:
    """
    Detect the lines in the image using the HoughLines function and filter them based on the error values given as
    parameters to the function (in degrees) and return them in a dictionary with the key 'vertical' and 'horizontal'
    
    :param image: The image to detect the lines in
    :param lines_threshold: The threshold for the HoughLines function
    :param vertical_error: The error for the vertical lines (in degrees)
    :param horizontal_error: The error for the horizontal lines (in degrees)
    :return: A dictionary containing the horizontal and vertical lines
    """
    lines = cv2.HoughLines(image, 2, np.pi / 180, lines_threshold, None, 0, 0)
    if lines is None:
        return None

    # Filter lines based on angle and threshold values
    filtered_lines = {'vertical': [], 'horizontal': []}
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        angle = np.arctan2(y0, x0) * 180.0 / np.pi

        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        if not (abs(angle) < (90-vertical_error) or
            abs(angle) > (90+vertical_error)):
            filtered_lines['vertical'].append((pt1, pt2))
        elif not (abs(angle) < (horizontal_error) or
            abs(angle) < vertical_error or abs(angle) > 1 + vertical_error): # Not sure if this is correct
            filtered_lines['horizontal'].append((pt1, pt2))

    return filtered_lines

def get_intersection_points(lines: dict[str, list[Line]]) -> list[Point]:
    """
    Calculate the intersection points of all the lines in the dictionary
    and return them in a list of tuples containing the x and y coordinates
    
    :param lines: A dictionary containing the horizontal and vertical lines
    :return: A list of tuples containing the intersection points
    """
    if "horizontal" not in lines or "vertical" not in lines:
        raise ValueError("Input dictionary must contain keys 'horizontal' and 'vertical'")
    
    intersection_points = []
    for horizontal_line in lines["horizontal"]:
        for vertical_line in lines["vertical"]:
            # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
            x1, y1 = horizontal_line[0][0], horizontal_line[0][1]
            x2, y2 = horizontal_line[1][0], horizontal_line[1][1]
            x3, y3 = vertical_line[0][0], vertical_line[0][1]
            x4, y4 = vertical_line[1][0], vertical_line[1][1]
            
            x = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
            x /= (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            
            y = (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)
            y /= (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            
            intersection_points.append((int(x), int(y)))
    
    return intersection_points

def merge_points(points: list[Point], threshold_x: int = 10, threshold_y: int = 10) -> np.ndarray:
    """
    Merge points that are close to each other based on the threshold values given as parameters to the function.

    :param points: The points to merge.
    :param threshold_x: The threshold for the x axis.
    :param threshold_y: The threshold for the y axis.
    :return: An array of tuples containing the merged points.
    """
    # Initialize points_merged with the first point
    points_merged = [points[0]]
    
    for p in points[1:]:
        found = False
        for i, mp in enumerate(points_merged):
            if (abs(mp[0] - p[0]) < threshold_x and
                abs(mp[1] - p[1]) < threshold_y):
                points_merged[i] = ((mp[0] + p[0]) / 2,
                                    (mp[1] + p[1]) / 2)
                found = True
                break
        if not found:
            points_merged.append(p)
    # Sort points by y coordinate
    points_merged = sorted(points_merged, key=lambda x: x[1])
    # Sort every row by x coordinate
    for i in range(0, 9):
        points_merged[i*9:(i+1)*9] = sorted(points_merged[i*9:(i+1)*9], key=lambda x: x[0])

    return np.array(points_merged, dtype=np.int32)

def get_chessboard_corners(points: list[Point]) -> list[Point]:
    """
    Get the list of points that are corners of chessboard squares.
    List of points can be imaged as a 9x9 matrix, where each cell is a square on the chessboard.
    The function returns the upper left and lower right corners of each square.

    :param points: The points to get the corners from. Must contain only 81 points
    :return: A list of tuples containing the upper left and lower right corners of each square
    """
    squares = []
    for i, point in enumerate(points):
        pt1 = point
        pt2 = None if i in (8, 17, 26, 35, 44, 53, 62, 71, 80) or i > 70 else points[i + 10]
        if pt2 is not None:
            squares.append((pt1, pt2))
    return squares

def long_fen_to_short(fen: str) -> str:
    """
    Convert a long FEN string to a short FEN string.
    Example: 0r00000000q00k00000R00000P00000R00K00000000000000000000000000000 -> 1r6-2q2k2-3R4-1P5R-2K5-8-8-8

    :param fen: long FEN string
    :return: short FEN string
    """
    short_fen = ''
    for i in range(8):
        row = fen[i*8:(i+1)*8]
        empty_squares = 0
        for piece in row:
            if piece.isdigit():
                empty_squares += 1
            else:
                if empty_squares > 0:
                    short_fen += str(empty_squares)
                    empty_squares = 0
                short_fen += piece

        if empty_squares > 0:
            short_fen += str(empty_squares)

        if i < 7:
            short_fen += '-'

    return short_fen

def generate_fen(squares: list, model: torch.nn.Module, transform: transforms.Compose, show: bool = False) -> str:
    """
    Generate a FEN string from a list of squares

    :param squares: list of squares
    :param model: PyTorch model used to classify squares
    :param transform: PyTorch transform used to preprocess squares images
    :return: FEN string
    """

    with torch.no_grad():
        board = []
        for square in squares:
            square_tensor = transform(square).unsqueeze(0).to('cuda')
            prediction = model(square_tensor)
            piece_type = np.argmax(prediction.cpu().detach().numpy())
            label = LABELS_MAP.get(piece_type)
            if label is None:
                raise ValueError("Failed to classify square")
            board.append(label)
            if show:
                cv2.imshow(f'Prediction: {label}', square[0].cpu().numpy().transpose(1, 2, 0))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    fen = long_fen_to_short(''.join(board))

    return fen