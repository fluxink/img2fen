import os
import cv2
import numpy as np


class SquareDetector():

    
    def __init__(self, mode=0) -> None:
        self.mode = {
            0: 'v1',
            1: 'v2',
        }[mode]
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self._kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self._lines_threshold = 600
        self._lines = {
            'horizontal': [],
            'vertical': [],
        }
        self._lines_vertical_error = 0.5
        self._lines_horizontal_error = 0.5
        self._points_merge_threshold_x = 20
        self._points_merge_threshold_y = 12

    def _crop_squares_from_image(self, image, squares) -> list:
        cropped_squares = []
        for i, square in enumerate(squares):
            x1 = int(square[0][0])
            y1 = int(square[0][1])
            x2 = int(square[1][0])
            y2 = int(square[1][1])
            cropped_squares.append(image[y1:y2, x1:x2])
        return cropped_squares

    def _create_squares(self, points) -> list:
        assert len(points) == 81, 'Expected 81 points, got {}'.format(len(points))
        squares = []
        for i, point in enumerate(points):
            pt1 = point
            pt2 = None if i in (8, 17, 26, 35, 44, 53, 62, 71, 80) or i > 70 else points[i + 10]
            if pt2 is not None:
                squares.append((pt1, pt2))
        return squares

    def _load_image(self, path):
        image = cv2.imread(path)
        return image

    def _merge_neighbouring_points(self, points):
        """
        Merge neighbouring points
        """
        points_merged = []
        for p in points:
            if len(points) == 0:
                points_merged.append(p)
            else:
                found = False
                for i in range(0, len(points_merged)):
                    if (abs(points_merged[i][0] - p[0]) < self._points_merge_threshold_x and
                        abs(points_merged[i][1] - p[1]) < self._points_merge_threshold_y):
                        points_merged[i] = ((points_merged[i][0] + p[0]) / 2,
                                            (points_merged[i][1] + p[1]) / 2)
                        found = True
                        break
                if not found:
                    points_merged.append(p)
        # Sort points
        points_merged = sorted(points_merged, key=lambda x: x[0])
        points_merged = sorted(points_merged, key=lambda x: x[1])
        return points_merged

    def _calculate_intersection_points(self) -> list:
        """
        Calculate the intersection points of all the lines
        """
        intersection_points = []
        for h in self._lines['horizontal']:
            for v in self._lines['vertical']:
                # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
                x = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * (v[0][0] - v[1][0]) - (h[0][0] - h[1][0]) * (v[0][0] * v[1][1] - v[0][1] * v[1][0])
                x /= (h[0][0] - h[1][0]) * (v[0][1] - v[1][1]) - (h[0][1] - h[1][1]) * (v[0][0] - v[1][0])
                y = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * (v[0][1] - v[1][1]) - (h[0][1] - h[1][1]) * (v[0][0] * v[1][1] - v[0][1] * v[1][0])
                y /= (h[0][0] - h[1][0]) * (v[0][1] - v[1][1]) - (h[0][1] - h[1][1]) * (v[0][0] - v[1][0])
                intersection_points.append((x, y))
        return intersection_points

    def _detec_lines_v2(self, image):
        pass

    def _detec_lines_v1(self, image):
        lines = cv2.HoughLines(image, 2, np.pi / 180, self._lines_threshold, None, 0, 0)
        if lines is None:
            raise Exception('No lines detected')

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

            if not (abs(angle) < (90-self._lines_vertical_error) or
                abs(angle) > (90+self._lines_vertical_error)):
                self._lines['vertical'].append((pt1, pt2))
            elif not (abs(angle) < (self._lines_horizontal_error) or
                abs(angle) > (180-self._lines_horizontal_error)): # Not sure if this is correct
                self._lines['horizontal'].append((pt1, pt2))

    def _detec_lines(self, image):
        if self.mode == 'v1':
            return self._detec_lines_v1(image)
        elif self.mode == 'v2':
            return self._detec_lines_v2(image)

    def _process_image_v2(self, image):
        pass

    def _process_image_v1(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        canny = cv2.Canny(blur, 100, 100)
        dillation = cv2.dilate(canny, self._kernel, iterations=2)
        erode = cv2.erode(dillation, self._kernel, iterations=1)
        _, thresh = cv2.threshold(erode, 210, 240, cv2.THRESH_BINARY_INV)
        thresh = cv2.GaussianBlur(thresh, (5,5), 0)
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self._kernel, iterations=1)
        close = cv2.erode(close, self._kernel, iterations=1)
        close = cv2.dilate(close, self._kernel, iterations=2)
        close = cv2.Canny(close, 100, 100)
        close = cv2.dilate(close, self._kernel, iterations=1)
        close = cv2.GaussianBlur(close, (7,7), 0)
        close = cv2.filter2D(close, -1, self._kernel_sharpen)
        return close

    def _process_image(self, image):
        if self.mode == 'v1':
            return self._process_image_v1(image)
        elif self.mode == 'v2':
            return self._process_image_v2(image)

    def detect(self, image_path) -> list:
        """
        Detect the squares in the image of chessboard
        """
        image = self._load_image(image_path)
        image_processed = self._process_image(image)
        self._detec_lines(image_processed)
        points = self._calculate_intersection_points()
        points = self._merge_neighbouring_points(points)
        squares = self._create_squares(points)
        cropped_squares = self._crop_squares_from_image(image, squares)
        return cropped_squares