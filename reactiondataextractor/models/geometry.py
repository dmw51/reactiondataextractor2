import math

import numpy as np


class Point:
    """Simple class for representing points in a uniform, image (row, col) manner"""
    def __init__(self, row, col):
        self.row = int(row)
        self.col = int(col)

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.row == other.row and self.col == other.col
        else:
            return self.row == other[1] and self.col == other[0]    # Assume a tuple

    def __hash__(self):
        return hash(self.row + self.col)

    def __str__(self):
        return f'{self.row, self.col}'

    def __repr__(self):
        return self.__str__()   # to de-clutter more complex objects

    def __iter__(self):
        return iter([self.row, self.col])

    def separation(self, other):
        """
        Calculates distance between self and another point
        :param Point other: another Point object
        :return float: distance between two Points
        """

        drow = self.row - other.row
        dcol = self.col - other.col
        return np.hypot(drow, dcol)


class Line:
    """This is a utility class representing a line in 2D defined by two points
    :param pixels: pixels belonging to a line
    :type pixels: list[Point]"""

    def __init__(self, pixels):
        self.pixels = pixels
        self.is_vertical = None
        self.slope, self.intercept = self.get_line_parameters()

    def __iter__(self):
        return iter(self.pixels)

    def __getitem__(self, index):
        return self.pixels[index]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pixels})'

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = value
        self.is_vertical = True if self.slope == np.inf or abs(self.slope) > 10 else False

    def get_line_parameters(self):
        """
        Calculates slope and intercept of ``line``
        :return: slope and intercept of the line
        :rtype: tuple
        """
        p1 = self.pixels[0]
        x1, y1 = p1.col, p1.row

        p2 = self.pixels[-1]  # Can be any two points, but non-neighbouring points increase accuracy of calculation
        x2, y2 = p2.col, p2.row

        delta_x = x2 - x1
        delta_y = y2 - y1

        if delta_x == 0:
            slope = np.inf
        else:
            slope = delta_y / delta_x

        intercept_1 = y1 - slope * x1
        intercept_2 = y2 - slope * x2
        intercept = (intercept_1 + intercept_2) / 2

        return slope, intercept

    def distance_from_point(self, other):
        """Calculates distance between the line and a point
        :param Point other: Point from which the distance is calculated
        :return float: distance between line and a point
        """
        # p1, *_, p2 = self.points
        p1 = self.pixels[0]
        x1, y1 = p1.col, p1.row
        p2 = self.pixels[-1]
        x2, y2 = p2.col, p2.row

        x0, y0 = other.col, other.row

        top = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1-y2*x1)
        bottom = np.sqrt((y2-y1)**2+(x2-x1)**2)

        return top/bottom

    @staticmethod
    def approximate_line(point_1, point_2):
        """
        Implementation of a Bresenham's algorithm. Approximates a straight line between ``point_1`` and ``point_2`` with
        pixels. Output is a list representing pixels forming a straight line path from ``point_1`` to ``point_2``
        """

        slope = Line([point_1, point_2]).slope  # Create Line just to get slope between two points

        if not isinstance(point_1, Point) and not isinstance(point_2, Point):
            point_1 = Point(row=point_1[1], col=point_1[0])
            point_2 = Point(row=point_2[1], col=point_2[0])

        if slope is np.inf:
            ordered_points = sorted([point_1, point_2], key=lambda point: point.row)
            return Line(
                [Point(row=row, col=point_1.col) for row in range(ordered_points[0].row, ordered_points[1].row)])

        elif abs(slope) >= 1:
            ordered_points = sorted([point_1, point_2], key=lambda point: point.row)
            return Line.bresenham_line_y_dominant(*ordered_points, slope)

        elif abs(slope) < 1:
            ordered_points = sorted([point_1, point_2], key=lambda point: point.col)
            return Line.bresenham_line_x_dominant(*ordered_points, slope)

    @staticmethod
    def bresenham_line_x_dominant(point_1, point_2, slope):
        """
        bresenham algorithm implementation when change in x is larger than change in y
        :param Point point_1: one endpoint of a line
        :param Point point_2: other endpoint of a line
        :param float slope: pre-calculated slope of the line
        :return: Line formed between the two points
        """
        y1 = point_1.row
        y2 = point_2.row
        deltay = y2 - y1
        domain = range(point_1.col, point_2.col + 1)

        deltaerr = abs(slope)
        error = 0
        y = point_1.row
        line = []
        for x in domain:
            line.append((x, y))
            error += deltaerr
            if error >= 0.5:
                deltay_sign = int(math.copysign(1, deltay))
                y += deltay_sign
                error -= 1
        pixels = [Point(row=y, col=x) for x, y in line]

        return Line(pixels=pixels)

    @staticmethod
    def bresenham_line_y_dominant(point_1, point_2, slope):
        """bresenham algorithm implementation when change in y is larger than change in x
        :param Point point_1: one endpoint of a line
        :param Point point_2: other endpoint of a line
        :param float slope: pre-calculated slope of the line
        :return: Line formed between the two points
        """

        x1 = point_1.col
        x2 = point_2.col
        deltax = x2 - x1
        domain = range(point_1.row, point_2.row + 1)

        deltaerr = abs(1 / slope)
        error = 0
        x = point_1.col
        line = []
        for y in domain:
            line.append((x, y))
            error += deltaerr
            if error >= 0.5:
                deltax_sign = int(math.copysign(1, deltax))
                x += deltax_sign
                error -= 1
        pixels = [Point(row=y, col=x) for x, y in line]

        return Line(pixels=pixels)


class OpencvToSkimageHoughLineAdapter:
    """Adapts output from probabilistic Hough line transform in openCV to conform to the output from scikit-image.
    This is equivalent to remodelling an array of shape [num_lines, 1, (x1,y1, x2, y2)] to a list of [(x1,y2), (x2,y2)]
    pairs containing `num_lines` tuples.
    """
    def __init__(self, cv_lines):
        self.cv_lines = cv_lines

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i == self.cv_lines.shape[0]:
            raise StopIteration

        cv_line = self.cv_lines[self._i]
        x1, y1, x2, y2 = cv_line.squeeze()
        self._i += 1
        return ((x1, y1), (x2, y2))

