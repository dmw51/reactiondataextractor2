# -*- coding: utf-8 -*-
"""
Processing
==========
This module contains low level processing routines
author: Damian Wilary
email: dmw51@cam.ac.uk
"""
from __future__ import absolute_import
from __future__ import division

from collections.abc import Container
import copy
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.stats import mode
from configs import config
from reactiondataextractor.models.geometry import Line, Point, OpencvToSkimageHoughLineAdapter
from reactiondataextractor.models.segments import Rect, Panel, Figure, FigureRoleEnum


class DisabledNegativeIndices:
    """If a negative index is passed to an underlying sequence, then an empty element of appropriate type is returned.
    Slices including negative start indices are corrected to start at 0
    :param sequence: underlying sequence-type object
    :type sequence: sequence"""
    def __init__(self, sequence):
        self._sequence = sequence

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if idx.start < 0:
                idx = slice(0, idx.stop, idx.step)

        elif isinstance(idx, int):
            if self._sequence:
                type_ = type(self._sequence[0])
                if idx < 0:
                    return type_()

        return self._sequence[idx]


class PrettyFrozenSet(frozenset):
    """Frozenset with a pretty __str__ method; used for depicting output
    :param frozenset_: underlying frozenset
    :type frozenset_: frozenset"""

    def __new__(cls, frozenset_):
        obj = super().__new__(cls, frozenset_)
        return obj

    def __init__(self, frozenset_):
        self._frozenset_ = frozenset_
        super().__init__()

    def __str__(self):
        return ", ".join([str(elem) for elem in self._frozenset_])


class PrettyList(list):
    """list with a pretty __str__ method; used for depicting output
        :param list_: underlying list
        :type list_: list"""
    # def __new__(cls, list_):
    #     obj = super().__new__(cls, list_)
    #     return obj

    def __init__(self, list_):
        self._list = list_
        super().__init__(list_)

    def __str__(self):
        try:
            return '\n'.join([str(elem) for elem in self._list])
        except Exception as e:
            print()
            print()
            print()
            print(self._list)
            print(e)


def convert_greyscale(img):
    """
    Wrapper around skimage `rgb2gray` used for backward compatilibity
    :param np.ndarray img: input image
    :return np.ndarrat: image in grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop(img, left=None, right=None, top=None, bottom=None):
    """
    Crop image.
    Automatically limits the crop if bounds are outside the image.
    :param numpy.ndarray img: Input image.
    :param int left: Left crop.
    :param int right: Right crop.
    :param int top: Top crop.
    :param int bottom: Bottom crop.
    :return: Cropped image.
    :rtype: numpy.ndarray
    """

    height, width = img.shape[:2]

    left = max(0, left if left else 0)
    right = min(width, right if right else width)
    top = max(0, top if top else 0)
    bottom = min(height, bottom if bottom else width)
    out_img = img[top: bottom, left: right]
    return {'img': out_img, 'rectangle': Rect((top, left, bottom, right))}


def crop_rect(img, rect_boundary):
    """
    A convenience crop function that crops an image given boundaries as a Rect object
    :param np.ndarray img: input image
    :param Rect rect_boundary: object containing boundaries of the crop
    :return: cropped image
    :rtype: np.ndarray
    """
    left, right = rect_boundary.left, rect_boundary.right
    top, bottom = rect_boundary.top, rect_boundary.bottom
    return crop(img, left, right, top, bottom)


def binary_floodfill(fig):
    """ Converts all pixels inside closed contour to 1"""
    fig.img = ndi.binary_fill_holes(fig.img)
    return fig


def pixel_ratio(img, rect):
    """ Calculates the ratio of 'on' pixels to bounding box area for a rectangular patch of `img` bounded by `rect`.
    if the bounding box exceeds boundary of `img`, then area outside of `img` is treated as if it was background (values
    of 0)
    :param fig : Input binary Figure
    :param diag : Area to calculate pixel ratio
    :return ratio: Float detailing ('on' pixels / bounding box area)
    """
     # TODO: Adjust for panels extending beyond image boundary (pad with background pixels)

    # cropped_img = crop_rect(fig.img, rect)
    top, left, bottom, right = rect.coords
    cropped_img = crop(img, left, right, top, bottom)
    cropped_img = cropped_img['img']

    ones = np.count_nonzero(cropped_img)
    all_pixels = rect.area
    ratio = ones / all_pixels
    return ratio


# def binary_tag(fig):
#     """ Tag connected regions with pixel value of 1
#     :param fig: Input Figure
#     :returns fig: Connected Figure
#     """
#     fig = copy.deepcopy(fig)
#     fig.img, no_tagged = ndi.labels(fig.img)
#     return fig


def erase_elements(fig, elements, copy_fig=True):
    """
    Erase elements from an image on a pixel-wise basis. if no `pixels` attribute, the function erases the whole
    region inside the bounding box. Automatically assigns roles to ccs in the new figure based on the original.
    :param Figure fig: Figure object containing binarized image
    :param iterable of panels elements: list of elements to erase from image
    :return: copy of the Figure object with elements removed
    """
    if copy_fig:
        temp_fig = copy.deepcopy(fig)
    else:
        img_copy = copy.deepcopy(fig.img)
        raw_img = copy.deepcopy(fig.raw_img)
        img_detectron = copy.deepcopy(fig.img_detectron)
        temp_fig = Figure(img=img_copy, raw_img=raw_img, img_detectron=img_detectron)
        temp_fig._scaling_factor = fig.scaling_factor
        
    try:
        for panel in elements:
            panel.mask_off(temp_fig)
        temp_fig.set_connected_components()

    except AttributeError:
        for element in elements:
            temp_fig.img[element.top:element.bottom+1, element.left:element.right+1] = 0
            coords = [element.top, element.left, element.bottom, element.right]
            if fig.scaling_factor:
                scaled_element = np.rint(np.asarray(coords) / fig.scaling_factor).astype(np.int32)
                # list(map(lambda elem: np.rint(elem/fig.scaling_factor), coords), dtype=np.int32)
                bg_value = mode(temp_fig.img_detectron.ravel(), keepdims=False)[0]
                temp_fig.img_detectron[scaled_element[0]:scaled_element[2]+1, scaled_element[1]:scaled_element[3]+1] = bg_value
            else:
                temp_fig.img_detectron = None
    return temp_fig

def dilate_fig(fig, num_iterations):
    """
    Applies binary dilation to `fig.img` using a disk-shaped (3, 3) structuring element ``num_iterations`` times .
    :param Figure fig: Processed figure
    :param int num_iterations: number of iterations for dilation
    :return Figure: new Figure object
    """
    # selem = disk(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    f = cv2.dilate(fig.img, kernel, iterations=int(num_iterations))
    f = Figure(f, raw_img=fig.raw_img)
    return f
    # return Figure(binary_dilation(fig.img, selem), raw_img=fig.raw_img)


def is_slope_consistent(lines):
    """
    Checks if the slope of multiple lines is the same or similar. Useful when multiple lines found when searching for
    arrows
    :param [((x1,y1), (x2,y2))] lines: iterable of pairs of coordinates
    :return: True if slope is similar amongst the lines, False otherwise
    """
    if not all(isinstance(line, Line) for line in lines):
        pairs = [[Point(*coords) for coords in pair] for pair in OpencvToSkimageHoughLineAdapter(lines)]
        lines = [Line(pixels=pair, endpoints=pair) for pair in pairs]

    if all(abs(line.slope) > 10 for line in lines):  # very high/low slope == inf
        return True
    if all([line.slope == np.inf or line.slope == -np.inf for line in lines]):
        return True
    slopes = [line.slope for line in lines if abs(line.slope) != np.inf]
    if any([line.slope == np.inf or line.slope == -np.inf for line in lines]):
        slopes = [line.slope for line in lines if abs(line.slope) != np.inf]
    avg_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    abs_tol = 0.15
    rel_tol = 0.15

    tol = abs_tol if abs(avg_slope < 1) else rel_tol * avg_slope
    if std_slope > abs(tol):
        return False

    return True


def remove_small_fully_contained(connected_components):
    """
    Remove smaller connected components if their bounding boxes are fully enclosed within larger connected components
    :param iterable connected_components: set of all connected components
    :return: a smaller set of ccs without the enclosed ccs
    """
    enclosed_ccs = [small_cc for small_cc in connected_components if any(large_cc.contains(small_cc) for large_cc
                    in remove_connected_component(small_cc, connected_components))]
    # print(enclosed_ccs)
    refined_ccs = connected_components.difference(set(enclosed_ccs))
    return refined_ccs


def merge_rect(rect1, rect2):
    """ Merges rectangle with another, such that the bounding box enclose both
    :param Rect rect1: A rectangle
    :param Rect rect2: Another rectangle
    :return: Merged rectangle
    """

    left = min(rect1.left, rect2.left)
    right = max(rect1.right, rect2.right)
    top = min(rect1.top, rect2.top)
    bottom = max(rect1.bottom, rect2.bottom)
    return Rect(left=left, right=right, top=top, bottom=bottom)


def remove_connected_component(cc, connected_components):
    """
    Attempt to remove connected component and return the smaller set
    :param Panel cc: connected component to remove
    :param iterable connected_components: set of all connected components
    :return: smaller set of connected components
    """
    if not isinstance(connected_components, set):
        connected_components = set(copy.deepcopy(connected_components))
    connected_components.remove(cc)
    return connected_components


def isolate_patches(fig, to_isolate):
    """
    Creates an empty np.ndarray of shape `fig.img.shape` and populates it with pixels from `to_isolate`
    :param Figure|Crop fig: Figure object with binarized image
    :param iterable of Panels to_isolate: a set or a list of connected components to isolate
    :return: np.ndarray of shape `fig.img.shape` populated with only the isolated components
    """
    isolated = np.zeros(shape=fig.img.shape, dtype=np.uint8)

    for connected_component in to_isolate:
        top = connected_component.top
        bottom = connected_component.bottom
        left = connected_component.left
        right = connected_component.right
        isolated[top:bottom+1, left:right+1] = fig.img[top:bottom+1, left:right+1]

    fig = Figure(img=isolated, raw_img=fig.raw_img, eager_cc_init=False)
    fig.connected_components = to_isolate

    return fig

def HoughLinesP(*args, **kwargs):
    """Wrapper around cv2.HoughLinesP returning a sequence type if no lines have been found. Otherwise, it behaves
    exactly like cv2.HoughLinesP"""
    lines = cv2.HoughLinesP(*args, **kwargs)
    if lines is None:
        return []
    return lines


def intersect_rectangles(rect1, rect2):
    """
    Forms a new Rect object in the space shared by the two rectangles. Similar to intersection operation in set theory.
    :param Rect rect1: any Rect object
    :param Rect rect2: any Rect object
    :return: Rect formed by taking intersection of the two initial rectangles
    """
    left = max(rect1.left, rect2.left)
    right = min(rect1.right, rect2.right)
    top = max(rect1.top, rect2.top)
    bottom = min(rect1.bottom, rect2.bottom)
    return Rect(left, right, top, bottom)


def clean_output(text):
    """ Remove whitespace and newline characters from input text."""
    return text.replace('\n', '')


def flatten_list(data):
    """
    Flattens multi-level iterables into a list of elements
    :param [[..]] data: multi-level iterable data structure to flatten
    :return: flattened list of all elements
    """
    if len(data) == 0:
        return data

    if isinstance(data[0], Container):
        return flatten_list(data[0]) + flatten_list(data[1:])

    return data[:1] + flatten_list(data[1:])


def normalize_image(img):
    """
    Normalise image values to fit range between 0 and 1, and ensure it can be further proceseed. Useful e.g. after
    blurring operation
    :param np.ndarray img: analysed image
    :return: np.ndarray - image with values scaled to fit inside the [0,1] range
    """
    min_val = np.min(img)
    max_val = np.max(img)
    img -= min_val
    img /= (max_val - min_val)

    return img


def standardize(data):
    """
    Standardizes data to mean 0 and standard deviation of 1
    :param np.ndarray data: array of data
    :return np.ndarray: standardized data array
    """
    if data.dtype != 'float':
        data = data.astype('float')
    feature_mean = np.mean(data, axis=0)
    feature_std = np.std(data, axis=0)
    data -= feature_mean
    data /= feature_std
    return data


def find_minima_between_peaks(data, peaks):
    """
    Find deepest minima in ``data``, one between each adjacent pair of entries in ``peaks``, where ``data`` is a 2D
    array describing kernel density estimate. Used to cut ``data`` into segments in a way that allows assigning samples
    (used to create the estimate) to specific peaks.
    :param np.ndarray data: analysed data
    :param [int, int...] peaks: indices of peaks in ``data``
    :return: np.ndarray containing the indices of local minima
    """
    pairs = zip(peaks, peaks[1:])
    minima = []
    for pair in pairs:
        start, end = pair
        min_idx = np.argmin(data[1, start:end])+start
        minima.append(min_idx)

    return minima


def is_a_single_line(fig, panel, line_length):
    """
    Checks if the connected component is a single line by checking slope consistency of lines between randomly
    selected pixels
    :return:
    """

    lines = cv2.HoughLinesP(isolate_patches(fig, [panel]).img, 1, np.pi/180, minLineLength=line_length,
                            threshold=line_length)
    if lines is None:
        return False

    return is_slope_consistent(lines)


def skeletonize(fig):
    """
    A convenience function operating on Figure objects working similarly to skimage.morphology.skeletonize
    :param fig: analysed figure object
    :return: figure object with a skeletonised image
    """

    img = cv2.ximgproc.thinning(fig.img)
    fig_copy = copy.deepcopy(fig)
    fig_copy.img = img

    return fig_copy


def skeletonize_area_ratio(fig, panel):
    """ Calculates the ratio of skeletonized image pixels to total number of pixels
    :param fig: Input figure
    :param panel: Original _panel object
    :return: Float : Ratio of skeletonized pixels to total area (see pixel_ratio)
    """
    skel_fig = skeletonize(fig)
    return pixel_ratio(skel_fig, panel)


def mark_tiny_ccs(fig):
    """Marks all tiny connected components
    :param Figure fig: Analysed figure"""
    [setattr(cc, 'role', FigureRoleEnum.TINY) for cc in fig.connected_components if
     cc.area < np.percentile([cc.area for cc in fig.connected_components], 4) and cc.role is None]


def find_relative_directional_position(point1, point2):
    """Finds relative directional position between two points defined at the angle between +ve y-axis and the line
    defined by the two points.

    The direction is recovered from a dot product between the unit vector in the y axis and the vector defined by the
    two points"""

    v = point2[0] - point1[0], point2[1] - point1[1]
    j = (0, 1)  # (x, y) expected
    epsilon = 1e-5
    l_v = max(np.sqrt((v[0]**2 + v[1]**2)), epsilon)  # Avoid division by zero
    theta = np.arccos(np.dot(v, j)/(l_v * 1))
    theta = theta * 180 / np.pi
    return theta

def find_points_on_line(p0, t, distance):
    """Finds points on a line defined by point p0 and direction vector t, separated from p0 by a distance `distance`"""

    ## Any vector lying on a line defined by (p0, t0 can be described as p1 = p0 + at, where a is a real number
    ## for a given distance `distance` from p0, the two coefficients (There are two equidistant pts a1 and a2
    # are derived as
    a = distance / np.sqrt(np.sum(t[0]**2 + t[1]**2))

    return p0 + a * t, p0 - a * t


def lies_along_arrow_normal(arrow, obj):
    """Checks whether an object lies along arrow normal.

    Fit an min area rectangle to arrow contour.
    Form 4 points, 2 at end of arrows (or extending a bit further), 2 extending from a line normal to the
    arrow's bounding box (minimal), and check which is closest. Reclassify as conditions if obj is closer to
    a normal point"""

    # (x, y), (MA, ma), angle = cv2.minAreaRect(arrow.contour)
    min_rect = cv2.minAreaRect(arrow.contour[0])
    box_points = cv2.boxPoints(min_rect)
    diffs = [box_points[idx+1] - box_points[idx] for idx in range(3)] + [box_points[0] - box_points[-1]]
    box_segment_lengths = [np.sqrt(np.sum(np.power(x,2))) for x in diffs]
    largest_idx = np.argmax(box_segment_lengths)
    points = box_points[largest_idx], box_points[(largest_idx+1)%4]
    x_diff = points[1][0] - points[0][0]
    y_diff = points[1][1] - points[0][1]
    eps  = 1e-5
    dir_array = np.array([x_diff, y_diff])
    direction_arrow = dir_array / np.linalg.norm(dir_array)
    # angle = angle - 90  # Angle should be anti-clockwise relative to +ve x-axis

    # normal_angle = angle + 90
    center = np.asarray(arrow.panel.center)
    # direction = np.asarray([1, np.tan(np.radians(angle))])
    direction_normal = np.asarray([-1*direction_arrow[1], direction_arrow[0]])
    dist = box_segment_lengths[largest_idx] / 2
    p_a1, p_a2 = find_points_on_line(center, direction_arrow, distance=dist * 1.5)
    p_n1, p_n2 = find_points_on_line(center, direction_normal, distance=dist * .5)
    closest_pt = min([p_a1, p_a2, p_n1, p_n2], key=lambda pt: obj.center_separation(pt))
    ## Visualize created points
    # import matplotlib.pyplot as plt
    # plt.imshow(self.fig.img)
    # plt.scatter(p_a1[0], p_a1[1], c='r', s=3)
    # plt.scatter(p_a2[0], p_a2[1], c='r', s=3)
    # plt.scatter(p_n1[0], p_n1[1], c='b', s=3)
    # plt.scatter(p_n2[0], p_n2[1], c='b', s=3)
    # plt.show()

    if any(np.array_equal(closest_pt, p) for p in [p_n1, p_n2]):
        return True
    return False

def compute_ioa(panel1, panel2):
    """Compute intersection of two bounding boxes represented by Panels,
    then divide by the area of box stored inside panel1"""

    t1, l1, b1, r1 = panel1
    t2, l2, b2, r2 = panel2

    t = max(t1, t2)
    b = min(b1, b2)
    l = max(l1, l2)
    r = min(r1, r2)

    area_i = max(0, (r - l)) * max(0, (b - t))

    return area_i/panel1.area


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum([(x2-x1)**2 for x1, x2 in zip(p1, p2)]))
