# # -*- coding: utf-8 -*-
# """
# Actions
# ========
# This module contains important high level processing routines
# author: Damian Wilary
# email: dmw51@cam.ac.uk
# """
# import os
# import logging
# import numpy as np
#
# # from skimage.transform import probabilistic_hough_line
#

from reactiondataextractor.utils import isolate_patches, skeletonize, HoughLinesP
# from . import settings
from configs import config
#
# log = logging.getLogger('extract.actions')
#
# formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
# file_handler = logging.FileHandler(os.path.join(settings.ROOT_DIR, 'actions.log'))
# file_handler.setFormatter(formatter)
#
# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
#
# log.addHandler(file_handler)
# log.addHandler(ch)
#
# #TODO: change from skimage to opencv
import numpy as np

from reactiondataextractor.models.segments import Panel


def estimate_single_bond(fig):
    """Estimates length of a single bond in an image
    Uses a skeletonise image to find the number of lines of differing lengths. The single bond length is chosen using
    a graph of number of detected lines vs. the length of a line. The optimal value is where the change in number of
    lines as the length varies is greatest.
    :param Figure fig: analysed figure
    :return: approximate length of a single bond
    :rtype: int"""
    ccs = fig.connected_components
    # Get a rough bond length (line length) value from the two largest structures
    ccs = sorted(ccs, key=lambda cc: cc.area, reverse=True)
    estimation_fig = isolate_patches(fig, ccs[:3])
    biggest_cc = ccs[0]
    length_scan_param = 0.1 * min(biggest_cc.width, biggest_cc.height)
    lines = HoughLinesP(estimation_fig.img, rho=1, theta=np.pi/180, minLineLength=int(length_scan_param), threshold=15)
    lengths = []
    for l in lines:
        x1, y1, x2, y2 = l.squeeze()
        x = x2 - x1
        y = y2 - y1
        length = np.sqrt(x**2 + y ** 2)
        lengths.append(length)
    if len(lengths) > 100: # Handle cases where text dominates
        percentile = 85
    else:
        percentile = 50
    single_bond = np.percentile(lengths, percentile) ## This is too low for some figures, where there is a lot of text, but not enough diagrams
    # Could cluster lines depending on length into two clusters, and alternatively, choose length based on the larger cluster
    # length_scan_start = length_scan_param if length_scan_param > 50 else 50
    # min_line_lengths = np.linspace(length_scan_start, 8 * length_scan_start, 100)
    # num_lines = [(length, len(HoughLinesP(estimation_fig.img, rho=1, theta=np.pi/180, minLineLength=int(length), threshold=15)))
                 # for length in min_line_lengths]

    # Choose the value where the number of lines starts to drop most rapidly and assign it as the ''single_bond''
    # (single_bond, _), (_, _) = min(zip(num_lines, num_lines[1:]), key=lambda pair: pair[1][1] - pair[0][1])
    # the key function is difference in number of detected lines between adjacent pairs
    # return int(single_bond)
    config.ExtractorConfig.SOLID_ARROW_MIN_LENGTH = int(single_bond // 2)
    config.ExtractorConfig.SOLID_ARROW_THRESHOLD = int(single_bond // 2)
