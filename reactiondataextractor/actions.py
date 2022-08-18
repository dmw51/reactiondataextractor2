# # -*- coding: utf-8 -*-
# """
# Actions
# ========
# This module contains important high level processing routines
# author: Damian Wilary
# email: dmw51@cam.ac.uk
# """

import numpy as np

from configs import config
from reactiondataextractor.utils import isolate_patches, HoughLinesP


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
    if len(lengths) > 100:  # Handle cases where text dominates
        percentile = 85
    else:
        percentile = 50
    single_bond = np.percentile(lengths, percentile)
    config.ExtractorConfig.SOLID_ARROW_MIN_LENGTH = int(single_bond // 2)
    config.ExtractorConfig.SOLID_ARROW_THRESHOLD = int(single_bond // 2)
