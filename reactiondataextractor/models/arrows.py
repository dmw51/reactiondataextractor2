# -*- coding: utf-8 -*-
"""
Base
=====
This module contains arrow extraction classes
author: Damian Wilary
email: dmw51@cam.ac.uk
"""
import abc
import copy
import logging
from itertools import product

import cv2
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import label

from config import ExtractorConfig
from models.base import BaseExtractor
from models.exceptions import NotAnArrowException
from models.segments import FigureRoleEnum, PanelMethodsMixin, Panel
from models.utils import skeletonize, is_a_single_line
from models.geometry import Point, Line

log = logging.getLogger('arrows')

class ArrowExtractor(BaseExtractor):

    def __init__(self, fig):
        self.fig = fig
        self.solid_arrow_extr = SolidArrowExtractor(fig)
        self.curly_arrow_extr = CurlyArrowExtractor(fig)
        self.arrows = None

    @property
    def extracted(self):
        return self.arrows

    def extract(self):
        """Extracts all types of arrows, then combines them, filters duplicates and finally reclassifies them"""
        solid_arrows = self.solid_arrow_extr.extract()
        curly_arrows = self.curly_arrow_extr.extract()
        arrows = self.remove_duplicates(solid_arrows+curly_arrows)
        solid_arrows, curly_arrows = self.reclassify(arrows)
        self._solid_arrows = solid_arrows
        self._curly_arrows = curly_arrows
        self.arrows = solid_arrows + curly_arrows
        return solid_arrows, curly_arrows

    def remove_duplicates(self, arrows):
        filtered = []
        for arrow in arrows:
            if all(arrow.panel != filtered_arrow.panel for filtered_arrow in filtered):
                filtered.append(arrow)

        return filtered

    def reclassify(self, arrows):
        #TODO
        pass




class SolidArrowExtractor(BaseExtractor):

    def __init__(self, fig):
        self.arrows = None

        super().__init__(fig)

    def extract(self):
        return self.find_solid_arrows()


    @property
    def extracted(self):
        """Returns extracted objects"""
        return self.arrows

    def find_solid_arrows(self, ):
        """
        Finds all solid (straight) arrows in ``fig`` subject to ``threshold`` number of pixels and ``min_arrow_length``
        minimum line length.
        :param int threshold: threshold number of pixels needed to define a line (Hough transform param).
        :return: collection of arrow objects
        :rtype: list
        """
        def inrange(cc, point):
            """Returns True if a ``point`` lies inside ``cc``, else return False."""
            return point.row in range(cc.top, cc.bottom+1) and point.col in range(cc.left, cc.right+1)

        fig = self.fig
        img = copy.deepcopy(fig.img)

        arrows = []
        skeletonized = skeletonize(fig)
        all_lines = cv2.HoughLinesP(skeletonized.img, rho=1, theta=np.pi/2,
                                    threshold=ExtractorConfig.SOLID_ARROW_THRESHOLD, minLineLength=ExtractorConfig.SOLID_ARROW_MIN_LENGTH, maxLineGap=3)

        for line in all_lines:
            x1, y1, x2, y2 = line.squeeze()
            # points = [Point(row=y, col=x) for x, y in line]
            # Choose one of points to find the label and pixels in the image
            p1, p2 = Point(row=y1, col=x1), Point(row=y2, col=x2)
            labelled_img, _ = label(img)
            p1_label = labelled_img[p1.row, p1.col]
            p2_label = labelled_img[p2.row, p2.col]
            if p1_label != p2_label:  # Hough transform can find lines spanning several close ccs; these are discarded
                log.debug('A false positive was found when detecting a line. Discarding...')
                continue
            else:
                parent_label = labelled_img[p1.row, p1.col]

                parent_panel = [cc for cc in fig.connected_components if inrange(cc, p1) and inrange(cc, p2)][0]

            # Break the line down and check whether it's a single line
            if not is_a_single_line(skeletonized, parent_panel, int(ExtractorConfig.SOLID_ARROW_MIN_LENGTH*0.5)):
                continue

            arrow_pixels = np.nonzero(labelled_img == parent_label)
            arrow_pixels = list(zip(*arrow_pixels))
            try:
                new_arrow = SolidArrow(arrow_pixels, line=Line.approximate_line(p1, p2), panel=parent_panel)
            except NotAnArrowException as e:
                log.info('An arrow candidate was discarded - ' + str(e))
            else:
                arrows.append(new_arrow)
                parent_cc = [cc for cc in fig.connected_components if cc == new_arrow.panel][0]
                parent_cc.role = FigureRoleEnum.ARROW

        return list(set(arrows))

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        if not self.extracted:
            pass
        else:
            for arrow in self.extracted:
                panel = arrow.panel
                rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                      facecolor='y', edgecolor=None, alpha=0.4)
                ax.add_patch(rect_bbox)


class CurlyArrowExtractor(BaseExtractor):

    def __init__(self, fig):
        self.arrows = None
        super().__init__(fig)

    @property
    def extracted(self):
        return self.arrows

    def extract(self):
        img_area = self.img.shape[0] * self.img.shape[1]
        # img = cv2.dilate(self.img, np.ones((2,2)), iterations=2)
        contours, hierarchy = cv2.findContours(self.img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        found_arrows = []
        for cnt, hier in zip(contours, hierarchy.squeeze()):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            hull = cv2.convexHull(approx, returnPoints=False)
            sides = len(hull)
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = w * h
            cnt_area = cv2.contourArea(cnt)
            area_ratio = cnt_area / bbox_area

            min_area_cond = w * h > ExtractorConfig.CURLY_ARROW_MIN_AREA_FRACTION * img_area
            cnt_area_to_bbox_area_cond = area_ratio < ExtractorConfig.CURLY_ARROW_CNT_AREA_TO_BBOX_AREA_RATIO
            no_parent_cond = hier[-1] == -1 #and hier[-2] == -1  # Contour has no children
            # if 6 > sides > 2 and no_parent_cond and min_area_cond and cnt_area_to_bbox_area_cond:
            if 6 > sides > 2 and no_parent_cond and min_area_cond and cnt_area_to_bbox_area_cond:
                mask = np.zeros(self.img.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                pixels = np.transpose(np.nonzero(mask))
                top, left, bottom, right = y, x, y + w, x + w
                arrow = CurlyArrow(pixels, Panel((top, left, bottom, right)))


                found_arrows.append(arrow)
        self.arrows = found_arrows
        return found_arrows



class BaseArrow(PanelMethodsMixin):
    """Base arrow class common to all arrows
    :param pixels: pixels forming the arrows
    :type pixels: list[Point] or list[(int, int)]
    :param panel: bounding box of an arrow
    :type panel: Panel"""

    def __init__(self, pixels, panel):
        if not all(isinstance(pixel, Point) for pixel in pixels):
            self.pixels = [Point(row=coords[0], col=coords[1]) for coords in pixels]
        else:
            self.pixels = pixels

        # self.line = line
        self._panel = panel
        # slope = self.line.slope
        self._center_px = None

    @property
    def panel(self):
        return self._panel



    @property
    def center_px(self):
        """
        Based on a geometric centre of an arrow panel, looks for a pixel nearby that belongs to the arrow.
        :return: coordinates of the pixel that is closest to geometric centre and belongs to the object.
        If multiple pairs found, return the floor average.
        :rtype: Point
        """
        if self._center_px is not None:
            return self._center_px

        log.debug('Finding center of an arrow...')
        x, y = self.panel.geometric_centre

        log.debug('Found an arrow with geometric center at (%s, %s)' % (y, x))

        # Look at pixels neighbouring center to check which actually belong to the arrow
        x_candidates = [x+i for i in range(-3, 4)]
        y_candidates = [y+i for i in range(-3, 4)]
        center_candidates = [candidate for candidate in product(x_candidates, y_candidates) if
                             Point(row=candidate[1], col=candidate[0]) in self.pixels]

        log.debug('Possible center pixels: %s', center_candidates)
        if center_candidates:
            self._center_px = np.mean(center_candidates, axis=0, dtype=int)
            self._center_px = Point(row=self._center_px[1], col=self._center_px[0])
        else:
            raise NotAnArrowException('No component pixel lies on the geometric centre')
        log.debug('Center pixel found: %s' % self._center_px)

        return self._center_px

    @abc.abstractmethod
    def sort_pixels(self):
        """This method is used to sort the arrow pixels"""
        pass


class SolidArrow(BaseArrow):
    """
    Class used to represent simple reaction arrows.
    :param pixels: pixels forming the arrows
    :type pixels: list[Point]
    :param line: line found by Hough transform, underlying primitive,
    :type line: Line
    :param panel: bounding box of an arrow
    :type panel: Panel"""

    def __init__(self, pixels, line, panel):
        super(SolidArrow, self).__init__(pixels, panel)
        self.line = line
        self.react_side = None
        self.prod_side = None
        self.sort_pixels()


        a_ratio = self.panel.aspect_ratio
        a_ratio = 1/a_ratio if a_ratio < 1 else a_ratio
        if a_ratio < 3:
            raise NotAnArrowException('aspect ratio is not within the accepted range')

        self.react_side, self.prod_side = self.get_direction()
        pixel_majority = len(self.prod_side) - len(self.react_side)
        num_pixels = len(self.pixels)
        min_pixels = min(int(0.1 * num_pixels), 15)
        if pixel_majority < min_pixels:
            raise NotAnArrowException('insufficient pixel majority')
        elif pixel_majority < 2 * min_pixels:
            log.warning('Difficulty detecting arrow sides - low pixel majority')

        log.debug('Arrow accepted!')


    @property
    def is_vertical(self):
        return self.line.is_vertical

    @property
    def slope(self):
        return self.line.slope

    def __repr__(self):
        return f'SolidArrow(pixels={self.pixels}, line={self.line}, panel={self.panel})'

    def __str__(self):
        left, right, top, bottom = self.panel
        return f'SolidArrow({left, right, top, bottom})'

    def __eq__(self, other):
        return self.panel == other.panel

    def __hash__(self):
        return hash(pixel for pixel in self.pixels)

    @property
    def hook(self):
        """
        Returns the last pixel of an arrow hook.
        :return:
        """
        if self.is_vertical:
            prod_side_lhs = True if self.prod_side[0].row < self.react_side[0].row else False
        else:
            prod_side_lhs = True if self.prod_side[0].col < self.react_side[0].col else False
        return self.prod_side[0] if prod_side_lhs else self.prod_side[-1]

    def sort_pixels(self):
        """
        Simple pixel sort.
        Sorts pixels by row in vertical arrows and by column in all other arrows
        :return:
        """
        if self.is_vertical:
            self.pixels.sort(key=lambda pixel: pixel.row)
        else:
            self.pixels.sort(key=lambda pixel: pixel.col)

    def get_direction(self):
        """Retrieves the direction of an arrow by looking at the number of pixels on each side.
        Splits an arrow in the middle depending on its slope and calculated the number of pixels in each part."""
        center = self.center
        center = Point(center[1], center[0])
        if self.is_vertical:
            part_1 = [pixel for pixel in self.pixels if pixel.row < center.row]
            part_2 = [pixel for pixel in self.pixels if pixel.row > center.row]

        elif self.line.slope == 0:
            part_1 = [pixel for pixel in self.pixels if pixel.col < center.col]
            part_2 = [pixel for pixel in self.pixels if pixel.col > center.col]

        else:
            p_slope = -1/self.line.slope
            p_intercept = center.row - center.col*p_slope
            p_line = lambda point: point.col*p_slope + p_intercept
            part_1 = [pixel for pixel in self.pixels if pixel.row < p_line(pixel)]
            part_2 = [pixel for pixel in self.pixels if pixel.row > p_line(pixel)]

        if len(part_1) > len(part_2):
            react_side = part_2
            prod_side = part_1
        else:
            react_side = part_1
            prod_side = part_2

        log.debug('Established reactant and product sides of an arrow.')
        log.debug('Number of pixel on reactants side: %s ', len(react_side))
        log.debug('product side: %s ', len(prod_side))
        return react_side, prod_side

class CurlyArrow(BaseArrow):

    def __init(self, pixels, panel):

        super().__init__(pixels, panel)


