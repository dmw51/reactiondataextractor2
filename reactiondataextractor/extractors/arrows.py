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
from operator import itemgetter

import cv2
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import label
from tensorflow.keras.models import load_model

from reactiondataextractor.config import ExtractorConfig, Config
from reactiondataextractor.extractors.base import BaseExtractor, Candidate
from reactiondataextractor.models.exceptions import NotAnArrowException
from reactiondataextractor.models.segments import FigureRoleEnum, PanelMethodsMixin, Panel
from reactiondataextractor.utils import skeletonize, is_a_single_line
from reactiondataextractor.models.geometry import Point, Line
from reactiondataextractor.processors import Isolator, EdgeExtractor

log = logging.getLogger('arrows')




class ArrowExtractor(BaseExtractor):

    def __init__(self, fig):
        super().__init__(fig)
        self.solid_arrow_extractor = SolidArrowExtractor(fig)
        self.curly_arrow_extractor = CurlyArrowExtractorGeneric(fig)

        self.arrow_detector = load_model(ExtractorConfig.ARROW_DETECTOR_PATH)
        self.arrow_classifier = load_model(ExtractorConfig.ARROW_CLASSIFIER_PATH)

        self.arrows = None
        self._class_dict = {0: SolidArrow,
                            1: EquilibriumArrow,
                            2: ResonanceArrow,
                            3: CurlyArrow
                            }
        self._solid_arrows = []
        self._eq_arrows = []
        self._res_arrows = []
        self._curly_arrows = []
    @property
    def extracted(self):
        return self.arrows

    def extract(self):
        """Extracts all types of arrows, then combines them, filters duplicates and finally reclassifies them"""
        solid_arrows = self.solid_arrow_extractor.extract()
        curly_arrows = self.curly_arrow_extractor.extract()
        arrows = self.remove_duplicates(solid_arrows + curly_arrows)
        arrows = self.filter_false_positives(arrows)
        solid_arrows, eq_arrows, res_arrows,  curly_arrows = self.reclassify(arrows)
        self._solid_arrows = solid_arrows
        self._eq_arrows = eq_arrows
        self._res_arrows = res_arrows
        self._curly_arrows = curly_arrows
        self.arrows = solid_arrows + curly_arrows + res_arrows, eq_arrows
        return solid_arrows, eq_arrows, res_arrows, curly_arrows

    def remove_duplicates(self, arrow_candidates):
        def is_different_px(px1, px2):
            return px1[0] != px2[0] and px1[1] != px2[1]

        filtered = []
        for cand in arrow_candidates:
            if all(is_different_px(cand.pixels[0], filtered_arrow.pixels[1]) for filtered_arrow in filtered):
                filtered.append(cand)

        return filtered

    def filter_false_positives(self, arrows):
        arrow_crops = [self.preprocess_model_input(arrow) for arrow in arrows]
        arrow_crops = np.stack(arrow_crops, axis=0)
        arrows_pred = self.arrow_detector.predict(x=arrow_crops).squeeze()
        arrows_pred = arrows_pred > ExtractorConfig.ARROW_DETECTOR_THRESH
        inliers = np.argwhere(arrows_pred == True).squeeze()
        arrows = itemgetter(*inliers)(arrows)
        return arrows
        # solid_arrows = []
        # curly_arrows = []
        # for a in arrows:
        #     if isinstance(a, SolidArrow):
        #         solid_arrows.append(a)
        #     else:
        #         curly_arrows.append(a)

        # return solid_arrows, curly_arrows

    def reclassify(self, arrows):
        arrow_crops = [self.preprocess_model_input(arrow) for arrow in arrows]
        arrow_crops = np.stack(arrow_crops, axis=0)

        arrow_classes = self.arrow_classifier.predict(arrow_crops[..., np.newaxis])
        arrow_classes = np.argmax(arrow_classes, axis=1)
        # filtered_arrows = arrows[inliers]
        arrows = [self.instantiate_arrow(arrow, cls_idx) for arrow, cls_idx in zip(arrows, arrow_classes)]
        self.fig.set_roles([a.panel for a in arrows], FigureRoleEnum.ARROW)

        return self.separate_arrows(arrows)

    def preprocess_model_input(self, arrow):
        """Converts arrow objects into image arrays and resizes them to the desired input shape as required by the
        novelty detector and arrow classifier"""

        def min_max_rescale(img):

            min_ = np.min(img)
            max_ = np.max(img)
            if min_ == max_:
                img = np.zeros_like(img)
            else:
                img = (img - min_) / (max_ - min_)
            return img

        isolated_arrow_fig = Isolator(self.fig, arrow, isolate_mask=True).process()
        arrow_crop = arrow.panel.create_crop(isolated_arrow_fig)
        arrow_crop = arrow_crop.resize( ExtractorConfig.ARROW_IMG_SHAPE)
        arrow_crop = np.pad(arrow_crop.img, ((2, 2), (2, 2)))
        arrow_crop = cv2.resize(arrow_crop, ExtractorConfig.ARROW_IMG_SHAPE)
        arrow_crop = min_max_rescale(arrow_crop)

        # _, arrow_crop = cv2.threshold(arrow_crop, 40, 255, cv2.THRESH_BINARY)
        # binarised = EdgeExtractor(arrow_crop, bin_thresh=[20, 255]).process()
        # arrow_crop = binarised.img

        # arrow_crop = (arrow_crop - arrow_crop.min())/(arrow_crop.max() - arrow_crop.min())
        # arrow_vector = scaler.fit_transform(arrow_vector)
        # arrow_crop = arrow_vector.reshape((arrow_crop.shape))

        # arrow_crop = self.rotate_arrow(arrow_crop.img)

        # arrow_vector = np.reshape(arrow_crop, -1)
        return arrow_crop

    # def postprocess_model_output(self, output):
    #     """Takes np.array `output` output from a model and postprocesses for novelty detection"""
    #     output = np.squeeze(output)
    #     output = output > ExtractorConfig.RECONSTR_BIN_THRESH
    #     return output

    def instantiate_arrow(self, arrow_candidate, cls_idx):
        return self._class_dict[cls_idx](**arrow_candidate.pass_attributes())

    def separate_arrows(self, arrows):
        solid_arrows, eq_arrows, res_arrows, curly_arrows = [], [], [], []
        for a in arrows:
            if isinstance(a, SolidArrow):
                solid_arrows.append(a)
            elif isinstance(a, EquilibriumArrow):
                eq_arrows.append(a)
            elif isinstance(a, ResonanceArrow):
                res_arrows.append(a)
            elif isinstance(a, CurlyArrow):
                curly_arrows.append(a)

        return solid_arrows, eq_arrows, res_arrows, curly_arrows


class SolidArrowExtractor(BaseExtractor):
    """This solid arrow extractor is less restrictive in filtering out candidates for solid arrows. Its main goal
    is to provide suitable candidates which are to be filtered out by another model."""

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
        :return: collection of arrow objects
        :rtype: list
        """
        def inrange(cc, point):
            """Returns True if a ``point`` lies inside ``cc``, else return False."""
            return point.row in range(cc.top, cc.bottom+1) and point.col in range(cc.left, cc.right+1)

        fig = self.fig
        img = copy.deepcopy(fig.img)

        arrow_candidates = []
        skeletonized = skeletonize(fig)
        all_lines = cv2.HoughLinesP(skeletonized.img, rho=1, theta=np.pi/2,
                                    threshold=ExtractorConfig.SOLID_ARROW_THRESHOLD,
                                    minLineLength=ExtractorConfig.SOLID_ARROW_MIN_LENGTH, maxLineGap=2)

        for line in all_lines:
            x1, y1, x2, y2 = line.squeeze()
            # points = [Point(row=y, col=x) for x, y in line]
            # Choose one of points to find the label and pixels in the image
            p1, p2 = Point(row=y1, col=x1), Point(row=y2, col=x2)


            parent_panel = [cc for cc in fig.connected_components if inrange(cc, p1) and inrange(cc, p2)][0]

            # Break the line down and check whether it's a single line
            if not is_a_single_line(skeletonized, parent_panel, int(ExtractorConfig.SOLID_ARROW_MIN_LENGTH*0.5)):
                continue

            labelled_img, _ = label(img)

            parent_label = labelled_img[p1.row, p1.col]
            arrow_pixels = np.nonzero(labelled_img == parent_label)
            arrow_pixels = np.array(list(zip(*arrow_pixels)))
            panel_top, panel_bottom = np.min(arrow_pixels[:, 0]), np.max(arrow_pixels[:, 0])+1
            panel_left, panel_right = np.min(arrow_pixels[:, 1]), np.max(arrow_pixels[:, 1])+1

            arrow_candidates.append(ArrowCandidate(arrow_pixels,
                                               panel=Panel((panel_top, panel_left, panel_bottom, panel_right))))
        return arrow_candidates

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


class CurlyArrowExtractorGeneric(BaseExtractor):

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
            # if 6 > sides > 2 and no_parent_cond and min_area_cond and cnt_area_to_bbox_area_cond:
            if 6 > sides > 2  and min_area_cond and cnt_area_to_bbox_area_cond:
                mask = np.zeros(self.img.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                pixels = np.transpose(np.nonzero(mask))
                top, left, bottom, right = y, x, y + h, x + w
                arrow_cand = ArrowCandidate(pixels, contour=cnt, panel=Panel((top, left, bottom, right)))
                # arrow = CurlyArrow(pixels, Panel((top, left, bottom, right)), cnt)


                found_arrows.append(arrow_cand)
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
        self.panel = panel
        # slope = self.line.slope
        self._center_px = None
        self.reference_pt = self.compute_reaction_reference_pt()
        self.initialize()

    @abc.abstractmethod
    def initialize(self):
        """Given `pixels` and `panel` attributes, this method checks if other (relevant) initialization attributes have been
        precomputed. If not, these should be computed and set accordingly."""
        pass

    def compute_reaction_reference_pt(self):
        """Computes a reference point for a reaction step. This point alongside arrow's centerpoint to decide whether
        a diagram belongs to reactants or products of a step (by comparing pairwise distances). This reference point
        is a centre of mass in an eroded arrow crop (erosion further moves the original centre of mass away from the
        center point"""
        scaling_factor = 2
        pad_width = 10

        isolated_arrow = Isolator(Config.FIGURE, self, isolate_mask=True).process()
        arrow_crop = self.panel.create_padded_crop(isolated_arrow, pad_width=(pad_width, pad_width))
        arrow_crop.img = cv2.resize(arrow_crop.img, (0, 0), fx=scaling_factor, fy=scaling_factor)
        binarised = EdgeExtractor(arrow_crop).process()
        eroded = cv2.erode(binarised.img, np.ones((6, 6)), iterations=2)

        #Compute COM in the crop, then transform back to main figure coordinates
        rows, cols = np.where(eroded == 255)
        rows, cols = rows/scaling_factor - pad_width, cols/scaling_factor - pad_width
        row, col = int(np.mean(rows)), int(np.mean(cols))
        row, col = arrow_crop.in_main_fig((row, col))

        return row, col


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

    def __init__(self, pixels, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        # self.react_side = None
        # self.prod_side = None
        super(SolidArrow, self).__init__(pixels, panel)
        self.sort_pixels()

    def initialize(self):
        if self.line is None:
            self.line = Line.approximate_line(self.pixels[0], self.pixels[-1])

    @property
    def is_vertical(self):
        return self.line.is_vertical

    @property
    def slope(self):
        return self.line.slope

    def __repr__(self):
        return f'SolidArrow(pixels={self.pixels[:5]},..., line={self.line}, panel={self.panel})'

    def __str__(self):
        left, right, top, bottom = self.panel
        return f'SolidArrow({top, left, bottom, right})'

    def __eq__(self, other):
        return self.panel == other.panel

    def __hash__(self):
        return hash(pixel for pixel in self.pixels)

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



class CurlyArrow(BaseArrow):

    def __init__(self, pixels, panel, contour=None, line=None):
        self.contour = contour
        super().__init__(pixels, panel)
        self._line = None

    def initialize(self):
        if self.contour is None:
            isolated_arrow_fig = Isolator(None, self, isolate_mask=True).process()
            cnt, _ = cv2.findContours(isolated_arrow_fig.img,
                                      ExtractorConfig.CURLY_ARROW_CNT_MODE, ExtractorConfig.CURLY_ARROW_CNT_METHOD)
            assert len(cnt) == 1
            self.contour = cnt[0]


class ResonanceArrow(BaseArrow):
    def __init__(self, pixels, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        super().__init__(pixels, panel)
        self.sort_pixels()

    def initialize(self):
        if self.line is None:
            pass
        if self.contour is None:
            pass


class EquilibriumArrow(BaseArrow):
    def __init__(self, pixels, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        super().__init__(pixels, panel)
        self.sort_pixels()

    def initialize(self):
        if self.line is None:
            pass
        if self.contour is None:
            pass


class ArrowCandidate(Candidate):
    """A class to store any attributes that have been computed in the arrow proposal stage. Acts as a cache of values
    which can be reused when an arrow candidate is accepted. All instances are required to have a `pixels` attribute,
    which is used to isolate the relevant connected component prior to arrow detection stage"""

    def __init__(self, pixels, panel=None, *, line=None, contour=None):
        self.pixels = np.array(pixels)
        self.panel = panel
        self.line = line
        self.contour = contour
