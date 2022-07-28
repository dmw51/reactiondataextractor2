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
import torch
from matplotlib.patches import Rectangle
from scipy.ndimage import label
# from tensorflow.keras.models import load_model
from torch import load, device, nn
from configs import ExtractorConfig, Config
from reactiondataextractor.models.base import BaseExtractor, Candidate
from reactiondataextractor.models.exceptions import NotAnArrowException
from reactiondataextractor.models.segments import FigureRoleEnum, PanelMethodsMixin, Panel, Figure
from reactiondataextractor.utils import skeletonize, is_a_single_line
from reactiondataextractor.models.geometry import Point, Line
from reactiondataextractor.processors import Isolator, Binariser

log = logging.getLogger('arrows')



class ArrowExtractor(BaseExtractor):

    def __init__(self, fig):
        super().__init__(fig)
        self.line_arrow_extractor = LineArrowCandidateExtractor(fig)
        self.curly_arrow_extractor = CurlyArrowCandidateExtractor(fig)

        # self.arrow_detector = load_model(ExtractorConfig.ARROW_DETECTOR_PATH)
        # self.arrow_classifier = load_model(ExtractorConfig.ARROW_CLASSIFIER_PATH)

        self.arrow_detector = load(ExtractorConfig.ARROW_DETECTOR_PATH, map_location=device('cpu'))
        self.arrow_detector.eval()
        self.arrow_classifier = load(ExtractorConfig.ARROW_CLASSIFIER_PATH, map_location=device('cpu'))
        self.arrow_classifier.eval()
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
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, val):
        self._fig = val
        self.curly_arrow_extractor._fig = val
        self.line_arrow_extractor._fig = val

    @property
    def extracted(self):
        return self.arrows

    def extract(self):
        """Extracts all types of arrows, then combines them, filters duplicates and finally reclassifies them"""
        line_arrow_proposals = self.line_arrow_extractor.extract()
        curly_arrow_proposals = self.curly_arrow_extractor.extract()
        arrows = self.remove_duplicates(line_arrow_proposals + curly_arrow_proposals)
        # import demo
        # out = demo.draw_fig(self.fig)
        # out = demo.plot_arrow_candidates(out, arrows)
        arrows = self.filter_false_positives(arrows)
        # out = demo.plot_selected_arrow_candidates(out, arrows)
        # demo.savefig('extracted_arrows.png')
        # plt.show()
        # plot_arrow_candidates(self.fig, arrows)
        solid_arrows, eq_arrows, res_arrows,  curly_arrows = self.reclassify(arrows)
        self._solid_arrows = solid_arrows
        self._eq_arrows = eq_arrows
        self._res_arrows = res_arrows
        self._curly_arrows = curly_arrows
        self.arrows = solid_arrows + curly_arrows + res_arrows + eq_arrows

        return solid_arrows, eq_arrows, res_arrows, curly_arrows

    def plot_extracted(self, ax):
        label_text = {SolidArrow: 'Solid arrow',
                      EquilibriumArrow: 'Equilibrium arrow',
                      ResonanceArrow: 'Resonance arrow',
                      CurlyArrow: 'Curly arrow'}

        for arrow in self.extracted:
            panel = arrow.panel
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor='y', edgecolor=None, alpha=0.4)
            ax.add_patch(rect_bbox)
            ax.text(panel.left, panel.top, label_text[arrow.__class__], c='b')



    def remove_duplicates(self, arrow_candidates):
        """Removes duplicate arrow candidates based on their panel intersection over union (IoU) metric"""


        filtered = []
        for cand in arrow_candidates:
            if all(cand.panel.compute_iou(other_cand.panel) < 0.9 for other_cand in filtered):
                filtered.append(cand)

        return filtered

    def filter_false_positives(self, arrows):
        arrow_crops = [self.preprocess_model_input(arrow) for arrow in arrows]
        arrow_crops = np.stack(arrow_crops, axis=0)
        # print(f'number of arrow crops: {len(arrow_crops)}')
        # arrows_pred = self.arrow_detector.predict(x=arrow_crops).squeeze()
        #torch syntax below
        with torch.no_grad():
            arrows_pred = self.arrow_detector(torch.tensor(arrow_crops)).numpy()
        arrows_pred = arrows_pred > ExtractorConfig.ARROW_DETECTOR_THRESH
        inliers = np.argwhere(arrows_pred == True)[:,0]
        arrows = [arrows[idx] for idx in inliers]
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

        # arrow_classes = self.arrow_classifier.predict(arrow_crops[..., np.newaxis])
        #torch below
        with torch.no_grad():
            arrow_classes = self.arrow_classifier(torch.tensor(arrow_crops)).numpy()
        arrow_classes = np.argmax(arrow_classes, axis=1)
        # filtered_arrows = arrows[inliers]
        completed_arrows = []
        final_classes = []
        for arrow, cls_idx in zip(arrows, arrow_classes):
            try:
                a = self.instantiate_arrow(arrow, cls_idx)
                completed_arrows.append(a)
                final_classes.append(cls_idx)
            except ValueError:
                log.info("Arrow was not instantiated - failed on eroding an arrow into its hook")
        # arrows = [self.instantiate_arrow(arrow, cls_idx) for arrow, cls_idx in zip(arrows, arrow_classes)]
        self.fig.set_roles([a.panel for a in completed_arrows], FigureRoleEnum.ARROW)

        return self.separate_arrows(completed_arrows)

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

        arrow_crop = self.crop_from_raw_img(arrow)

        arrow_crop = arrow_crop.resize((self.fig.raw_img.shape[1], self.fig.raw_img.shape[0]))

        arrow_crop = arrow_crop.resize(ExtractorConfig.ARROW_IMG_SHAPE)
        arrow_crop = np.pad(arrow_crop.img, ((2, 2), (2, 2)))
        arrow_crop = cv2.resize(arrow_crop, ExtractorConfig.ARROW_IMG_SHAPE)
        arrow_crop = min_max_rescale(arrow_crop)
        #torch preprocess below
        arrow_crop = arrow_crop[np.newaxis, ...].astype(np.float32)


        return arrow_crop

    def crop_from_raw_img(self, arrow):
        # isolated_arrow_fig = Isolator(self.fig, arrow, isolate_mask=True).process()
        # arrow_crop = arrow.panel.create_crop(isolated_arrow_fig)
        if self.fig.scaling_factor:
            raw_img = cv2.resize(self.fig.raw_img, (self.fig.img.shape[1], self.fig.img.shape[0]))
            # arrow_panel = list(map(lambda coord: coord * self.fig.scaling_factor))
        else:
            raw_img = self.fig.raw_img
        dummy_fig = Figure(img=raw_img, raw_img=raw_img)
        isolated_raw_arrow = Isolator(dummy_fig, arrow, isolate_mask=True).process()
        raw_arrow_crop = arrow.panel.create_crop(isolated_raw_arrow)
        return raw_arrow_crop


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


class LineArrowCandidateExtractor(BaseExtractor):
    """This line-based arrow candidate extractor is less restrictive in filtering out candidates for straight
    (solid, resonance, equilibrium) arrows. Its main goal is to provide suitable candidates which are to be
    filtered out by another model."""

    def __init__(self, fig):
        self.candidates = None

        super().__init__(fig)

    def extract(self):
        self.candidates = self.find_line_candidates()
        return self.candidates


    @property
    def extracted(self):
        """Returns extracted objects"""
        return self.candidates

    def find_line_candidates(self, ):
        """
        Finds all line arrow candidates in ``fig`` subject to ``threshold`` number of pixels and ``min_arrow_length``
        minimum line length.
        :return: collection of arrow candidates
        :rtype: list
        """
        def inrange(cc, point):
            """Returns True if a ``point`` lies inside ``cc``, else return False."""
            return point.row in range(cc.top, cc.bottom+1) and point.col in range(cc.left, cc.right+1)

        fig = self.fig
        img = copy.deepcopy(fig.img)

        arrow_candidates = []
        skeletonized = skeletonize(fig)
        all_lines = cv2.HoughLinesP(skeletonized.img, rho=1, theta=np.pi/360,
                                    threshold=ExtractorConfig.SOLID_ARROW_THRESHOLD,
                                    minLineLength=ExtractorConfig.SOLID_ARROW_MIN_LENGTH, maxLineGap=2)
    # TODO: Fix this Hough Transform (to find optimal number of candidates). Check exactly which arrows were found (visualize)
        for line in all_lines:
            x1, y1, x2, y2 = line.squeeze()
            # points = [Point(row=y, col=x) for x, y in line]
            # Choose one of points to find the label and pixels in the image
            p1, p2 = Point(row=y1, col=x1), Point(row=y2, col=x2)

            try:
                parent_panel = [cc for cc in fig.connected_components if inrange(cc, p1) and inrange(cc, p2)][0]
            except IndexError:
                continue  # Line crossing multiple connected components found
            # Break the line down and check whether it's a single line
            if not is_a_single_line(skeletonized, parent_panel, int(ExtractorConfig.SOLID_ARROW_MIN_LENGTH)):
                continue

            labelled_img, _ = label(img)

            parent_label = labelled_img[p1.row, p1.col]
            arrow_pixels = np.nonzero(labelled_img == parent_label)
            arrow_pixels = np.array(list(zip(*arrow_pixels)))
            panel_top, panel_bottom = np.min(arrow_pixels[:, 0]), np.max(arrow_pixels[:, 0])+1
            panel_left, panel_right = np.min(arrow_pixels[:, 1]), np.max(arrow_pixels[:, 1])+1

            arrow_candidates.append(ArrowCandidate(pixels=arrow_pixels,
                                               panel=parent_panel))
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


class CurlyArrowCandidateExtractor(BaseExtractor):

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

class BaseArrow(PanelMethodsMixin):
    """Base arrow class common to all arrows
    :param pixels: pixels forming the arrows
    :type pixels: list[Point] or list[(int, int)]
    :param panel: bounding box of an arrow
    :type panel: Panel"""

    def __init__(self, pixels, panel, line=None, contour=None):
        if not all(isinstance(pixel, Point) for pixel in pixels):
            self.pixels = [Point(row=coords[0], col=coords[1]) for coords in pixels]
        else:
            self.pixels = pixels

        self.panel = panel
        self.line = line
        self.contour = contour
        # slope = self.line.slope
        self._center_px = None
        self.reference_pt = self.compute_reaction_reference_pt()
        self.initialize()
        self.children = []  # set dynamically

    @property
    def conditions(self):
        return self.children

    def merge_children(self):
        """Merges child regions if they're close together"""
        # TODO: This is not the exact solution, consider when i-th region gets merged with n > 1 regions
        new_children = []
        unmerged_idx = list(range(len(self.children)))
        for i in range(len(self.children)):
            for j in range(i+1, len(self.children)):
                if self.children[i].panel.edge_separation(self.children[j].panel) < 150 and \
                        self.children[i].arrow == self.children[j].arrow:
                    unmerged_idx.remove(i)
                    unmerged_idx.remove(j)
                    new_children.append(self.children[i].merge_conditions_regions(self.children[j]))
        for i in unmerged_idx:
            new_children.append(self.children[i])

    def initialize(self):
        """Given `pixels` and `panel` attributes, this method checks if other (relevant) initialization attributes have been
        precomputed. If not, these should be computed and set accordingly."""
        if self.line is None:
            self.line = Line.approximate_line(self.pixels[0], self.pixels[-1])

        if self.contour is None:
            isolated_arrow_fig = Isolator(None, self, isolate_mask=True).process()
            cnt, _ = cv2.findContours(isolated_arrow_fig.img,
                                      ExtractorConfig.CURLY_ARROW_CNT_MODE, ExtractorConfig.CURLY_ARROW_CNT_METHOD)
            assert len(cnt) == 1
            self.contour = cnt[0]

    def compute_reaction_reference_pt(self):
        """Computes a reference point for a reaction step. This point alongside arrow's centerpoint to decide whether
        a diagram belongs to reactants or products of a step (by comparing pairwise distances). This reference point
        is a centre of mass in an eroded arrow crop (erosion further moves the original centre of mass away from the
        center point"""
        scaling_factor = 2
        pad_width = 10
        isolated_arrow = Isolator(None, self, isolate_mask=True).process()
        arrow_crop = self.panel.create_padded_crop(isolated_arrow, pad_width=pad_width)
        arrow_crop.img = cv2.resize(arrow_crop.img, (0, 0), fx=scaling_factor, fy=scaling_factor)
        binarised = Binariser(arrow_crop).process()
        eroded = cv2.erode(binarised.img, np.ones((6, 6)), iterations=2)

        #Compute COM in the crop, then transform back to main figure coordinates
        rows, cols = np.where(eroded == 255)
        rows, cols = rows/scaling_factor - pad_width, cols/scaling_factor - pad_width
        row, col = int(np.mean(rows)), int(np.mean(cols))
        row, col = arrow_crop.in_main_fig((row, col))

        return col, row  # x, y


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

    # def initialize(self):
    #     if self.line is None:
    #         self.line = Line.approximate_line(self.pixels[0], self.pixels[-1])
    #
    #     if self.contour is None:
    #         isolated_arrow_fig = Isolator(None, self, isolate_mask=True).process()
    #         cnt, _ = cv2.findContours(isolated_arrow_fig.img,
    #                                   ExtractorConfig.CURLY_ARROW_CNT_MODE, ExtractorConfig.CURLY_ARROW_CNT_METHOD)
    #         assert len(cnt) == 1
    #         self.contour = cnt[0]

    @property
    def is_vertical(self):
        return self.line.is_vertical

    @property
    def slope(self):
        return self.line.slope

    def __repr__(self):
        return f'SolidArrow(pixels={self.pixels[:5]},..., line={self.line}, panel={self.panel})'

    def __str__(self):
        top, left, bottom, right = self.panel
        return f'SolidArrow({top, left, bottom, right})'

    def __eq__(self, other):
        if not isinstance(other, BaseArrow):
            return False
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
        self.line = None

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

    # def initialize(self):
    #     if self.line is None:
    #         pass
    #     if self.contour is None:
    #         pass


class EquilibriumArrow(BaseArrow):
    def __init__(self, pixels, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        super().__init__(pixels, panel)
        self.sort_pixels()

    # def initialize(self):
    #     if self.line is None:
    #         pass
    #     if self.contour is None:
    #         pass


class ArrowCandidate(Candidate):
    """A class to store any attributes that have been computed in the arrow proposal stage. Acts as a cache of values
    which can be reused when an arrow candidate is accepted. All instances are required to have a `pixels` attribute,
    which is used to isolate the relevant connected component prior to arrow detection stage"""

    def __init__(self, pixels, panel=None, *, line=None, contour=None):
        self.pixels = np.array(pixels)
        self.panel = panel
        self.line = line
        self.contour = contour


class ArrowClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 128, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(4 * 4 * 16 * 32, 4)

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = nn.MaxPool2d(2)(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = nn.MaxPool2d(2)(x)
        # x = self.c3(x)
        # x = self.bn3(x)
        # x = nn.MaxPool2d(2)(x)
        # x = self.c4(x)
        # x = self.bn4(x)
        # x = nn.MaxPool2d(2)(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        # print(x.shape)
        x = nn.Softmax(dim=-1)(x)

        return x


class ArrowDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding='same', )
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 128, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.c4 = nn.Conv2d(128, 256, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(256)
        self.c5 = nn.Conv2d(256, 512, 3, padding='same')
        self.bn5 = nn.BatchNorm2d(512)
        self.d1 = nn.Linear(2048, 64)
        self.relu = nn.ReLU()
        # ), activation='relu', kernel_regularizer='l2')
        self.d2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = nn.MaxPool2d(2)(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = nn.MaxPool2d(2)(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = nn.MaxPool2d(2)(x)
        x = self.c4(x)
        x = self.bn4(x)
        x = nn.MaxPool2d(2)(x)
        x = self.c5(x)
        x = self.bn5(x)
        x = nn.MaxPool2d(2)(x)
        x = nn.Flatten()(x)
        x = self.d1(x)
        x = self.relu(x)
        x = self.d2(x)
        x = self.sigmoid(x)
        return x
