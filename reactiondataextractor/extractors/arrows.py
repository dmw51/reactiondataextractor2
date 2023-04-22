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
from reactiondataextractor.models.base import BaseExtractor
from reactiondataextractor.models.exceptions import NoArrowsFoundException
from reactiondataextractor.models.segments import FigureRoleEnum, Panel, Figure
from reactiondataextractor.models.geometry import Point
from reactiondataextractor.models.reaction import SolidArrow, CurlyArrow, EquilibriumArrow, ResonanceArrow
from reactiondataextractor.processors import Isolator
# from utils.utils import skeletonize, is_a_single_line, HoughLinesP

log = logging.getLogger('arrows')
from torchvision.models import resnet18
from torch.nn import Sequential, Linear, Sigmoid

class ArrowExtractor(BaseExtractor):
    """Main arrow extractor class combining the different elements of the arrow extraction process"""

    def __init__(self, fig):
        super().__init__(fig)

        # self.line_arrow_extractor = LineArrowCandidateExtractor(fig)
        # self.curly_arrow_extractor = CurlyArrowCandidateExtractor(fig)

        # self.arrow_detector = load_model(ExtractorConfig.ARROW_DETECTOR_PATH)
        # self.arrow_classifier = load_model(ExtractorConfig.ARROW_CLASSIFIER_PATH)

        # self.arrow_detector = load(ExtractorConfig.ARROW_DETECTOR_PATH, map_location=device('cpu'))
        self.arrow_detector = resnet18()
        self.arrow_detector.fc = Sequential(*[Linear(in_features=512, out_features=1), Sigmoid()])
        self.arrow_detector.load_state_dict(load(ExtractorConfig.ARROW_DETECTOR_PATH, map_location=device('cpu')))
        self.arrow_detector.eval()
        self.arrow_classifier = load(ExtractorConfig.ARROW_CLASSIFIER_PATH, map_location=device('cpu'))
        self.arrow_classifier.eval()
        self.arrow_detector.to(ExtractorConfig.DEVICE)
        self.arrow_classifier.to(ExtractorConfig.DEVICE)

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
        # self.curly_arrow_extractor._fig = val
        # self.line_arrow_extractor._fig = val

    @property
    def extracted(self):
        return self.arrows

    def extract(self):
        """Extracts all types of arrows, then combines them, filters duplicates and finally reclassifies them"""
        # line_arrow_proposals = self.line_arrow_extractor.extract()
        # curly_arrow_proposals = self.curly_arrow_extractor.extract()
        # arrows = self.remove_duplicates(line_arrow_proposals + curly_arrow_proposals)

        # arrows = self.filter_false_positives(arrows)
        arrows = self.filter_false_positives(self.fig.connected_components)
        
        # out = demo.plot_selected_arrow_candidates(out, arrows)
        # demo.savefig('extracted_arrows.png')
        # plt.show()
        # plot_arrow_candidates(self.fig, arrows)
        solid_arrows, eq_arrows, res_arrows,  curly_arrows = self.reclassify(arrows)
        # curly_arrows = self.validate_curly_arrows(curly_arrows)
        self._solid_arrows = solid_arrows
        self._eq_arrows = eq_arrows
        self._res_arrows = res_arrows
        self._curly_arrows = curly_arrows
        self.arrows = solid_arrows + curly_arrows + res_arrows + eq_arrows
        if not self.arrows:
            raise NoArrowsFoundException
        return solid_arrows, eq_arrows, res_arrows, curly_arrows

    def plot_extracted(self, ax):
        label_text = {SolidArrow: 'Solid arrow',
                      EquilibriumArrow: 'Equilibrium arrow',
                      ResonanceArrow: 'Resonance arrow',
                      CurlyArrow: 'Curly arrow'}

        for arrow in self.extracted:
            panel = arrow.panel
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor='y', edgecolor=None, alpha=0.7)
            ax.add_patch(rect_bbox)
            # ax.text(panel.left, panel.top, label_text[arrow.__class__], c='b')

    # def validate_curly_arrows(self, curly_arrows):
    #     validated = []
    #     for arrow in curly_arrows:
    #         crop = arrow.panel.create_crop(self.fig)
    #         min_line_length = int(np.mean(crop.img.shape) / 2)
    #         straight_lines = HoughLinesP(crop.img, rho=1, theta=np.pi/180, minLineLength=min_line_length, threshold=min_line_length-10)
    #         if straight_lines == []:
    #             validated.append(arrow)
    #     return validated
                
    def remove_duplicates(self, arrow_candidates):
        """Removes duplicate arrow candidates based on their panel intersection over union (IoU) metric"""


        filtered = []
        for cand in arrow_candidates:
            if all(cand.panel.compute_iou(other_cand.panel) < 0.9 for other_cand in filtered):
                filtered.append(cand)

        return filtered

    def filter_false_positives(self, arrows):
        arrow_crops = [self.preprocess_model_input(arrow) for arrow in arrows]
        arrow_crops = [np.concatenate(3*[x], axis=0) for x in arrow_crops]
        arrow_crops = np.stack(arrow_crops, axis=0)
        print(f'number of potential arrow crops: {len(arrow_crops)}')
        # print(f'number of arrow crops: {len(arrow_crops)}')
        # arrows_pred = self.arrow_detector.predict(x=arrow_crops).squeeze()
        #torch syntax below
        BATCH_SIZE = 32
        batches = np.arange(BATCH_SIZE, arrow_crops.shape[0], BATCH_SIZE)
        arrow_crops = np.split(arrow_crops, batches)
        arrows_pred = []
        with torch.no_grad():
            for batch in arrow_crops:

                arrows_pred.append(self.arrow_detector(torch.tensor(batch)).numpy())
        arrows_pred = np.concatenate(arrows_pred, axis=0)
        arrows_pred = arrows_pred > ExtractorConfig.ARROW_DETECTOR_THRESH
        inliers = np.argwhere(arrows_pred == True)[:, 0]
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
        if not arrows:
            return [], [], [], []
        arrow_crops = [self.preprocess_model_input(arrow) for arrow in arrows]
        arrow_crops = np.stack(arrow_crops, axis=0)
        with torch.no_grad():
            arrow_classes = self.arrow_classifier(torch.tensor(arrow_crops)).numpy()
        arrow_classes = np.argmax(arrow_classes, axis=1)
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

        ##TODO: Retrain the model with a more consistent preprocessing
        arrow_crop = self.crop_from_raw_img(arrow)
        arrow_crop = arrow_crop.resize((self.fig.raw_img.shape[1], self.fig.raw_img.shape[0]), eager_cc_init=False)
        arrow_crop = arrow_crop.resize(ExtractorConfig.ARROW_IMG_SHAPE, eager_cc_init=False)
        arrow_crop = np.pad(arrow_crop.img, ((2, 2), (2, 2)))
        arrow_crop = cv2.resize(arrow_crop, ExtractorConfig.ARROW_IMG_SHAPE)
        arrow_crop = min_max_rescale(arrow_crop)
        #torch preprocess below
        arrow_crop = arrow_crop[np.newaxis, ...].astype(np.float32)


        return arrow_crop

    def crop_from_raw_img(self, panel):
        # isolated_arrow_fig = Isolator(self.fig, arrow, isolate_mask=True).process()
        # arrow_crop = arrow.panel.create_crop(isolated_arrow_fig)
        if self.fig.scaling_factor:
            raw_img = cv2.resize(self.fig.raw_img, (self.fig.img.shape[1], self.fig.img.shape[0]))
            # arrow_panel = list(map(lambda coord: coord * self.fig.scaling_factor))
        else:
            raw_img = self.fig.raw_img
        dummy_fig = Figure(img=raw_img, raw_img=raw_img)
        isolated_raw_arrow = Isolator(dummy_fig, panel, isolate_mask=True).process()
        raw_arrow_crop = panel.create_crop(isolated_raw_arrow)
        return raw_arrow_crop


    def instantiate_arrow(self, panel, cls_idx):
        return self._class_dict[cls_idx](panel)

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

    # def filter_inside_diagrams(self, diags):
    #     """Filters poor detections which are fully contained within detected diagrams. Skips horizontal and vertical
    #     solid arrows"""
    #     filtered_arrows = []
    #
    #     def is_at_boundary(arrow, diag):
    #         """Returns whether an arrow lies at the boundary of a detected diagram - this can happen when dilation
    #         is too large and an arrow becomes part of a diagram"""
    #         diffs = np.abs(np.asarray(arrow.panel.coords) - np.asarray(diag.panel.coords))
    #         return np.any(diffs < 10)
    #
    #     for a in self.arrows:
    #             # if not (d.contains(a) and not is_at_boundary(a, d)):
    #         if not any(d.contains(a) and not is_at_boundary(a, d) for d in diags):
    #             filtered_arrows.append(a)
    #     self.arrows = filtered_arrows



class ArrowClassifier(nn.Module):
    """Simple classifier used to group alle extracted arrows into the correct classes"""
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
