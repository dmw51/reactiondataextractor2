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
from typing import List, Tuple, Union

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
from reactiondataextractor.models.segments import FigureRoleEnum, Panel, Figure, Crop
from reactiondataextractor.models.reaction import SolidArrow, CurlyArrow, EquilibriumArrow, ResonanceArrow, BaseArrow
from reactiondataextractor.processors import Isolator

log = logging.getLogger('arrows')
from torchvision.models import resnet18
from torch.nn import Sequential, Linear, Sigmoid, Softmax, Module

class StepwiseClassifier(Module):
    """Heads of the arrow detector + classifier model. The model itself is a resnet18 model
       with the FC layer substituted by an instance of this StepwiseClassifier class"""
    def __init__(self, 
                 in_features:int):
        super().__init__()
        self.linear_1 = Linear(in_features=in_features, out_features=128)
        self.linear_out_binary = Linear(in_features=128, out_features=1)
        self.linear_2 = Linear(in_features=128, out_features=128)
        self.classifier_1 = Sigmoid()

        self.linear_3 = Linear(in_features=128, out_features=128)
        self.linear_4 = Linear(in_features=128, out_features=5)
        self.softmax = Softmax(dim=-1)
        
    def forward(self, x):
        x = self.linear_1(x)
        x_out_step = self.linear_out_binary(x)
        x_out_step = self.classifier_1(x_out_step)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = self.linear_4(x)
        x = self.softmax(x)
        return x_out_step, x


class ArrowExtractor(BaseExtractor):
    """Main arrow extractor class combining the different elements of the arrow extraction process"""

    def __init__(self, 
                 fig: Figure):
        super().__init__(fig)
        self.arrow_detector = resnet18()
        self.arrow_detector.fc = StepwiseClassifier(512)
        self.arrow_detector.load_state_dict(load(ExtractorConfig.ARROW_DETECTOR_PATH, map_location=device('cpu')))
        self.arrow_detector.eval()
        self.arrow_detector.to(ExtractorConfig.DEVICE)

        self.arrows = None
        self._class_dict = {1: SolidArrow,
                            2: EquilibriumArrow,
                            3: ResonanceArrow,
                            4: CurlyArrow
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

    @property
    def extracted(self):
        return self.arrows

    def extract(self):
        """Extracts all types of arrows.
           All individual connected components are selected and their patches fed to the classifier.
           Additionally, to handle equilibrium arrows, pairs of nearby connected components fulfilling certain criteria are
           also selected and checked by the classifier.
           """
        len_ccs = len(self.fig.connected_components)
        equilibrium_arrow_cands = []
        for idx1 in range(len_ccs-1):
            cc1 = self.fig.connected_components[idx1]
            closest = min(self.fig.connected_components[idx1+1:], key=lambda cc2: cc1.center_separation(cc2))
            if cc1.edge_separation(closest) < 40:
                candidate = Panel.create_megapanel([cc1,closest], fig=cc1.fig)
                equilibrium_arrow_cands.append(candidate)
        solid_arrows, eq_arrows, res_arrows,  curly_arrows = self.detect_arrows([self.fig.connected_components, equilibrium_arrow_cands])

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
            ax.text(panel.left, panel.top, label_text[arrow.__class__], c='b')
                
    def detect_arrows(self, panels: Tuple[List[Panel]])-> Tuple[List[BaseArrow]]:
        """Detects all arrows in an image.
        This method takes a tuple of two lists. The first list contains all single connected
        components from the image. Additionally, a second list contains pairs of nearby connected
        components to perform detection of equilibrium arrows.
        The detection model is run separately on the two lists. From the detections on first list, all
        positive (non-background) detections are selected, whereas from the second list's detections, 
        only equilibrium arrow detections are selected.
        :param panels: _description_
        :type panels: Tuple[List[Panel]]
        :return: _description_
        :rtype: Tuple[List[BaseArrow]]
        """
        arrow_candidates, eq_arrow_candidates = panels
        arrows_pred = self._detect_arrows(arrow_candidates)
        arrows_eq_pred = self._detect_arrows(eq_arrow_candidates)
        
        inliers = np.argmax(arrows_pred, -1)
        inliers = np.argwhere(inliers != 0).squeeze(1)
        arrows = [arrow_candidates[idx] for idx in inliers]
        inliers_eq = np.argmax(arrows_eq_pred, -1)
        inliers_eq = np.argwhere(inliers_eq == 2).squeeze(1)
        arrows_eq = [eq_arrow_candidates[idx] for idx in inliers_eq]

        arrow_classes = [arrows_pred[idx].argmax() for idx in inliers] + [2 for _ in range(len(inliers_eq))]
        arrows = arrows + arrows_eq

        completed_arrows = []
        final_classes = []
        for arrow, cls_idx in zip(arrows, arrow_classes):
            try:
                a = self.instantiate_arrow(arrow, cls_idx)
                completed_arrows.append(a)
                final_classes.append(cls_idx)
            except ValueError:
                log.info("Arrow was not instantiated - failed on eroding an arrow into its hook")

        filtered_arrows = []
        equilibrium_arrows = [a for a in completed_arrows if isinstance(a, EquilibriumArrow)]
        for arrow in completed_arrows:
            if isinstance(arrow, SolidArrow) and any( a.panel.contains(arrow.panel) for a in equilibrium_arrows):
                continue
            elif isinstance(arrow, CurlyArrow) and arrow.panel.area / self.fig.get_bounding_box().area * 100 < 0.1:
                continue
            else:
                filtered_arrows.append(arrow)
                
        self.fig.set_roles([a.panel for a in filtered_arrows], FigureRoleEnum.ARROW)
        return self.separate_arrows(filtered_arrows)

    def _detect_arrows(self, panels):
        crops = [self.preprocess_model_input(arrow) for arrow in panels]
        crops = [np.concatenate(3*[x], axis=0) for x in crops]
        crops = np.stack(crops, axis=0)
        print(f'number of potential arrow crops: {len(crops)}')
        BATCH_SIZE = 32
        batches = np.arange(BATCH_SIZE, crops.shape[0], BATCH_SIZE)
        crops = np.split(crops, batches)
        arrows_pred = []
        
        with torch.no_grad():
            for batch in crops:
                _, out = self.arrow_detector(torch.tensor(batch))
                arrows_pred.append(out.numpy())
        arrows_pred = np.concatenate(arrows_pred, axis=0)
        return arrows_pred

    def preprocess_model_input(self, panel: Panel) -> np.ndarray:
        """Converts a panel into image array and resizes it to the desired input shape as required by the arrow detection model.
        :param panel: panel to preprocess
        :type panel: Panel"""

        def min_max_rescale(img):
            min_ = np.min(img)
            max_ = np.max(img)
            if min_ == max_:
                img = np.zeros_like(img)
            else:
                img = (img - min_) / (max_ - min_)
            return img

        arrow_crop = self.crop_from_raw_img(panel)
        arrow_crop = arrow_crop.resize((self.fig.raw_img.shape[1], self.fig.raw_img.shape[0]), eager_cc_init=False)
        arrow_crop = arrow_crop.resize(ExtractorConfig.ARROW_IMG_SHAPE, eager_cc_init=False)
        arrow_crop = np.pad(arrow_crop.img, ((2, 2), (2, 2)))
        arrow_crop = cv2.resize(arrow_crop, ExtractorConfig.ARROW_IMG_SHAPE)
        arrow_crop = min_max_rescale(arrow_crop)
        arrow_crop = arrow_crop[np.newaxis, ...].astype(np.float32)

        return arrow_crop

    def crop_from_raw_img(self, panel: Panel) -> Crop:
        """Helper function used to crop an image patch from the initial (raw) img
        :param panel: panel delineating the patch to crop
        :type panel: Panel"""
        if self.fig.scaling_factor:
            raw_img = cv2.resize(self.fig.raw_img, (self.fig.img.shape[1], self.fig.img.shape[0]))
        else:
            raw_img = self.fig.raw_img
        dummy_fig = Figure(img=raw_img, raw_img=raw_img)
        isolated_raw_arrow = Isolator(dummy_fig, panel, isolate_mask=True).process()
        raw_arrow_crop = panel.create_crop(isolated_raw_arrow)
        return raw_arrow_crop

    def instantiate_arrow(self, panel: Panel, cls_idx: int) -> Union[SolidArrow,CurlyArrow,EquilibriumArrow,ResonanceArrow]:
        """Helper function. Instantiates an arrow of a specific class given cls_idx returned by the detection model
        :param panel: Panel delineating the arrow
        :type panel: Panel
        :param cls_idx: integer index that maps into an arrow class
        :type cls: int"""
        return self._class_dict[cls_idx](panel)

    def separate_arrows(self, arrows: List[BaseArrow]) -> Tuple[List[BaseArrow]]:
        """Helper function used to group a single sequence of arrows into multiple sequences 
        according to their types.
        :param arrows: a sequence of arrows
        :type arrows: List"""
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
