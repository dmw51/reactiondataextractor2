"""This module contains all parts of the unified extractor for diagrams, labels and reaction conditions"""
import copy
from itertools import product
from math import ceil
import numpy as np
import json
import os
import re
from typing import List, Tuple, Union
import warnings

import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances, Boxes
import torch
from torch import Tensor

from models.exceptions import NoDiagramsFoundException
from reactiondataextractor.models import BaseExtractor, Candidate
from reactiondataextractor.models.reaction import Label, Conditions, Diagram, CurlyArrow
from reactiondataextractor.models.segments import Panel, Rect, FigureRoleEnum, Crop, PanelMethodsMixin, Figure
from reactiondataextractor.extractors import ConditionsExtractor, LabelExtractor
from configs.config import ExtractorConfig
from reactiondataextractor.utils.utils import dilate_fig, erase_elements, find_relative_directional_position, \
    compute_ioa, lies_along_arrow_normal, pixel_ratio

parent_dir = os.path.dirname(os.path.abspath(__file__))
superatom_file = os.path.join(parent_dir, '..', 'dict', 'filter_superatoms.txt')
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'


class UnifiedExtractor(BaseExtractor):
    """The main object detection model. Combines an underlying detectron2 object detection model, as well
    as the individual diagram, labels and conditions extractors"""
    def __init__(self, 
                 fig:'Figure',
                 arrows: List['BaseArrow'],
                 use_tiler: bool=True):
        """
        :param fig: Analysed figure
        :type fig: Figure
        :param arrows: all extracted arrows from the arrow extractor model
        :type arrows: list[BaseArrow]
        :param use_tiler: Whether to perform small object detection on image patches
        :type use_tiler: bool
        """
        super().__init__(fig)
        self.model = Detectron2Adapter(fig, use_tiler)
        self._arrows = arrows
        self.diagram_extractor = DiagramExtractor(self.fig, diag_priors=None, arrows=self.all_arrows)
        self.label_extractor = LabelExtractor(self.fig, priors=None)
        self.conditions_extractor = ConditionsExtractor(self.fig, priors=None)
        self.masked_fig = None
        self._diags_only = False
        self.diagram_extractor._diags_only = False
        self._class_dict = {
            0: Diagram,
            1: Conditions,
            2: Label
        }

    @property
    def extracted(self):
        return self._extracted

    @property
    def all_arrows(self):
        return self._arrows

    @all_arrows.setter
    def all_arrows(self, val):
        self._arrows = val
        self.diagram_extractor._arrows = val

    @property
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, val):
        """Sets the `fig` attribute for this object, as well as all models and constituent extractors for consistency"""
        self._fig = val
        self.model.fig = val
        self.diagram_extractor._fig = val
        self.label_extractor._fig = val
        self.conditions_extractor._fig = val
        
    @property
    def diags_only(self):
        return self._diags_only
    
    @diags_only.setter
    def diags_only(self, value):
        self._diags_only = value
        self.diagram_extractor._diags_only = value

    def extract(self,
                report_raw_results:bool=False):
        """The main extraction method.
        Delegates extraction to the inner model instance.
        Processes outputs from the object detection model by delegating to the inner extractors and methods.
        Postprocesses diagrams using dilation, then uses these along with arrows to postprocess viable labels and conditions
        :param report_raw_results: a helper flag which can be used for evaluation purposes. 
        When set, the raw results from the object detection models are returned additionally to the main return values
        :type report_raw_results: bool
        return: postprocessed diagrams, conditions, and labels
        rtype: tuple[list]"""
        print('Running the main object detection model...')
        boxes, classes = self.model.detect()

        out_diag_boxes = [box for box, class_ in zip(boxes, classes) if self._class_dict[class_] == Diagram]

        diags = self.postprocess_diagrams(out_diag_boxes)
        if not diags:
            raise NoDiagramsFoundException
        
        text_regions = [TextRegionCandidate(box, class_) for box, class_ in zip(boxes, classes)
                        if self._class_dict[class_] in [Label, Conditions]]
        conditions, labels = self.postprocess_text_regions(text_regions)
        if conditions:
            self.set_parents_for_text_regions(conditions, self.all_arrows)
        if labels:
            self.set_parents_for_text_regions(labels, diags)
        self._clean_up_diag_label_matchings(diags)
        
        if not self.diags_only:
            self.add_diags_to_conditions(diags)
            
        self._extracted = diags, conditions, labels
        if report_raw_results:
            return diags, conditions, labels,  (boxes, classes)
        return diags, conditions, labels

    def add_diags_to_conditions(self, diags: List['Diagram']):
        """Adds diagrams to conditions regions where applicable. This is done by assessing the relationship
        between diagrams and arrows
        :param diags: All extracted diagrams
        :type diags: list[Diagram]"""
        for diag in diags:
            for arrow in self.all_arrows:
                if not isinstance(arrow, CurlyArrow) and lies_along_arrow_normal(arrow, diag) and diag.edge_separation(arrow.panel) < ExtractorConfig.ARROW_DIAG_MAX_DISTANCE:
                    conditions = [c for c in arrow.children if isinstance(c, Conditions)]
                    if len(conditions) > 0:
                        conditions[0]._diags.append(diag)
                        conditions[0].panel = Panel.create_megapanel([conditions[0].panel, diag.panel], diag.panel.fig)
                    else:
                        arrow.children.append(Conditions(diag.panel,None,None,None,diags=[diag]))

    def _clean_up_diag_label_matchings(self, diags):
        diags_multiple_labels = [d for d in diags if len(d.children) > 1]
        if not diags_multiple_labels:
            return

        for d in diags_multiple_labels:
            for label in d.children:
                dists = sorted([(d, label.panel.edge_separation(d)) for d in diags], key=lambda x: x[1])[:2]
                if len(dists) < 2:
                    return
                parent, dist_p = dists[0]
                second_closest, dist_s = dists[1]
                no_children = not second_closest.children
                similar_distance = dist_s - dist_p < ExtractorConfig.DIAG_LABEL_MAX_REASSIGNMENT_DISTANCE
                if no_children and similar_distance:
                    d.children.remove(label)
                    second_closest.children.append(label)

    def postprocess_diagrams(self, out_diag_boxes: List['Diagram']) -> List['Diagram']:
        """Postprocesses diagram bounding boxes from the detectron model.
        First, select a prior as the largest connected component inside each detection,
        Then filter false positives by considering overlap with arrows
        Finally, perform dilation and collect final diagram predictions inside the diagram extractor
        :param out_diag_boxes: diagram bounding box predictions from the object detection model.
        :type out_diag_boxes: list[Diagram]
        """
        diag_priors = [self.select_diag_prior(bbox) for bbox in out_diag_boxes]
        diag_priors = [Panel(diag) for diag in diag_priors if diag]
        # diag_priors = self.filter_diag_false_positives(diag_priors)
        self.diagram_extractor.diag_priors = diag_priors
        diags = self.diagram_extractor.extract()
        return diags

    def postprocess_text_regions(self, 
                                 text_regions: List['TextRegionCandidate']) -> Tuple[List['Conditions'], List['Label']]:
        """Postprocesses conditions and labels detections.
        First, bounding boxes are adjusted to match connected component boundaries exactly.
        Then, they are conditionally reclassified, and duplicated removed
        Conditions are further cleaned if they aren't matched to any arrow
        An image is then created where all arrows and chemical diagrams are erased. This image is used for text OCR process.
        The two lists are then passed to their individual extractors for text recognition and parsing
        Finally, detections with poor text detection confidence are filtered out
        :param text_regions: detected text regions (labels + conditions)
        :type text_regions: List[TextRegionCandidate]
        :rtype: Tuple[List['Conditions'], List['Label']"""
        adjusted_candidates = self.adjust_bboxes(text_regions)
        if not self.diags_only:
            conditions, labels = self.reclassify(adjusted_candidates)
            conditions, labels = [self.remove_duplicates(group) for group in [conditions, labels]]
            conditions = self.clean_conditions(conditions)
        else:
            labels = adjusted_candidates
            conditions = []
        
        self._set_ocr_fig()

        conditions, labels = [self.extract_elements(group, extractor) for group, extractor
                              in zip([conditions, labels], [self.conditions_extractor, self.label_extractor])]
        
        if not self.diags_only:
            self.conditions_extractor._extracted = self._filter_text_false_positives(conditions, self.diagram_extractor.extracted)
        self.label_extractor._extracted = self._filter_text_false_positives(labels, self.diagram_extractor.extracted)

        return self.conditions_extractor.extracted, self.label_extractor.extracted

    def _set_ocr_fig(self):
        panels_to_mask = [d.panel for d in self.diagram_extractor.extracted] + [a.panel for a in self._arrows]
        self.masked_fig = erase_elements(self.fig, panels_to_mask, copy_fig=False)
        to_isolate = [cc for cc in self.masked_fig.connected_components]
        all_coords = np.array([cc.coords for cc in to_isolate])
        all_coords = (all_coords / self.fig.scaling_factor).astype(np.int32)
        img_canvas = np.zeros_like(self.masked_fig.raw_img)
        for coord in all_coords:
            top, left, bottom, right = coord
            img_canvas[top:bottom+1, left:right+1] = self.masked_fig.raw_img[top:bottom+1, left:right+1]
        img_canvas = cv2.resize(img_canvas, (self.masked_fig.img.shape[1], self.masked_fig.img.shape[0]))
        ocr_fig = Figure(img_canvas, self.masked_fig.raw_img)

        self.conditions_extractor.ocr_fig = ocr_fig
        self.label_extractor.ocr_fig = ocr_fig
        
    def set_parents_for_text_regions(self, 
                                     text_regions: List['TextRegionCandidate'], 
                                     possible_parents: List[Union['BaseArrow', 'Diagram']]):
        """Assigns each text region to a diagram (for labels) or arrow (for conditions) parent object based on distance
        and relative orientation between the child and potential parent
        :param text_regions: detected text regions (labels or conditions)
        :type text_regions: list[Conditions|Label]
        :param possible_parents: list of possible parent regions (arrows for conditions, and diagrams for labels
        :type possible_parents: list[BaseArrow|Diagram]"""
        orientations = []
        for region in text_regions:
            closest_possible_parent = min(possible_parents, key=region.panel.edge_separation)
            orientations.append(closest_possible_parent.panel.find_relative_orientation(region.panel))
        num_overlapping = len([orient for orient in orientations if sum(orient)== 0])
        orientations = list(zip(*orientations))
        
        # if half of labels are below their nearest possible parent, use this information to deduce the correct parent
        below_panel = sum(orientations[2]) > 0.5 * (len(text_regions) - num_overlapping)
        for region in text_regions:
            region.set_nearest_as_parent(possible_parents, below_panel=below_panel)

    def _filter_text_false_positives(self, text_regions, diags):
        """Filter out false positives: parts of diagrams (usually superatom labels) falsely marked as conditions or labels, and plus signs.
        :param text_regions: detected text regions (labels or conditions)
        :type text_regions: list[Conditions|Label]
        :param diags: detected diagrams
        :type diags: list[Diagram]
        """
        with open(superatom_file) as file:
            superatoms = [token.strip() for line in file.readlines() for token in line.split(' ')]

        regions_overlapping_diags = []
        
        to_keep = list(range(len(text_regions)))
        for idx_to_remove, region in enumerate(text_regions):
            text = region.text
            if isinstance(text, list):
                text = ' '.join(text)
            if text is None:
                continue
            plus_matched = re.search(r'\+', text)
            if plus_matched and len(text) < 4:
                to_keep.remove(idx_to_remove)
            elif any(compute_ioa(region.panel, diag) > ExtractorConfig.UNIFIED_IOA_FILTER_THRESH for diag in diags):
                regions_overlapping_diags.append(region)

        filtered_regions = [text_regions[idx] for idx in to_keep]
        
        to_keep = list(range(len(filtered_regions)))
        for idx, region in enumerate(regions_overlapping_diags):
            is_superatom = False
            text = region.text
            if isinstance(text, list):
                text = ' '.join(text)
            for superatom in superatoms:
                matched = re.search(superatom, text)
                # The text region should consist of the superatom and be short enough:
                if matched and len(text) <= len(superatom) + 4:
                    is_superatom = True
                    break
            if is_superatom:
                to_keep.remove(idx)
                
        filtered_regions = [text_regions[idx] for idx in to_keep]

        return filtered_regions

    def clean_conditions(self, conditions: List['Conditions']) -> List['Conditions']:
        """Cleans poor conditions predictions which cover large image patches or cannot be associated with any arrow.
        :param conditions: postprocessed conditions regions
        :type conditions: list[Conditions]
        """
        # Clean patches that span way beyond the arrow along its direction -- this might not be necessary

        max_diag_area = np.max([d.panel.area for d in self.diagram_extractor.diags])
        filtered_conditions = []
        for c in conditions:
            closest_arrow = self._find_closest(c, self.all_arrows)
            dist = c.panel.edge_separation(closest_arrow.panel)
            well_optimized = c.panel.area < ExtractorConfig.CONDITIONS_MAX_AREA_FRACTION * max_diag_area
            close_to_arrow = dist < ExtractorConfig.CONDITIONS_ARROW_MAX_DIST
            if well_optimized and close_to_arrow:
                filtered_conditions.append(c)

        return filtered_conditions

    def plot_extracted(self, ax):
        """Plots extracted objects on ``ax``
        :param ax: axes object on which to plot
        :type ax: plt.Axes"""
        self.diagram_extractor.plot_extracted(ax)
        self.label_extractor.plot_extracted(ax)
        self.conditions_extractor.plot_extracted(ax)

    def remove_duplicates(self, panels: List[Union['Label','Conditions']]) -> List['Diagram']:
        """Removes duplicate panels inside `group`. In this context, duplicates are all panels which cover the same
        region, whether an entire region or just a part of it. In such cases, the biggest panel is selected.

        :param panels: panels to be analysed
        :type panels: list[Panel]
        :return: filtered, unique panels
        :rtype: list[Panel]"""
        dists = np.array([[p1.edge_separation(p2) for p1 in panels] for p2 in panels])
        not_yet_grouped = set(list(range(len(panels))))
        if len(not_yet_grouped) == 1:
            return panels
        groups = []
        idx1 = 0
        while idx1 <= len(panels) - 1:
            if idx1 not in not_yet_grouped:
                idx1 += 1
                continue
            group = [panels[idx1]]
            not_yet_grouped.remove(idx1)
            for idx2 in range(len(panels)):
                if idx2 in not_yet_grouped and dists[idx1, idx2] == 0:
                    group.append(panels[idx2])
                    not_yet_grouped.remove(idx2)
            groups.append(group)
            idx1 += 1

        filtered = []
        for group in groups:
            largest = max(group, key=lambda p: p.area)
            filtered.append(largest)
            for panel in group:
                if not largest.contains(panel):
                    filtered.append(panel)

        return filtered

    def extract_elements(self, 
                         elements: List[Union['Label', 'Conditions']],
                         extractor: Union['LabelExtractor', 'ConditionsExtractor']):
        """Wrapper method used to set priors to an extractor prior to calling its main extract method
        :param elements: priors/elements to perform extraction on
        :type: list[Label|Conditions]
        :param extractor: extractor to be used for extraction
        :type extractor: BaseExtractor
        :return: extracted elements
        :rtype: list[Label|Conditions"""
        extractor.priors = elements
        return extractor.extract()

    def select_diag_prior(self, 
                          bbox: np.ndarray) -> np.array:
        """Selects diagram prior as the largest connected component bounded by the ``bbox``
        :param bbox: detection from the object detection model
        :type bbox: np.ndarray
        :return: the largest connected component in main figure coordinates
        :type: np.ndarray"""
        bbox_crop = Crop(self.fig, bbox)
        if any([s == 0 for s in bbox_crop.img.shape]):
                return
        try:
            prior = max(bbox_crop.connected_components, key=lambda cc: cc.area)
        except ValueError: # no connected components in the crop
            return None
        return bbox_crop.in_main_fig(prior)

    def filter_diag_false_positives(self, 
                                    diag_priors: List[Panel]) -> List['Diagram']:
        """Filters diagram false positives by comparing with all extracted arrows. If a given region was marked
        as an arrow by the arrow extraction model, then it is removed from a list of potential diagrams
        :param diag_priors: all diagram priors
        :type diag_priors: list[Panel]
        :return: filtered, unique diagrams
        :rtype: list[Diagram]
        """
        filtered_diags = []
        for diag in diag_priors:
            is_diag = diag is not None
            if is_diag and all(diag.compute_iou(arrow) < ExtractorConfig.UNIFIED_DIAG_FP_IOU_THRESH for arrow in self.all_arrows):
                filtered_diags.append(diag)

        return filtered_diags

    def adjust_bboxes(self, 
                      region_candidates: List[np.ndarray]) -> List[np.ndarray]:
        """Adjusts bboxes inside each of region_candidates by restricting them to cover connected components fully contained within crops bounded
        by the bboxes
        :param region_candidates: detections from the object detection model
        :type region_candidates: list[np.ndarray]
        :return: detections adjusted to connected component boundaries
        :rtype: list[np.ndarray]"""

        adjusted = []
        for cand in region_candidates:
            crop = cand.panel.create_extended_crop(self.fig, extension=1)
            if any([s == 0 for s in crop.img.shape]):
                continue
            relevant_ccs = [crop.in_main_fig(cc) for cc in crop.connected_components]
            if relevant_ccs:
                cand.panel = Panel.create_megapanel(relevant_ccs, self.fig)
                adjusted.append(cand)

        return adjusted

    def reclassify(self, candidates: List[Union['Conditions', 'Label']]):
        """Attempts to reclassify labels and conditions candidates based on proximity to arrows and diagrams.
        If a candidate is close to an arrow and has the correct relative orientation, then it is reclassified as a conditions region. 
        Conversely, if it is close to a diagram, then it is classified as a labels. Finally, if it is not within a given threshold distance 
        or does not have expected relative orientation, it is instantiated based on the prior labels given by the unified detection model. 
        Assigns the closest of these panels as a parent panel
        :param candidates: analysed text region predictions
        :type candidates: list[Conditions|Label]
        :return: textual elements with their final classes
        :rtype: list[Conditions|Label]"""

        conditions = []
        labels = []
        for cand in candidates:
            closest_arrow = self._find_closest(cand, self.all_arrows)
            closest_diag = self._find_closest(cand, self.diagram_extractor.extracted)
            cand_class = self._adjust_class(cand, ({'arrow': closest_arrow, 'diag': closest_diag}))

            if cand_class == Conditions:
                conditions.append(cand)
            else:
                labels.append(cand)

        return conditions, labels

    def _adjust_class(self, obj, closest):
        class_ = self._class_dict[obj._prior_class]
        seps = {k: obj.edge_separation(v) for k, v in closest.items()}
        thresh_reclassification_dist = ExtractorConfig.UNIFIED_RECLASSIFY_DIST_THRESH_COEFF * np.sqrt(obj.area)

        if seps['arrow'] <= seps['diag'] and seps['arrow'] < thresh_reclassification_dist:
            if lies_along_arrow_normal(closest['arrow'], obj):
                class_ = Conditions

        elif seps['diag'] < seps['arrow'] and seps['diag'] < thresh_reclassification_dist:
            ## Adjust to labels if it's somewhere below the diagram
            ## 'below' is defined as an angle between 135 and 225 degrees (angle desc in function below)
            direction = find_relative_directional_position(obj.center, closest['diag'].center)
            if 225 > direction > 135:
                class_ = Label

        # Visualize pair
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # f, ax = plt.subplots()
        # ax.imshow(self.fig.img, cmap=plt.cm.binary)
        # panel = obj.panel
        # rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
        #                       facecolor=(52 / 255, 0, 103 / 255), edgecolor=(6 / 255, 0, 99 / 255), alpha=0.4)
        # ax.add_patch(rect_bbox)
        # panel = closest['diag'].panel
        # rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
        #                       facecolor=(52 / 255, 0, 103 / 255), edgecolor=(6 / 255, 0, 99 / 255), alpha=0.4)
        # ax.add_patch(rect_bbox)
        # plt.show()

        return class_

    def to_json(self):
        out_lst = []
        diags = self._extracted[0]
        for diag in diags:
            diag_dct = {'smiles': diag.smiles, 'panel': str(diag.panel.in_original_fig()),
                        'labels': [label.text for label in diag.labels]}
            repeating_units = [{'identifier':fragment.label.text, 'smiles':fragment.smiles} for fragment in diag.repeating_units]
            if repeating_units:
                diag_dct['repeating_units'] = repeating_units
            out_lst.append(diag_dct)
        return json.dumps(out_lst, indent=4)

    # def visualize_diag_label_matching(self, diag):
    #     _X_SEPARATION = 50
    #     elements = [diag] + diag.labels
    #     orig_coords = [e.panel.in_original_fig() for e in elements]
    #
    #     canvas_width = np.sum([c[3] - c[1] for c in orig_coords]) + _X_SEPARATION * (len(elements) - 1)
    #     canvas_height = max([c[2] - c[0] for c in orig_coords])
    #
    #     canvas = np.zeros([canvas_height, canvas_width])
    #     x_end = 0
    #
    #     self._place_panel_on_canvas(diag.panel, canvas, self.fig,  (x_end, 0))
    #     orig_coords = diag.panel.in_original_fig()
    #     x_end += orig_coords[3] - orig_coords[1] + _X_SEPARATION
    #
    #     for labels in diag.labels:
    #         self._place_panel_on_canvas(labels.panel, canvas, self.fig, (x_end, 0))
    #         orig_coords = labels.panel.in_original_fig()
    #         x_end += orig_coords[3] - orig_coords[1] + _X_SEPARATION
    #
    #     return canvas

    # def visualize_label_matchings(self, diagrams):
    #     canvases = [self.visualize_diag_label_matching(diag) for diag in diagrams]
    #     _Y_SEPARATION = 50
    #     out_canvas_height = np.sum([c.shape[0] for c in canvases]) + _Y_SEPARATION * (len(canvases) - 1)
    #     out_canvas_width = np.max([c.shape[1] for c in canvases])
    #     out_canvas = np.zeros([out_canvas_height, out_canvas_width])
    #     y_end = 0
    #     for canvas in canvases:
    #         h, w = canvas.shape
    #         out_canvas[y_end:y_end + h, 0:0 + w] = canvas
    #         y_end += h + _Y_SEPARATION
    #
    #     plt.imshow(out_canvas)
    #     plt.show()

    def _find_closest(self, obj, other_objects):
        """
        Measure the distance between 'panel' and 'other_objects' to find the object that is closest to the `panel`
        """
        #TODO: This should be a method inside a `Panel` class (?)
        dists = [(other_obj, obj.edge_separation(other_obj)) for other_obj in other_objects]
        return sorted(dists, key=lambda elem: elem[1])[0][0]

    # def _place_panel_on_canvas(self, panel, canvas,fig,  left_top):

    #     ## Specify coords of the paste region
    #     x, y = left_top
    #     w, h = panel.width, panel.height

    #     ## Specify coords of the crop region
    #     top, left, bottom, right = panel

    #     canvas[y:y+h, x:x+w] = fig.img[top:bottom, left:right]


class DiagramExtractor(BaseExtractor):
    """Diagram extraction class. This class is responsible for refining the initial object detection outputs.
    It takes the priors, and performs dilation around them to complete the chemical diagrams"""

    def __init__(self, fig: 'Figure',
                 diag_priors: List[Panel], 
                 arrows: List['BaseArrow']):
        """Init method. Requires the main figure, panels containing the priors, and all arrows from the
        arrow extraction model

        :param fig: Figure containing the whole reaction scheme
        :type fig: Figure
        :param diag_priors: List of diagram prior panels
        :type diag_priors: List[Panel]
        :param arrows: all arrows detected in the reaction scheme
        :type arrows: List['BaseArrow']
        """
        super().__init__(fig)
        self.diag_priors = diag_priors
        self.diags = None
        self._arrows = arrows
        self._diags_only = False

    def extract(self):
        """Main extraction method.

        Extract diagrams using diagram priors and dilating around them, and later collecting all individual
        connected components from the original image
        :return: final diagram predictions
        :rtype: list[Diagram]"""
        assert self.diag_priors is not None, "Diag priors have not been set"
        self.fig.dilation_iterations = self._find_optimal_dilation_extent()

        diag_panels = self.complete_structures()
        diags = [Diagram(panel=panel) for panel in diag_panels]
        self.diags = diags

        return self.diags

    @property
    def extracted(self):
        return self.diags

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        if not self.extracted:
            pass
        else:
            for panel in self.extracted:
                rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                      facecolor=(52/255, 0, 103/255), edgecolor=(6/255, 0, 99/255), alpha=0.4)
                ax.add_patch(rect_bbox)

    def complete_structures(self) -> List['Diagram']:
        """
        Dilates a figure and uses priors to find complete chemical structures (prior + superatoms etc.).

        Figure is dilated around each prior according to density of features around it. The diagrams are derived from
        the dilated backbones. Roles are assigned to the disconnected diagram parts.
        :return: bounding boxes of chemical structures
        :rtype: list[Diagram]
        """
        dilated_structure_panels, other_ccs = self.find_dilated_structures()
        structure_panels = self._complete_structures(dilated_structure_panels)
        self._assign_diagram_parts(structure_panels, other_ccs)  # Assigns cc roles
        
        # simple filtering to account for potential multiple priors corresponding to the same diagram
        duplicated = []
        for idx_to_remove, panel1 in enumerate(structure_panels):
            for panel2 in structure_panels:
                if panel2.contains(panel1) and panel2 != panel1:
                    duplicated.append(idx_to_remove)

        duplicated = set(duplicated)
        unique = [structure_panels[idx]  for idx in range(len(structure_panels)) if idx not in duplicated]

        return list(set(unique))

    def find_dilated_structures(self) -> Tuple[List[Panel]]:
        """
        Finds dilated structures by first dilating the image several times depending on the density of features.

        For each backbone, the figure is dilated a number of times dependent on the density of features.
        Dilated structure panel is then found based on comparison with the original prior. A crop is made for each
        structure. If there is more than one connected component that is fully contained within the crop,
        it is noted and this information used later when the small disconnected ccs are assigned roles
        (This additional connected component is likely a label).
        :return: (dilated_structure_panels, other_ccs) pair of collections containing the dilated panels and
        separate ccs present within these dilated panels
        :rtype: tuple of lists
        """
        fig = self.fig
        dilated_structure_panels = []
        other_ccs = []
        dilated_figs = {}
        for diag in self.diag_priors:
            num_iterations = fig.dilation_iterations[diag]
            try:
                dilated_temp = dilated_figs[num_iterations] # use cached
            except KeyError:
                dilated_temp = dilate_fig(erase_elements(fig, [a.panel for a in self._arrows], copy_fig=False),
                                          num_iterations)
                dilated_figs[num_iterations] = dilated_temp

            # try:
            corresponding_panels_in_dilated_fig = [cc for cc in dilated_temp.connected_components if cc.contains(diag)]
            if len(corresponding_panels_in_dilated_fig) > 0:
                dilated_structure_panel = min(corresponding_panels_in_dilated_fig, key=lambda panel: panel.area)
            else:
                continue
            # except IndexError: ## Not found
            #     continue
            # Crop around with a small extension to get the connected component correctly
            structure_crop = dilated_structure_panel.create_extended_crop(dilated_temp, extension=5)
            other = [structure_crop.in_main_fig(c) for c in structure_crop.connected_components if
                     structure_crop.in_main_fig(c) != dilated_structure_panel]
            other_ccs.extend(other)
            dilated_structure_panels.append(dilated_structure_panel)
        return dilated_structure_panels, other_ccs

    def _assign_diagram_parts(self, structure_panels: List[Panel], cno_ccs:List[Panel]) -> None:
        """
        Assigns roles to small disconnected diagram parts.

        Takes in the detected structures panels and ccs that are contained inside structure panels but are
        non-overlapping  (``cno_ccs``) - including in the dilated figure. Assigns roles to all (small) connected
        components contained within structure panels, and finally resets role for the special ``cno_ccs``. These are
        likely to be labels lying very close to the diagrams themselves.
        :param [Panel,...] structure_panels: iterable of found structure panels
        :param [Panel,...] cno_ccs: contained-non-overlapping cc;ccs that are not parts of diagrams even though
        their panels are situated fully inside panels of chemical diagrams (common with labels).
        :return: None (mutates ''role'' attribute of each relevant connected component)
        """
        fig = self.fig

        for parent_panel in structure_panels:
            for cc in fig.connected_components:
                if parent_panel.contains(cc):  # Set the parent panel for all
                    setattr(cc, 'parent_panel', parent_panel)
                    if cc.role != FigureRoleEnum.DIAGRAMPRIOR:
                        # Set role for all except backbone which had been set
                        setattr(cc, 'role', FigureRoleEnum.DIAGRAMPART)

        for cc in cno_ccs:
            # ``cno_ccs`` are dilated - find raw ccs in ``fig``
            fig_ccs = [fig_cc for fig_cc in fig.connected_components if cc.contains(fig_cc)]

            [setattr(fig_cc, 'role', None) for fig_cc in fig_ccs]


    def _complete_structures(self, dilated_structure_panels: List[Panel]) -> List[Panel]:
        """Uses ``dilated_structure_panels`` to find all constituent ccs of each chemical structure.

        Finds connected components belonging to a chemical structure and creates a large panel out of them. This
        effectively normalises panel sizes to be independent of chosen dilation kernel sizes.
        :return [Panel,...]: iterable of Panels bounding complete chemical structures.
        """
        structure_panels = []
        disallowed_roles = [FigureRoleEnum.ARROW]
        for dilated_structure in dilated_structure_panels:
            constituent_ccs = [cc for cc in self.fig.connected_components if dilated_structure.contains_any_pixel_of(cc)
                               and cc.role not in disallowed_roles]
            parent_structure_panel = Panel.create_megapanel(constituent_ccs, fig=self.fig)
            if self._diags_only or parent_structure_panel.area/self.fig.area < ExtractorConfig.DIAG_MAX_AREA_FRACTION:
                structure_panels.append(parent_structure_panel)
        return structure_panels

    def _find_optimal_dilation_extent(self) -> int:
        """
        Use structural prior to calculate local skeletonised-pixel ratio and find optimal dilation kernel extent
        (in terms of number of iterations) for structural segmentation. Each diagram priod is assigned its own
        dilation extent to account for varying skel-pixel ratio around different priors
        :return: number of dilation iterations appropriate for each diagram prior
        :rtype: dict
        """
        num_iterations = {}
        img = cv2.ximgproc.thinning(self.fig.img)
        for prior in self.diag_priors:
            top, left, bottom, right = prior
            horz_ext, vert_ext = prior.width // 2, prior.height // 2
            horz_ext = max(horz_ext, ExtractorConfig.DIAG_DILATION_EXT)
            vert_ext = max(vert_ext, ExtractorConfig.DIAG_DILATION_EXT)
            crop_rect = Rect((top - vert_ext, left - horz_ext, bottom + vert_ext, right + horz_ext ))
            p_ratio = pixel_ratio(img, crop_rect)

            if p_ratio >= 0.03:
                n_iterations = 4
            elif 0.005 < p_ratio < 0.03:
                n_iterations = np.ceil(10 - 200 * p_ratio)
            else:
                n_iterations = 18
            num_iterations[prior] = n_iterations
        return num_iterations


class TextRegionCandidate(Candidate, PanelMethodsMixin):
    """This class is used to represent regions which consist of text and contain either labels or reaction conditions.
    In the pipeline, outputs marked by the unified model as conditions or labels are instantiated as objects of this class"""

    def __init__(self, bbox, class_):
        self.panel = Panel(bbox)
        self._prior_class = class_


class Detectron2Adapter:
    """Adapter to the object detection model. Wraps the model to ensure output compatibility with rest of the pipeline.
    Performs necessary preprocessing steps, runs predictions using detectron2, and applies postprocessing"""

    ### detectron2 configs ###
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = ExtractorConfig.DEVICE
    cfg.MODEL.WEIGHTS = ExtractorConfig.UNIFIED_EXTR_MODEL_WT_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16], [16, 32], [32, 64], [64, 128], [256, 512]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    def __init__(self, fig: Figure, use_tiler:bool=True):
        """
        :param fig: Analysed figure
        :type fig: Figure
        :param use_tiler: Whether to divide the figure into patches and run small object detection on those
        :type use_tiler: bool
        """
        self.model = Rde2Predictor(self.cfg)
        self.fig = fig
        self.use_tiler = use_tiler

    def detect(self) -> Tuple[np.ndarray]:
        """Detects the objects and applies postprocessing (changes the order of coordinates to match pipeline's
        convention and rescales according to the image size used in the main pipeline)
        :return: postprocessed detections
        :rtype: tuple[list]"""
        boxes, classes = self._detect()
        boxes = self.adjust_coord_order_detectron(boxes)
        boxes = self.rescale_to_img_dims(boxes)

        return boxes, classes

    def _detect(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self.use_tiler:
                with torch.no_grad():
                    predictions = self.model([self.fig.img_detectron])[0]
                    predictions = predictions['instances']
            else:
                tiler = ImageTiler(self.fig.img_detectron, ExtractorConfig.TILER_MAX_TILE_DIMS, main_predictions=None)
                tiles = tiler.create_tiles()
                BATCH_SIZE = 4
                batches = np.arange(BATCH_SIZE, tiles.shape[0], BATCH_SIZE)
                tiles_batches = np.split(tiles, batches)
                predictions = []
                with torch.no_grad():
                    predictions.append(self.model([self.fig.img_detectron])[0])
                    for batch in tiles_batches:
                        predictions += self.model(batch)
                main_predictions = predictions[0]['instances']
                tiler.main_predictions = main_predictions
                tile_predictions = predictions[1:]

                tile_predictions = tiler.transform_tile_predictions(tile_predictions)
                tile_predictions = tiler.filter_small_boxes(tile_predictions)
                predictions = self.combine_predictions(main_predictions, tile_predictions)

            high_scores = predictions.scores.numpy() > ExtractorConfig.UNIFIED_PRED_THRESH
            pred_boxes = predictions.pred_boxes.tensor.numpy()[high_scores]
            pred_classes = predictions.pred_classes.numpy()[high_scores]

        # visualize predictions
        # from detectron2.utils.visualizer import Visualizer
        # import matplotlib.pyplot as plt
        # vis = Visualizer(self.fig.img_detectron)
        # vis.draw_instance_predictions(predictions['instances'])
        # plt.imshow(vis.output.get_image(), cmap=plt.cm.binary)
        # plt.title('full_image')
        # plt.show()
        # for i, p in zip([i1, i2, i3, i4], [p1, p2, p3, p4]):
        #     vis = Visualizer(i)
        #     vis.draw_instance_predictions(p['instances'])
        #     plt.imshow(vis.output.get_image(), cmap=plt.cm.binary)
        #     plt.show()
        return pred_boxes, pred_classes,

    def adjust_coord_order_detectron(self, boxes:np.ndarray) -> np.ndarray:
        """Adjusts order of coordinates to the expected format

        Detectron outputs boxes with coordinates (x1, y1, x2, y2), however the expected
        format is (top, left, bottom, right) i.e. (y1, x1, y2, x2). This function switches
        the coordinates to achieve consistency
        :param boxes: array containing detected bounding boxes
        :type boxes: np.ndarray"""
        boxes = boxes[:, [1, 0, 3, 2]]
        return boxes

    def rescale_to_img_dims(self, boxes: np.ndarray) -> np.ndarray:
        """Rescales detections to match the rescaled figure used in the main pipeline
        :param boxes: detectron2's detections
        :type boxes: np.ndarray
        :return: rescaled bounding boxes, in line with the image dimensions used in the main pipeline
        :rtype: np.ndarray"""
        if self.fig.scaling_factor:
            boxes = boxes * self.fig.scaling_factor
        return boxes.astype(np.int32)

    def combine_predictions(self, *preds: 'Instances') -> 'Instances':
        """Combines predictions from the whole image, and small object detection from tiles
        :param preds: predictions from the whole image, and from image tiles
        :type preds: Instances
        :return: combined predictions as Instances object
        :rtype: Instances"""

        boxes = Boxes(Tensor(np.concatenate([p.pred_boxes.tensor.numpy() for p in preds], axis=0)))
        classes = Tensor(np.concatenate([p.pred_classes.numpy() for p in preds], axis=0))
        scores = Tensor(np.concatenate([p.scores.numpy() for p in preds], axis=0))

        instances = Instances(image_size=self.fig.img.shape[:2])
        instances.set('pred_boxes', boxes)
        instances.set('pred_classes', classes)
        instances.set('scores', scores)
        return instances


class ImageTiler:
    """Class used for creating, handling, and postrprocessing of image tiles.
    The image is divided into several patches, which allow more precise detection of smaller objects"""
    def __init__(self, img, max_tile_dims, main_predictions, extension=100):
        self.img = img
        self.max_tile_dims = max_tile_dims
        self.main_predictions = main_predictions

        self.extension = extension
        self.tile_dims = []

    def create_tiles(self) -> np.ndarray:
        """Creates image tiles.
        Splits image into segments of dims no larger than tile_dims by dividing it into NxM overlapping patches
        :return: image patches
        :rtype: np.ndarray"""
        h, w = self.img.shape[:2]
        tile_h, tile_w = self.max_tile_dims
        num_h_divisor_points = ceil(h / tile_h) + 1
        num_w_divisor_points = ceil(w / tile_w) + 1
        h_segments = np.linspace(0, h, num_h_divisor_points, dtype=np.int32)
        w_segments = np.linspace(0, w, num_w_divisor_points, dtype=np.int32)
        h_segments = zip(h_segments[:-1], h_segments[1:])
        w_segments = zip(w_segments[:-1], w_segments[1:])
        tiles_dims = product(h_segments, w_segments)
        tiles = []
        for dims in tiles_dims:
            # print(dims)
            (h_start, h_end), (w_start, w_end) = dims

            h_start = h_start - self.extension
            h_end = h_end + self.extension
            w_start = w_start - self.extension
            w_end = w_end + self.extension

            h_start = max(h_start, 0)
            h_end = min(h_end, h)
            w_start = max(w_start, 0)
            w_end = min(w_end, w)

            tile = self.img[h_start:h_end, w_start:w_end]
            self.tile_dims.append(((h_start, h_end), (w_start, w_end)))
            tiles.append(tile)
        h_max = max(t.shape[0] for t in tiles)
        w_max = max(t.shape[1] for t in tiles)
        tiles = [cv2.resize(t, (w_max, h_max)) for t in tiles]
        return np.stack(tiles, 0)

    def filter_small_boxes(self, instances: 'Instances'):
        """Filters small predictions from image patch detections to add them to the main predictions

        :param instances: detections from images tiles
        :type: Instances
        :return: filtered small detections
        :rtype: Instances"""
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()

        def area(box):
            x0, y0, x1, y1 = box
            return(x1 - x0) * (y1 - y0)

        main_pred_boxes = self.main_predictions.pred_boxes.tensor.numpy()
        thresh_area = np.percentile([area(b) for b in main_pred_boxes], ExtractorConfig.TILER_THRESH_AREA_PERCENTILE)

        filter_idx = [i for i in range(boxes.shape[0]) if area(boxes[i]) < thresh_area]
        boxes = Boxes(boxes[filter_idx])
        classes = Tensor(classes[filter_idx])
        scores = Tensor(scores[filter_idx])
        instances = self.create_detectron_instances(boxes, classes, scores)
        return instances

    def transform_tile_predictions(self, preds: List['Instances']):
        """Transforms coordinates of predictions from tile into the main image coordinate system
        :param preds: all predictions from image tiles
        :type preds: list[Instances]
        :return: transformed predictions packed into a single Instances object
        :rtype: Instances"""
        preds_transformed_boxes = []
        preds_transformed_classes = []
        preds_transformed_scores = []
        for tile_preds, tile_dims in zip(preds, self.tile_dims):
            ((h_start, h_end), (w_start, w_end)) = tile_dims
            tile_preds = tile_preds['instances']
            classes = tile_preds.pred_classes.numpy()
            scores = tile_preds.scores.numpy()

            x0, y0, x1, y1 = np.split(tile_preds.pred_boxes.tensor.numpy(), 4, axis=1)
            x0, x1 = x0 + w_start, x1 + w_start
            y0, y1 = y0 + h_start, y1 + h_start
            tile_preds_transformed = np.concatenate((x0, y0, x1, y1), axis=1)
            preds_transformed_boxes.extend(tile_preds_transformed.tolist())
            preds_transformed_classes.extend(classes.tolist())
            preds_transformed_scores.extend(scores.tolist())
        boxes = Boxes(Tensor(preds_transformed_boxes))
        instances = self.create_detectron_instances(boxes, Tensor(preds_transformed_classes),
                                                    Tensor(preds_transformed_scores))
        return instances
            # Translate the coords using  and w_start as the translation vector

    def create_detectron_instances(self, boxes, classes, scores):
        """Helper function used to transform raw boxes, classes and scores (torch.Tensor) into a single
        detectron2 Instances object
        :param boxes: predicted bounding boxes
        :type boxes: torch.Tensor
        :param classes: predicted classes
        :type classes: torch.Tensor
        :param scores: prediction scores
        :type scores: torch.Tensor
        :return: packed Instances object containing the same boxes, classes and scores
        :type: Instances"""
        instances = Instances(image_size=self.img.shape[:2])
        instances.set('pred_boxes', boxes)
        instances.set('pred_classes', classes)
        instances.set('scores', scores)
        return instances


class Rde2Predictor(DefaultPredictor):
    """Simple custom predictor class"""
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, images):
        """This call method is very similar to that in DefaultPredictor, but supports batched inference"""
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            batched_inputs = []
            for original_image in images:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs = {"image": image, "height": height, "width": width}
                batched_inputs.append(inputs)
            predictions = self.model(batched_inputs)
            return predictions
