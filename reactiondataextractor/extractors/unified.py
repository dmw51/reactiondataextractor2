"""This module contains all parts of the unified extractor for diagrams, labels and reaction conditions"""
import copy
from itertools import product
from math import ceil
import numpy as np
import os
import re
import warnings

import torch
from detectron2.config import get_cfg
from matplotlib import pyplot as plt
from torch import Tensor
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import  Instances
from detectron2.structures.boxes import Boxes
from matplotlib.patches import Rectangle
from scipy.stats import mode
from detectron2 import model_zoo

from reactiondataextractor.models.base import BaseExtractor, Candidate
from reactiondataextractor.extractors.conditions import ConditionsExtractor
from reactiondataextractor.extractors.labels import LabelExtractor
from configs.config import ExtractorConfig
from reactiondataextractor.models.reaction import Diagram, Conditions, Label
from reactiondataextractor.models.segments import Panel, Rect, FigureRoleEnum, Crop, PanelMethodsMixin
from reactiondataextractor.utils import skeletonize_area_ratio, dilate_fig, erase_elements, find_relative_directional_position, \
    compute_ioa, lies_along_arrow_normal

parent_dir = os.path.dirname(os.path.abspath(__file__))
# superatom_file = os.path.join(parent_dir, '..', 'dict', 'superatom.txt')
superatom_file = os.path.join(parent_dir, '..', 'dict', 'filter_superatoms.txt')
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'


class UnifiedExtractor(BaseExtractor):
    """The main object detection model. Combines an underlying detectron2 object detection model, as well
    as the individual diagram, label and conditions extractors"""
    def __init__(self, fig, all_arrows, use_tiler=True):
        """
        :param fig: Analysed figure
        :type fig: Figure
        :param all_arrows: all extracted arrows from the arrow extractor model
        :type all_arrows: list[BaseArrow]
        :param use_tiler: Whether to perform small object detection on image patches
        :type use_tiler: bool
        """
        super().__init__(fig)
        self.model = Detectron2Adapter(fig, use_tiler)
        self._all_arrows = all_arrows
        self.diagram_extractor = DiagramExtractor(self.fig, diag_priors=None, arrows=self.all_arrows)
        self.label_extractor = LabelExtractor(self.fig, priors=None)
        self.conditions_extractor = ConditionsExtractor(self.fig, priors=None)

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
        return self._all_arrows

    @all_arrows.setter
    def all_arrows(self, val):
        self._all_arrows = val
        self.diagram_extractor._arrows = val

    @property
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, val):
        self._fig = val
        self.model.fig = val
        self.diagram_extractor._fig = val
        self.label_extractor._fig = val
        self.conditions_extractor._fig = val

    def extract(self):
        """The main extraction method.

        Processes outputs from the object detection model by delegating to the posprocessing classes and methods.
        Postprocesses diagrams using dilation, then uses these along with arrows to filter viable labels and conditions
        Finally, matches diagrams inside conditions regions with appropriate arrows to extend their conditions regions
        appropriately
        return: postprocessed diagrams, conditions, and labels
        rtype: tuple[list]"""
        boxes, classes = self.model.detect()
        out_diag_boxes = [box for box, class_ in zip(boxes, classes) if self._class_dict[class_] == Diagram]
        diags = self.postprocess_diagrams(out_diag_boxes)
        text_regions = [TextRegionCandidate(box, class_) for box, class_ in zip(boxes, classes)
                        if self._class_dict[class_] in [Label, Conditions]]
        conditions, labels = self.postprocess_text_regions(text_regions)
        self.set_parents_for_text_regions(conditions, self.all_arrows)
        self.set_parents_for_text_regions(labels, diags)
        self.add_diags_to_conditions(diags)
        self._extracted = diags, conditions, labels
        return diags, conditions, labels

    def add_diags_to_conditions(self, diags):
        """Adds diagrams to conditions regions where applicable. This is done by assessing the relationship
        between diagrams and arrows
        :param diags: All extracted diagrams
        :type diags: list[Diagram]"""
        for diag in diags:
            for arrow in self.all_arrows:
                if lies_along_arrow_normal(arrow, diag) and diag.edge_separation(arrow.panel) < ExtractorConfig.ARROW_DIAG_MAX_DISTANCE:
                    arrow.children.append(diag)

    def postprocess_diagrams(self, out_diag_boxes):
        """Postprocesses diagram bounding boxes from the detectron model.
        First, select a prior as the largest connected component inside each detection,
        Then filter false positives by considering overlap with arrows
        Finally, perform dilation and collect final diagram predictions inside the diagram extractor
        :param out_diag_boxes: diagram bounding box predictions from the object detection model.
        :type out_diag_boxes: list[Diagram]
        """
        diag_priors = [self.select_diag_prior(bbox) for bbox in out_diag_boxes]
        diag_priors = [Panel(diag) for diag in diag_priors if diag]
        diag_priors = self.filter_diag_false_positives(diag_priors)
        self.diagram_extractor.diag_priors = diag_priors
        diags = self.diagram_extractor.extract()
        return diags

    def postprocess_text_regions(self, text_regions):
        """Postprocesses conditions and label detections.
        First, bounding boxes are adjusted to match connected component boundaries exactly.
        Then, they are conditionally reclassified, and duplicated removed
        Conditions are further cleaned if they aren't matched to any arrow
        The two lists are then passed to their individual extractors for text parsing
        Finally, detections with poor text detections are filtered
        :param text_regions: detected text regions (labels + conditions)
        :type text_regions: list[TextRegionCandidate]
        :rtype: tuple[list]"""
        adjusted_candidates = self.adjust_bboxes(text_regions)
        conditions, labels = self.reclassify(adjusted_candidates)
        conditions, labels = [self.remove_duplicates(group) for group in [conditions, labels]]
        conditions = self.clean_conditions(conditions)
        conditions, labels = [self.extract_elements(group, extractor) for group, extractor
                              in zip([conditions, labels], [self.conditions_extractor, self.label_extractor])]
        self.conditions_extractor._extracted = self.filter_text_false_positives(conditions, self.diagram_extractor.extracted)
        self.label_extractor._extracted = self.filter_text_false_positives(labels, self.diagram_extractor.extracted)

        return self.conditions_extractor.extracted, self.label_extractor.extracted

    def set_parents_for_text_regions(self, text_regions, possible_parents):
        """Assigns each text region to a diagram (for labels) or arrow (for conditions) parent object based on distance
        :param text_regions: detected text regions (labels or conditions)
        :type text_regions: list[Conditions|Label]
        :param possible_parents: list of possible parent regions (arrows for conditions, and diagrams for labels
        :type possible_parents: list[BaseArrow|Diagram]"""
        for region in text_regions:
            region.set_nearest_as_parent(possible_parents)

    def filter_text_false_positives(self, text_regions, diags):
        """Filter out parts of diagrams (usually superatom labels) falsely marked as conditions or labels, and plus signs.
        :param text_regions: detected text regions (labels or conditions)
        :type text_regions: list[Conditions|Label]
        :param diags: detected diagrams
        :type diags: list[Diagram]
        """
        with open(superatom_file) as file:
            superatoms = [token.strip() for line in file.readlines() for token in line.split(' ')]

        regions_overlapping_diags = []
        filtered_regions = copy.deepcopy(text_regions)

        for region in text_regions:
            text = region.text
            if isinstance(text, list):
                text = ' '.join(text)
            if text is None:
                continue
            plus_matched = re.search(r'\+', text)
            if plus_matched and len(text) < 4:
                filtered_regions.remove(region)
            elif any(compute_ioa(region.panel, diag) > ExtractorConfig.UNIFIED_IOA_FILTER_THRESH for diag in diags):
                regions_overlapping_diags.append(region)

        for region in regions_overlapping_diags:
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
                filtered_regions.remove(region)

        return filtered_regions

    def clean_conditions(self, conditions):
        """Cleans poor conditions prediction which cover large image patches or cannot be associated with any arrow.
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

    def detect(self):
        """A wrapper method used to perform any necessary preprocessing on an input before feeding
        it into the object detection modeland making predictions.
        return: predicted bounding boxes and classes
        rtype: tuple(list(np.ndarray))"""
        if mode(self.fig.img.reshape(-1))[0][0] == 0:
            img = np.invert(self.fig.img)
        else:
            img = self.fig.img
        img = (img - img.min()) / (img.max() - img.min())
        return map(lambda x: x[0].numpy(), self.model.predict(img))  # Predictions for first (and only) image in a batch

    def remove_duplicates(self, panels):
        """Removes duplicate panels inside `group`. In this context, duplicates are all panels which cover the same
        region, whether an entire region or just a part of it. In such cases, the biggest panel is selected.

        The panels are first grouped into sets of overlapping objects and then removed on a group-by-group basis
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

    def extract_elements(self, elements, extractor):
        """Wrapper method used to set priors to an extractor prior to calling its main extract method
        :param elements: priors/elements to perform extraction on
        :type: list[Diagram|Label|Conditions]
        :param extractor: extractor to be used for extraction
        :type extractor: BaseExtractor
        :return: extracted elements
        :rtype: list[Diagram|Label|Conditions"""
        extractor.priors = elements
        return extractor.extract()

    def select_diag_prior(self, bbox):
        """Selects diagram prior as the largest connected component bounded by the ``bbox``
        :param bbox: detection from the object detection model
        :type bbox: np.ndarray
        :return: the largest connected component in main figure coordinates
        :type: np.ndarray"""
        bbox_crop = Crop(self.fig, bbox)
        try:
            prior = max(bbox_crop.connected_components, key=lambda cc: cc.area)
        except ValueError: # no connected components in the crop
            return None
        return bbox_crop.in_main_fig(prior)

    def filter_diag_false_positives(self, diag_priors):
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

    def adjust_bboxes(self, region_candidates):
        """Adjusts bboxes inside each of region_candidates by restricting them to cover connected components fully contained within crops bounded
        by the bboxes
        :param region_candidates: detections from the object detection model
        :type region_candidates: list[np.ndarray]
        :return: detections adjusted to connected component boundaries
        :rtype: list[np.ndarray]"""

        adjusted = []
        for cand in region_candidates:
            crop = cand.panel.create_extended_crop(self.fig, extension=20)
            relevant_ccs = [crop.in_main_fig(cc) for cc in crop.connected_components]
            # unassigned_relevant_ccs = [cc for cc in relevant_ccs if cc.role is None]  # Previously unassigned ccs
            # if unassigned_relevant_ccs: # Remove overlaps with other ccs
            #     cand.panel = Panel.create_megapanel(unassigned_relevant_ccs, self.fig)
            #     adjusted.append(cand)
            if relevant_ccs: # if another cc covers this one entirely, don't remove overlaps
                cand.panel = Panel.create_megapanel(relevant_ccs, self.fig)
                adjusted.append(cand)

        return adjusted

    def reclassify(self, candidates):
        """Attempts to reclassify label and conditions candidates based on proximity to arrows and diagrams.
        If a candidate is close to an arrow, then it is reclassified as a conditions region. Conversely, if it is close
        to a diagram, then it is classified as a label. Finally, if it is not within a given threshold distance, it is
        instantiated based on the prior label given by the unified detection model. Assigns the closest of these panels
        as a parent panel
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

            # cand.set_parent_panel({'arrow': closest_arrow, 'diag': closest_diag}, cand_class)
            ## Try OCR on all of them, then postprocess accordingly and instantiate?
            # reclassified.append(cand_class(**cand.pass_attributes()))
            if cand_class == Conditions:
                conditions.append(cand)
            else:
                labels.append(cand)

        return conditions, labels

    def _adjust_class(self, obj, closest):
        class_ =  self._class_dict[obj._prior_class]
        seps = {k: obj.edge_separation(v) for k, v in closest.items()}
        thresh_reclassification_dist = ExtractorConfig.UNIFIED_RECLASSIFY_DIST_THRESH_COEFF * np.sqrt(obj.area)

        if seps['arrow'] <= seps['diag'] and seps['arrow'] < thresh_reclassification_dist:
            # ### Form 4 points, 2 at end of arrows (or extending a bit further), 2 extending from a line normal to the
            # ### arrow's bounding ellipse, and check which is closest. Reclassify as conditions if obj is closer to
            # ### a normal point
            # (x, y), (MA, ma), angle = cv2.fitEllipse(closest['arrow'].contour)
            # angle = angle - 90 # Angle should be anti-clockwise relative to +ve x-axis
            #
            # normal_angle = angle + 90
            # center = np.asarray([x, y])
            # direction_arrow = np.asarray([1, np.tan(np.radians(angle))])
            # direction_normal = np.asarray([1, np.tan(np.radians(normal_angle))])
            # dist = max(ma, MA) / 2
            # p_a1, p_a2 = find_points_on_line(center, direction_arrow, distance=dist*1.5)
            # p_n1, p_n2 = find_points_on_line(center, direction_normal, distance=dist* 0.5)
            # closest_pt = min([p_a1, p_a2, p_n1, p_n2], key=lambda pt: obj.center_separation(pt))
            # ## Visualize created points
            # # import matplotlib.pyplot as plt
            # # plt.imshow(self.fig.img)
            # # plt.scatter(p_a1[0], p_a1[1], c='r', s=3)
            # # plt.scatter(p_a2[0], p_a2[1], c='r', s=3)
            # # plt.scatter(p_n1[0], p_n1[1], c='b', s=3)
            # # plt.scatter(p_n2[0], p_n2[1], c='b', s=3)
            # # plt.show()
            #
            # if any(np.array_equal(closest_pt, p) for p in[p_n1, p_n2]):
            if lies_along_arrow_normal(closest['arrow'], obj):
                class_ = Conditions



        elif seps['diag'] < seps['arrow'] and seps['diag'] < thresh_reclassification_dist:
            ## Adjust to label if it's somewhere below the diagram
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
    #     for label in diag.labels:
    #         self._place_panel_on_canvas(label.panel, canvas, self.fig, (x_end, 0))
    #         orig_coords = label.panel.in_original_fig()
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

    def _separate_labels_conditions(self, objects):
        conditions, labels =[], []
        for o in objects:
            if isinstance(o, Conditions):
                conditions.append(o)
            elif isinstance(o, Label):
                labels.append(o)
        return conditions, labels

    def _place_panel_on_canvas(self, panel, canvas,fig,  left_top):

        ## Specify coords of the paste region
        x, y = left_top
        w, h = panel.width, panel.height

        ## Specify coords of the crop region
        top, left, bottom, right = panel

        canvas[y:y+h, x:x+w] = fig.img[top:bottom, left:right]


class DiagramExtractor(BaseExtractor):
    """Diagram extraction class"""

    def __init__(self, fig, diag_priors, arrows):
        super().__init__(fig)
        self.diag_priors = diag_priors
        self.diags = None
        self._arrows = arrows

    def extract(self):
        """Main extraction method.

        Extracts diagrams using diagram priors and dilating around them, and later collecting all individual
        connected components from the original image
        :return: final diagram predictions
        :rtype: list[Diagram]"""
        assert self.diag_priors is not None, "Diag priors have not been set"
        self.fig.dilation_iterations = self._find_optimal_dilation_extent()
        self.fig.set_roles(self.diag_priors, FigureRoleEnum.DIAGRAMPRIOR)

        diag_panels = self.complete_structures()
        diags = [Diagram(panel=panel, crop=panel.create_crop(self.fig)) for panel in diag_panels]
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

    def complete_structures(self):
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
        temp = copy.deepcopy(structure_panels)
        # simple filtering to account for potential multiple priors corresponding to the same diagram
        for panel1 in temp:
            for panel2 in temp:
                if panel2.contains(panel1) and panel2 != panel1:
                    try:
                        structure_panels.remove(panel1)
                    except ValueError:
                        pass

        return list(set(structure_panels))

    def find_dilated_structures(self):
        """
        Finds dilated structures by first dilating the image several times depending on the density of features.

        For each backbone, the figure is dilated a number of times dependent n the density of features.
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
                dilated_temp = dilate_fig(erase_elements(fig, [a.panel for a in self._arrows]),
                                          num_iterations)
                dilated_figs[num_iterations] = dilated_temp
            try:
                dilated_structure_panel = [cc for cc in dilated_temp.connected_components if cc.contains(diag)][0]
            except IndexError: ## Not found
                continue
            # Crop around with a small extension to get the connected component correctly
            structure_crop = dilated_structure_panel.create_extended_crop(dilated_temp, extension=5)
            other = [structure_crop.in_main_fig(c) for c in structure_crop.connected_components if
                     structure_crop.in_main_fig(c) != dilated_structure_panel]
            other_ccs.extend(other)
            dilated_structure_panels.append(dilated_structure_panel)

        return dilated_structure_panels, other_ccs

    def _assign_diagram_parts(self, structure_panels, cno_ccs):
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

        # log.debug('Roles of structure auxiliaries have been assigned.')

    def _complete_structures(self, dilated_structure_panels):
        """Uses ``dilated_structure_panels`` to find all constituent ccs of each chemical structure.

        Finds connected components belonging to a chemical structure and creates a large panel out of them. This
        effectively normalises panel sizes to be independent of chosen dilation kernel sizes.
        :return [Panel,...]: iterable of Panels bounding complete chemical structures.
        """
        structure_panels = []
        disallowed_roles = [FigureRoleEnum.ARROW]
        for dilated_structure in dilated_structure_panels:
            constituent_ccs = [cc for cc in self.fig.connected_components if dilated_structure.contains(cc)
                               and cc.role not in disallowed_roles]
            parent_structure_panel = Panel.create_megapanel(constituent_ccs, fig=self.fig)
            if parent_structure_panel.area/self.fig.area < ExtractorConfig.DIAG_MAX_AREA_FRACTION:
                structure_panels.append(parent_structure_panel)
        return structure_panels

    def _find_optimal_dilation_extent(self):
        """
        Use structural prior to calculate local skeletonised-pixel ratio and find optimal dilation kernel extent
        (in terms of number of iterations) for structural segmentation. Each diagram priod is assigned its own
        dilation extent to account for varying skel-pixel ratio around different priors
        :return: number of dilation iterations appropriate for each diagram prior
        :rtype: dict
        """

        # prio = [cc for cc in self.fig.connected_components if cc.role == FigureRoleEnum.STRUCTUREBACKBONE]

        num_iterations = {}
        for prior in self.diag_priors:
            top, left, bottom, right = prior
            horz_ext, vert_ext = prior.width // 2, prior.height // 2
            horz_ext = max(horz_ext, ExtractorConfig.DIAG_DILATION_EXT)
            vert_ext = max(vert_ext, ExtractorConfig.DIAG_DILATION_EXT)

            crop_rect = Rect((top - vert_ext,left - horz_ext, bottom + vert_ext, right + horz_ext ))
            p_ratio = skeletonize_area_ratio(self.fig, crop_rect)
            #print(p_ratio)
            # log.debug(f'found in-crop skel_pixel ratio: {p_ratio}')

            if p_ratio >= 0.03:
                n_iterations = 4
            elif 0.01 < p_ratio < 0.03:
                n_iterations = np.ceil(10 - 200 * p_ratio)
            else:
                n_iterations = 8
            num_iterations[prior] = n_iterations

        return num_iterations


class TextRegionCandidate(Candidate, PanelMethodsMixin):
    """This class is used to represent regions which consist of text and contain either labels or reaction conditions.
    This is determined at a later stage and attributes from each object are transferred to the new instances. In the
    pipeline, outputs marked by the unified model as conditions or labels are instantiated as objects of this class"""

    def __init__(self, bbox, class_):
        self.panel = Panel(bbox)
        self._prior_class = class_
        # self.parent_panel = None


class Detectron2Adapter:
    """Adapter to the object detection model. Wraps the model to ensure output compatibility with rest of the pipeline.
    Performs necessary preprocessing steps, runs predictions using detectron2, and applies postprocessing"""

    ### detectron2 configs ###
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = ExtractorConfig.UNIFIED_EXTR_MODEL_WT_PATH
    # model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16], [16, 32], [32, 64], [64, 128], [256, 512]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64], [64, 128], [128, 256], [128, 256], [256, 512]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ExtractorConfig.UNIFIED_PRED_THRESH

    def __init__(self, fig, use_tiler=True):
        """
        :param fig: Analysed figure
        :type fig: Figure
        :param use_tiler: Whether to divide the figure into patches and run small object detection on those
        :type use_tiler: bool
        """
        self.model = DefaultPredictor(self.cfg)
        self.fig = fig
        self.use_tiler = use_tiler

    def detect(self):
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
            with torch.no_grad():
                predictions = self.model(self.fig.img_detectron)
            #visualize predictions
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

            #Idea: Grab all predictions from the whole image detections, and only the smaller boxes from tiled predictions
            # From big labels, estimate line height and then grab all single line labels from tiles
            # Merge tiled labels if they span multiple tiles
            predictions = predictions['instances']
            if self.use_tiler:
                tiler = ImageTiler(self.fig.img_detectron, ExtractorConfig.TILER_MAX_TILE_DIMS, predictions)
                tiles = tiler.create_tiles()
                tile_preds = [self.model(t) for t in tiles]

                tile_predictions = tiler.transform_tile_predictions(tile_preds)
                tile_predictions = tiler.filter_small_boxes(tile_predictions)
                predictions = self.combine_predictions(predictions, tile_predictions)

            high_scores = predictions.scores.numpy() > ExtractorConfig.UNIFIED_PRED_THRESH
            pred_boxes = predictions.pred_boxes.tensor.numpy()[high_scores]
            pred_classes = predictions.pred_classes.numpy()[high_scores]
        return pred_boxes, pred_classes,

    def adjust_coord_order_detectron(self, boxes):
        """Adjusts order of coordinates to the expected format

        Detectron outputs boxes with coordinates (x1, y1, x2, y2), however the expected
        format is (top, left, bottom, right) i.e. (y1, x1, y2, x2). This function switches
        the coordinates to achieve consistency"""
        boxes = boxes[:, [1, 0, 3, 2]]
        return boxes

    def rescale_to_img_dims(self, boxes):
        """Rescales detections to match the rescaled figure used in the main pipeline
        :param boxes: detectron2's detections
        :type boxes: np.ndarray
        :return: rescaled bounding boxes, in line with the image dimensions used in the main pipeline
        :rtype: np.ndarray"""
        if self.fig.scaling_factor:
            boxes = boxes * self.fig.scaling_factor
        return boxes.astype(np.int32)

    def combine_predictions(self, *preds):
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
    """Class used for creating, handling, and postrprocessing of image tiles"""
    def __init__(self, img, max_tile_dims, main_predictions, extension=100):
        self.img = img
        self.max_tile_dims = max_tile_dims
        self.main_predictions = main_predictions

        self.extension = extension
        self.tile_dims = []

    def create_tiles(self):
        """Creates image tiles.

        Splits image into segments of dims no larger than tile_dims by dividing it into NxM overlapping patches
        :return: image patches
        :rtype: list[np.ndarray]"""
        h, w = self.img.shape[:2]
        tile_h, tile_w = self.max_tile_dims
        num_h_divisor_points = ceil(h / tile_h) + 1
        num_w_divisor_points = ceil(w / tile_w) + 1
        h_segments = np.linspace(0, h, num_h_divisor_points, dtype=np.int32)
        w_segments = np.linspace(0, w, num_w_divisor_points, dtype=np.int32)
        h_segments = zip(h_segments[:-1], h_segments[1:])
        # print(h_segments)
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
        return tiles

    def filter_small_boxes(self, instances):
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

    def transform_tile_predictions(self, preds):
        """Transforms coordinates of predictions from tile into the main image coordinate system
        :param preds: all predictions from image tiles
        :type preds: list[np.ndarray]
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

