"""This module contains all parts of the unified extractor for diagrams, labels and reaction conditions"""
import copy

import numpy as np


from .base import BaseExtractor
from ..config import ExtractorConfig
from ..models.ml_models.unified_detection import RDEModel, RSchemeConfig
from ..models.reaction import Diagram, Conditions, Label
from ..models.segments import Panel, Rect, FigureRoleEnum, Crop
from ..utils import skeletonize_area_ratio, dilate_fig


class UnifiedExtractor(BaseExtractor):

    def __init__(self, fig):
        self.fig = fig
        self.model = self.load_model(ExtractorConfig.UNIFIED_EXTR_MODEL_PATH)

        self._class_dict = {
            1: Diagram,
            2: Conditions,
            3: Label

        }

    def load_model(self, weights_path):
        model_config = RSchemeConfig()
        model = RDEModel(training=False, config=model_config)
        model(np.zeros(model_config.IMAGE_SHAPE), np.zeros((1, 15)))
        model.load_weights(weights_path)
        return model

    def extract(self, all_arrows):
        boxes, classes, scores = self.model.predict(self.img)
        out_diag_bboxes = [box for box, class_ in zip(boxes, classes) if self._class_dict[class_] == Diagram]
        diag_priors = [self.select_diag_prior(bbox) for bbox in out_diag_bboxes]
        diag_priors = [Panel(diag) for diag in diag_priors]
        diag_priors = self.filter_diag_false_positives(diag_priors, all_arrows)
        diags = DiagramExtractor(diag_priors).extract()
        conditions_labels = [box for box, class_ in zip(boxes, classes)
                             if self._class_dict[class_] in [Label, Conditions]]
        conditions, labels = self.reclassify(conditions_labels, all_arrows, diags)

        return diags, conditions, labels


    def select_diag_prior(self, bbox):
        bbox_crop = Crop(self.fig, bbox)
        prior = max(bbox_crop.connected_components, key=lambda cc: cc.area)
        return bbox_crop.in_main_fig(prior)


    def filter_diag_false_positives(self, diag_priors, all_arrows):
        """Filters diagram false positives by comparing with all extracted arrows. If a given region was marked
        as an arrow by the arrow extraction model, then it is removed from a list of potential diagrams"""
        filtered_diags = []
        for diag in diag_priors:
            if all(diag.compute_iou(arrow) < ExtractorConfig.UNIFIED_DIAG_FP_IOU_THRESH for arrow in all_arrows):
                filtered_diags.append(diag)

        return filtered_diags



    def reclassify(self, candidates, all_arrows, all_diags):
        """Attempts to reclassify label and conditions candidates based on proximity to arrows and diagrams.
        If a candidate is close to an arrow, then it is reclassified as a conditions region. Conversely, if it is close
        to a diagram, then it is classified as a label. Finally, if it is not within a given threshold distance, it is
        instantiated based on the prior label given by the unified detection model."""

        reclassified = []
        for cand in candidates:
            closest_arrow = self._find_closest(cand, all_arrows)
            closest_diag = self._find_closest(cand, all_diags)
            cand_class = self._adjust_class(cand,(closest_arrow, closest_diag))
            reclassified.append(cand_class(**cand.pass_attributes))

        return self._separate_labels_conditions(reclassified)


    def _adjust_class(self, obj, closest):
        seps = [obj.separation(cc) for cc in closest]
        if seps[0] < seps[1] and seps[0] < ExtractorConfig.UNIFIED_RECLASSIFY_DIST_THRESH_COEFF * np.sqrt(obj.area):
            class_ = Conditions
        elif seps[1] < seps[0] and seps[1] < ExtractorConfig.UNIFIED_RECLASSIFY_DIST_THRESH_COEFF * np.sqrt(obj.area):
            class_ = Label
        else:
            class_ = self._class_dict[obj._class_]

        return class_

    def _find_closest(self, panel, other_objects):
        """
        Measure the distance between 'panel' and 'other_objects' to find the object that is closest to the `panel`
        """

        dists = [(o,panel.separation(o.panel)) for o in other_objects]
        return sorted(dists, key=lambda elem: elem[1])[0][0]

    def _separate_labels_conditions(self, objects):
        conditions, labels =[], []
        for o in objects:
            if isinstance(o, Conditions):
                conditions.append(o)
            elif isinstance(o, Label):
                labels.append(o)
        return conditions, labels


class DiagramExtractor(BaseExtractor):

    def __init__(self, fig, diag_priors):
        super().__init__(fig)
        self.diag_priors = diag_priors

    def extract(self):
        diag_panels = self.complete_structures()

    def complete_structures(self):
        """
        Dilates a figure and uses backbones to find complete chemical structures (backbones + superatoms etc.).

        Arrows are first removed to increase accuracy of the process. Figure is dilates around each backbone according
        to density of features around it. The diagrams are derived from the dilated backbones. Roles are assigned
        to the disconnected diagram parts.
        :return:bounding boxes of chemical structures
        :rtype: list
        """
        fig = self.fig
        # fig_no_arrows = erase_elements(fig, self.arrows)
        dilated_structure_panels, other_ccs = self.find_dilated_structures(fig)
        structure_panels = self._complete_structures(dilated_structure_panels)
        self._assign_backbone_auxiliaries(structure_panels, other_ccs)  # Assigns cc roles
        temp = copy.deepcopy(structure_panels)
        # simple filtering to account for multiple backbone parts (disconnected by heteroatom characters)
        # corresponding to the same diagram
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
        Finds dilated structures by first dilating the image several times using backbone-specific kernel size.

        For each backbone, the figure is dilated using a backbone-specific kernel size. Dilated structure panel is then
        found based on comparison with the original backbone. A crop is made for each structure. If there is more than
        one connected component that is fully contained within the crop, it is noted and this information used later
        when the small disconnected ccs are assigned roles (This additional connected component is likely a label).

        :return: (dilated_structure_panels, other_ccs) pair of collections containing the dilated panels and
        separate ccs present within these dilated panels
        :rtype: tuple of lists
        """
        fig = self.fig
        dilated_structure_panels = []
        other_ccs = []
        dilated_imgs = {}

        for diag in self.diag_priors:
            ksize = fig.kernel_sizes[diag]
            try:
                dilated_temp = dilated_imgs[ksize] # use cached
            except KeyError:
                dilated_temp = dilate_fig(fig, ksize)
                dilated_imgs[ksize] = dilated_temp

            dilated_structure_panel = [cc for cc in dilated_temp.connected_components if cc.contains(diag)][0]
            # Crop around with a small extension to get the connected component correctly
            structure_crop = dilated_structure_panel.create_extended_crop(dilated_temp, extension=5)
            other = [structure_crop.in_main_fig(c) for c in structure_crop.connected_components if
                     structure_crop.in_main_fig(c) != dilated_structure_panel]
            other_ccs.extend(other)
            dilated_structure_panels.append(dilated_structure_panel)

        return dilated_structure_panels, other_ccs



    def _assign_backbone_auxiliaries(self, structure_panels, cno_ccs):
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
                    # if cc.role != FigureRoleEnum.STRUCTUREBACKBONE:
                    #     # Set role for all except backbone which had been set
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
        for dilated_structure in dilated_structure_panels:
            constituent_ccs = [cc for cc in self.fig.connected_components if dilated_structure.contains(cc)]
            parent_structure_panel = Panel.create_megarect(constituent_ccs)
            structure_panels.append(parent_structure_panel)
        return structure_panels


    def _find_optimal_dilation_ksize(self):
            """
            Use structural prio to calculate local skeletonised-pixel ratio and find optimal dilation kernel sizes for
            structural segmentation. Each backbone is assigned its own dilation kernel to account for varying skel-pixel
            ratio around different prio
            :return: kernel sizes appropriate for each backbone
            :rtype: dict
            """

            # prio = [cc for cc in self.fig.connected_components if cc.role == FigureRoleEnum.STRUCTUREBACKBONE]

            kernel_sizes = {}
            for prior in self.diag_priors:
                top, left, bottom, right = prior
                horz_ext, vert_ext = prior.width // 2, prior.height // 2
                crop_rect = Rect((top - vert_ext,left - horz_ext, bottom + vert_ext, right + horz_ext ))
                p_ratio = skeletonize_area_ratio(self.fig, crop_rect)
                # log.debug(f'found in-crop skel_pixel ratio: {p_ratio}')

                if p_ratio >= 0.02:
                    kernel_size = 4
                elif 0.01 < p_ratio < 0.02:
                    kernel_size = np.ceil(20 - 800 * p_ratio)
                else:
                    kernel_size = 12
                kernel_sizes[prior] = kernel_size

class TextRegionCandidate:
    """This class is used to represent regions which consist of text and contain either labels or reaction conditions.
    This is determined at a later stage and attributes from each object are transferred to the new instances. In the
    pipeline, outputs marked by the unified model as conditions or labels are instantiated as objects of this class"""

    def __init(self, bbox, class_):
        self.panel = Panel(bbox)
        self._class_ = class_



