"""This module contains all parts of the unified extractor for diagrams, labels and reaction conditions"""
import copy

import numpy as np
import torch
from detectron2.config import get_cfg
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from matplotlib.patches import Rectangle
from scipy.stats import mode
from detectron2 import model_zoo


from .base import BaseExtractor, Candidate
from .conditions import ConditionsExtractor
# from .labels import LabelExtractor
from .labels import LabelExtractor
from ..config import ExtractorConfig
# from ..models.ml_models.unified_detection import RDEModel, RSchemeConfig
from ..models.reaction import Diagram, Conditions, Label
from ..models.segments import Panel, Rect, FigureRoleEnum, Crop, PanelMethodsMixin
from ..utils import skeletonize_area_ratio, dilate_fig

###detectron2 config ###
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.DEVICE = 'cpu'
cfg.DATASETS.TRAIN = ("artificial_data_train",)
cfg.DATASETS.TEST = ("artificial_data_test",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS =ExtractorConfig.UNIFIED_EXTR_MODEL_WT_PATH
    #model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[8,16 ], [16,32 ], [32,64 ], [64,128 ], [256,512 ]]
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
###

class UnifiedExtractor(BaseExtractor):

    def __init__(self, fig, all_arrows):
        super().__init__(fig)
        # self.model = self.load_model(ExtractorConfig.UNIFIED_EXTR_MODEL_WT_PATH)
        self.model = DefaultPredictor(cfg)

        # DetectionCheckpointer(self.model).load(ExtractorConfig.UNIFIED_EXTR_MODEL_WT_PATH)
        # self.model.eval()
        self.all_arrows = all_arrows
        self.diagram_extractor = DiagramExtractor(self.fig,diag_priors=None)
        self.label_extractor = LabelExtractor(self.fig, priors=None)
        self.conditions_extractor = ConditionsExtractor(self.fig, priors=None)


        # self._class_dict = {
        #     1: Diagram,
        #     2: Conditions,
        #     3: Label
        #
        # }
        ##torch
        self._class_dict = {
            0: Diagram,
            1: Conditions,
            2: Label

        }

    # def load_model(self, weights_path):
    #     model_config = RSchemeConfig()
    #     model = RDEModel(training=False, config=model_config)
    #     primer_input = np.zeros(model_config.IMAGE_SHAPE)[np.newaxis, ...]
    #     model([primer_input, np.zeros((1, 15))])
    #     model.load_weights(weights_path)
    #     return model

    def extract(self):
        boxes, classes, scores = self.detect_detectron2()
        boxes = self.adjust_coord_order_detectron(boxes)
        out_diag_bboxes = [box for box, class_ in zip(boxes, classes) if self._class_dict[class_] == Diagram]
        diag_priors = [self.select_diag_prior(bbox) for bbox in out_diag_bboxes]
        # diag_priors = [Panel(diag) for diag in diag_priors]
        diag_priors = self.filter_diag_false_positives(diag_priors, self.all_arrows)
        self.diagram_extractor.diag_priors = diag_priors
        diags = self.diagram_extractor.extract()
        conditions_labels = [TextRegionCandidate(box, class_) for box, class_ in zip(boxes, classes)
                             if self._class_dict[class_] in [Label, Conditions]]
        adjusted_candidates = self.adjust_bboxes(conditions_labels)
        conditions, labels = self.reclassify(adjusted_candidates, self.all_arrows, diags)
        conditions, labels = [self.remove_duplicates(group) for group in [conditions, labels]]
        conditions, labels = [self.extract_elements(group, extractor) for group, extractor
                              in zip([conditions, labels], [self.conditions_extractor, self.label_extractor])]
        return diags, conditions, labels

    def adjust_coord_order_detectron(self, boxes):
        """Adjusts order of coordinates to the expected format

        Detectron outputs boxes with coordinates (x1, y1, x2, y2), however the expected
        format is (top, left, bottom, right) i.e. (y1, x1, y2, x2). This function switches
        the coordinates to achieve consistency"""
        boxes = boxes[:, [1, 0, 3, 2]]
        return boxes


    def plot_extracted(self, ax):
        self.diagram_extractor.plot_extracted(ax)
        self.label_extractor.plot_extracted(ax)
        self.conditions_extractor.plot_extracted(ax)

    def detect(self):
        """A method used to perform any necessary preprocessing on an input before feeding it into the object detection model
        and making predictions."""
        if mode(self.fig.img.reshape(-1))[0][0] == 0:
            img = np.invert(self.fig.img)
        else:
            img = self.fig.img
        img = (img - img.min()) / (img.max() - img.min())
        return map(lambda x: x[0].numpy(), self.model.predict(img)) # Predictions for first (and only) image in a batch

    def detect_detectron2(self):
        img = self.fig.img
        with torch.no_grad():
            predictions = self.model(img[..., np.newaxis])
        #visualize predictions
        # from detectron2.utils.visualizer import Visualizer
        # import matplotlib.pyplot as plt
        # vis = Visualizer(img)
        # vis.draw_instance_predictions(predictions['instances'])
        # plt.imshow(vis.output.get_image())
        # plt.show()

        predictions = predictions['instances']
        return predictions.pred_boxes.tensor.numpy().astype(np.int32), predictions.pred_classes.numpy(), predictions.scores.numpy()


    def remove_duplicates(self, panels):
        """Removes duplicate panels inside `group`. In this context, duplicates are all panels which cover the same
        region, whether an entire region or just a part of it. In such cases, the biggest panel is selected.

        The panels are first grouped into sets of overlapping objects and then removed on a group-by-group basis"""
        dists = np.array([[p1.edge_separation(p2) for p1 in panels] for p2 in panels])
        not_yet_grouped = set(list(range(len(panels))))
        groups = []

        idx1 = 0
        while idx1 < len(panels) - 1:
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
        extractor.priors = elements
        return extractor.extract()

    def select_diag_prior(self, bbox):
        bbox_crop = Crop(self.fig, bbox)
        try:
            prior = max(bbox_crop.connected_components, key=lambda cc: cc.area)
        except ValueError: # no connected components in the crop
            return None
        return bbox_crop.in_main_fig(prior)


    def filter_diag_false_positives(self, diag_priors, all_arrows):
        """Filters diagram false positives by comparing with all extracted arrows. If a given region was marked
        as an arrow by the arrow extraction model, then it is removed from a list of potential diagrams"""
        filtered_diags = []
        for diag in diag_priors:
            is_diag = diag is not None
            if is_diag and all(diag.compute_iou(arrow) < ExtractorConfig.UNIFIED_DIAG_FP_IOU_THRESH for arrow in all_arrows):
                filtered_diags.append(diag)

        return filtered_diags



    def adjust_bboxes(self, region_candidates):
        """Adjusts bboxes inside each of region_candidates by restricting them to cover connected components fully contained within crops bounded
        by the bboxes"""

        adjusted = []
        for cand in region_candidates:
            crop = cand.panel.create_extended_crop(self.fig, extension=20)
            relevant_ccs = [crop.in_main_fig(cc) for cc in crop.connected_components]
            relevant_ccs = [cc for cc in relevant_ccs if cc.role is None]  # Previously unassigned ccs
            if relevant_ccs:
                cand.panel = Panel.create_megapanel(relevant_ccs, self.fig)
                adjusted.append(cand)
        return adjusted


    def reclassify(self, candidates, all_arrows, all_diags):
        """Attempts to reclassify label and conditions candidates based on proximity to arrows and diagrams.
        If a candidate is close to an arrow, then it is reclassified as a conditions region. Conversely, if it is close
        to a diagram, then it is classified as a label. Finally, if it is not within a given threshold distance, it is
        instantiated based on the prior label given by the unified detection model. Assigns the closest of these panels
        as a parent panel"""

        conditions = []
        labels = []
        for cand in candidates:
            closest_arrow = self._find_closest(cand, all_arrows)
            closest_diag = self._find_closest(cand, all_diags)
            cand_class = self._adjust_class(cand, (closest_arrow, closest_diag))
            cand.parent_panel = cand.set_parent_panel([closest_diag, closest_arrow])
            ## Try OCR on all of them, then postprocess accordingly and instantiate?
            # reclassified.append(cand_class(**cand.pass_attributes()))
            if cand_class == Conditions:
                conditions.append(cand)
            else:
                labels.append(cand)





        return conditions, labels


    def _adjust_class(self, obj, closest):
        seps = [obj.edge_separation(cc) for cc in closest]
        if seps[0] < seps[1] and seps[0] < ExtractorConfig.UNIFIED_RECLASSIFY_DIST_THRESH_COEFF * np.sqrt(obj.area):
            class_ = Conditions

        elif seps[1] < seps[0] and seps[1] < ExtractorConfig.UNIFIED_RECLASSIFY_DIST_THRESH_COEFF * np.sqrt(obj.area):
            class_ = Label

        else:
            class_ = self._class_dict[obj._prior_class]


        return class_

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


class DiagramExtractor(BaseExtractor):

    def __init__(self, fig, diag_priors):
        super().__init__(fig)
        self.diag_priors = diag_priors
        self.diags = None

    def extract(self):
        assert self.diag_priors is not None, "Diag priors have not been set"
        self.fig.dilation_iterations = self._find_optimal_dilation_extent()
        self.fig.set_roles(self.diag_priors, FigureRoleEnum.DIAGRAMPRIOR)
        TODO Why are not diag_panels found? Is it to do with num_iterations / dilation etc?
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
        Dilates a figure and uses backbones to find complete chemical structures (backbones + superatoms etc.).

        Arrows are first removed to increase accuracy of the process. Figure is dilates around each backbone according
        to density of features around it. The diagrams are derived from the dilated backbones. Roles are assigned
        to the disconnected diagram parts.
        :return:bounding boxes of chemical structures
        :rtype: list
        """
        # fig = self.fig
        # fig_no_arrows = erase_elements(fig, self.arrows)
        dilated_structure_panels, other_ccs = self.find_dilated_structures()
        structure_panels = self._complete_structures(dilated_structure_panels)
        self._assign_diagram_parts(structure_panels, other_ccs)  # Assigns cc roles
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
            num_iterations = fig.dilation_iterations[diag]
            try:
                dilated_temp = dilated_imgs[num_iterations] # use cached
            except KeyError:
                dilated_temp = dilate_fig(fig, num_iterations)
                dilated_imgs[num_iterations] = dilated_temp

            dilated_structure_panel = [cc for cc in dilated_temp.connected_components if cc.contains(diag)][0]
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
                horz_ext = max(horz_ext, ExtractorConfig.DIAG_MIN_EXT)
                vert_ext = max(vert_ext, ExtractorConfig.DIAG_MIN_EXT)

                crop_rect = Rect((top - vert_ext,left - horz_ext, bottom + vert_ext, right + horz_ext ))
                p_ratio = skeletonize_area_ratio(self.fig, crop_rect)
                # print(p_ratio)
                # log.debug(f'found in-crop skel_pixel ratio: {p_ratio}')

                if p_ratio >= 0.03:
                    n_iterations = 4
                elif 0.01 < p_ratio < 0.02:
                    n_iterations = np.ceil(18 - 600 * p_ratio)
                else:
                    n_iterations = 12
                num_iterations[prior] = n_iterations

            return num_iterations

class TextRegionCandidate(Candidate, PanelMethodsMixin):
    """This class is used to represent regions which consist of text and contain either labels or reaction conditions.
    This is determined at a later stage and attributes from each object are transferred to the new instances. In the
    pipeline, outputs marked by the unified model as conditions or labels are instantiated as objects of this class"""

    def __init__(self, bbox, class_):
        self.panel = Panel(bbox)
        self._prior_class = class_
        self.parent_panel = None

    def set_parent_panel(self, nearby_panels):
        return min(nearby_panels, key=lambda p: p.center_separation(self))



