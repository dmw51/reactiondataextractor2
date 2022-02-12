import numpy as np
from .utils import Dataset
from .config import Config
import json

import skimage
from skimage.io import imread


def clean_empty_entries(annot_file):
    assert isinstance(annot_file, dict)
    new_annot = {}
    for key in annot_file.keys():
        if annot_file[key]['regions']:
            new_annot[key] = annot_file[key]

    return new_annot


class RSchemeDataset(Dataset):
    def __init__(self):
        super().__init__()

    def load_dataset(self, dir_paths, annot_paths):
        omitted = 0

        def get_info_from_cleaned_meta(annot, cls_dict):
            nonlocal dir_path
            for img in annot:
                a = {'filename': img['filename']}
                try:
                    # print(dir_path + a['filename'])
                    i = imread(dir_path + a['filename'])
                except FileNotFoundError:
                    try:
                        filename = a['filename'].split('.')[0] + '.gif'
                        i = imread(dir_path + filename)
                    except FileNotFoundError:
                        continue

                img_height = img['img_height']
                img_width = img['img_width']
                polygons = []
                classes = []

                for region in img['regions']:
                    panel_type = region['region_attributes']['panel_type']
                    if cls_dict.get(panel_type):  # proceed only if there is
                        # appropriate key in the dict, else skip this region
                        polygon = convert_annot_rect_to_polygon(region)
                        cls = cls_dict[panel_type]
                        polygons.append(polygon)
                        classes.append(cls)
                # print(a, polygons, classes, img_height, img_width)
                yield a, polygons, classes, img_height, img_width

        def get_info_from_via_json(annot, cls_dict):
            annot = clean_empty_entries(annot)
            for key in annot.keys():
                polygons = []
                classes = []
                a = annot[key]
                try:
                    i = imread(dir_path + a['filename'])
                except FileNotFoundError:
                    try:
                        print(a['filename'])
                        filename = a['filename'].split('.')[0] + '.gif'
                        i = imread(dir_path + filename)
                        a['filename'] = filename
                    except:
                        continue
                img_height = i.shape[0]
                img_width = i.shape[1]
                for region in a['regions']:
                    try:
                        panel_type = region['region_attributes']['panel_type']
                        # print(region['region_attributes'])
                    except KeyError:
                        print('A region does not have a panel_type attribute')
                        continue
                    if cls_dict.get(
                            panel_type):  # proceed only if there is appropriate key in the dict, else skip this regio
                        polygon = convert_annot_rect_to_polygon(region)
                        cls = cls_dict[panel_type]
                        # print(cls)
                        polygons.append(polygon)
                        classes.append(cls)
                # print(a, polygons, classes, img_height, img_width)
                yield a, polygons, classes, img_height, img_width

        self.add_class("dataset", 5, 'Solid Arrow')
        self.add_class("dataset", 2, 'Equilibrium Arrow')
        self.add_class("dataset", 3, 'ResonanceArrow')
        self.add_class("dataset", 4, 'Curly Arrow')
        self.add_class("dataset", 1, 'Diagram')
        self.add_class("dataset", 6, 'Conditions')
        self.add_class("dataset", 7, 'Label')

        cls_dict = {
            'diagram': 1,
            'compound_diagram': 1,
            'arrow': 5,
            'arrow_solid': 5,
            'arrow_equilibrium': 2,
            'arrow_resonance': 3,
            'arrow_curly': 4,
            'conditions': 6,
            'label': 7, }
        for dir_path, annot_path in zip(dir_paths, annot_paths):
            print(f'Loading image data from {annot_path}.')
            print(f'The images of interest are stored in {dir_path}.')
            with open(annot_path) as file:

                annot = json.load(open(annot_path))
                if isinstance(annot, list):
                    for a, polygons, classes, img_height, img_width in get_info_from_cleaned_meta(annot, cls_dict):
                        # print('adding...')
                        self.add_image("dataset", a['filename'], path=dir_path + '/' + a['filename'], polygons=polygons,
                                       classes=classes,
                                       img_height=img_height, img_width=img_width)
                elif isinstance(annot, dict):
                    for a, polygons, classes, img_height, img_width in get_info_from_via_json(annot, cls_dict):
                        self.add_image("dataset", a['filename'], path=dir_path + '/' + a['filename'], polygons=polygons,
                                       classes=classes,
                                       img_height=img_height, img_width=img_width)

    def load_image(self, image_id):
        # from skimage.io import imread
        from skimage.color import rgb2gray
        info = self.image_info[image_id]
        img = imread(info['path'])
        if img.shape[-1] == 4:
            raise NotImplemented
        if len(img.shape) == 3:
            img = rgb2gray(img)[:, :, np.newaxis]

        return img

    def load_bboxes(self, image_id):
        info = self.image_info[image_id]
        class_ids = np.array(info['classes'])
        bboxes = np.zeros((class_ids.shape[0], 4))
        for i, p in enumerate(info["polygons"]):
            x_min = np.min(p['all_points_x'])
            y_min = np.min(p['all_points_y'])
            x_max = np.max(p['all_points_x']) + 1
            y_max = np.max(p['all_points_y']) + 1
            bboxes[i, :] = (y_min, x_min, y_max, x_max)
        return bboxes, class_ids

    # def load_mask(self, image_id):
    #     info = self.image_info[image_id]
    #     mask = np.zeros((info['img_height'], info['img_width'], len(info['polygons'])))
    #     image_mask = np.zeros((info['img_height'], info['img_width']))
    #     class_ids = np.zeros([mask.shape[-1]], dtype=np.int32)
    #     img = self.load_image(image_id)
    #     img = np.mean(img, axis=2)
    #     if np.max(img) == 255:
    #         bin_img = img / 255 < 0.99
    #     else:
    #         bin_img = img < 0.99
    #     non_zero_y, non_zero_x = np.where(bin_img == True)
    #     image_mask[non_zero_y, non_zero_x] = 1
    #     for i, p in enumerate(info["polygons"]):
    #         rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
    #         class_ids[i] = info['classes'][i]
    #
    #         class_ids[i] = info['classes'][i]
    #         mask[rr, cc, i] = 1
    #
    #     return mask.astype(np.bool), class_ids


def crop_patch(img, polygon):
    x = polygon['all_points_x']
    y = polygon['all_points_y']
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    return (min_y, max_y), (min_x, max_x), img[min_y:max_y, min_x:max_x]


def polygonal_and(rr, cc, image):
    new_rr = []
    new_cc = []
    for i in range(len(rr)):
        if image[rr[i], cc[i]] == 1:
            new_rr.append(rr[i])
            new_cc.append(cc[i])
    return new_rr, new_cc


def convert_annot_rect_to_polygon(region):
    r = region

    r = r['shape_attributes']
    if r['name'] == 'rect':
        # print(r)
        x = r['x']
        y = r['y']
        w = r['width']
        h = r['height']
        p = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        x, y = zip(*p)
        p = {'all_points_x': x, 'all_points_y': y}
    elif r['name'] == 'polygon':
        p = {'all_points_x': r['all_points_x'], 'all_points_y': r['all_points_y']}

    return p


class RSchemeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "RDE_schemeparts"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    TRAIN_ROIS_PER_IMAGE = 200
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + diagrams, conditions, labels

    VALIDATION_STEPS = 20
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50  # len(train_dataset.image_info)//2+1

    # Sizes of anchor boxes ('priors') regressed to the bounding boxes
    RPN_ANCHOR_SCALES = [8, 16, 32, 64, 128]

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    #Overlap threshold set to a higher value to generate more proposals
    RPN_NMS_THRESHOLD = 0.85

    # Architecture altered to exclude FPN - only a single backbone stride for the last Resnet outputs is needed
    BACKBONE_STRIDE = 16

    # Skip detections with < 85% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    DETECTION_MAX_INSTANCES = 50

    IMAGE_SHAPE = (1024, 1024, 1)

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "triplet_loss": 0.5,
        "cnn_class_loss": 1.,
        "cnn_bbox_loss": 1.
    }

    #Shape of the output feature map
    FEATURE_MAP_SHAPE = [64, 64]

    #Size of the latent representation for each vector used in triplet loss
    TRIPLETS_NUM_FILTERS = 128

    #Size of the dense layers inside the classifier head
    FPN_CLASSIF_FC_LAYERS_SIZE = 256