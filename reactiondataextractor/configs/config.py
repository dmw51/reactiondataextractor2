import os
import sys
import cv2

THIS_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

class Config:
    FIGURE = None
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_PATH = None
    HOME = os.path.expanduser('~')
    TESSDATA_PATH = f'{THIS_DIR}/../../tessdata'
    SINGLE_BOND_LENGTH = 1.54


class ExtractorConfig(Config):
    """Config used for all extractor classes"""
    # Whether to use cpu or gpu to run deep models
    DEVICE = 'cpu'  ## 'cpu' or 'cuda'
    # Path to the arrow detector weights
    ARROW_DETECTOR_PATH = os.path.join(Config.ROOT_DIR, '../models/cnn_weights/torch_arrow_detector_classifier.pt')
    # Path to the arrow classifier weights
    ARROW_CLASSIFIER_PATH = os.path.join(Config.ROOT_DIR, '../models/cnn_weights/torch_arrow_classifier.pt')
    # Shape of an arrow image fed to the detector model
    ARROW_IMG_SHAPE = [64, 64]
    # cv2 params for arrow initialization
    ARROW_CNT_MODE = cv2.RETR_EXTERNAL
    ARROW_CNT_METHOD = cv2.CHAIN_APPROX_SIMPLE

    # Path to the main object detection model
    UNIFIED_EXTR_MODEL_WT_PATH = os.path.join(Config.ROOT_DIR,
                                              '../models/cnn_weights/model_best_15Mar_diou.pth')
    # Threshold for suppressing detected diagrams overlapping with arrows
    UNIFIED_DIAG_FP_IOU_THRESH = 0.75
    # Threshold for reclassifying textual elements based on proximity to arrows and diagrams
    UNIFIED_RECLASSIFY_DIST_THRESH_COEFF = 2
    # Threshold for object detection
    UNIFIED_PRED_THRESH = 0.25
    # Threshold for filtering false positives based on intersection over area
    UNIFIED_IOA_FILTER_THRESH = 0.9
    # Threshold for selecting small detections from image tiles (only small boxes are chosen)
    TILER_THRESH_AREA_PERCENTILE = 20
    # Maximum dimensions of tile patches
    TILER_MAX_TILE_DIMS = (600, 800)
    # Distance for classifying diagrams as part of a conditions region around the arrow
    ARROW_DIAG_MAX_DISTANCE = 80
    # Distance used to crop an image around diagrams to compute appropriate dilation parameters
    DIAG_DILATION_EXT = 50
    # Maximum diagram area relative to total image area
    DIAG_MAX_AREA_FRACTION = 0.45
    # Path to the dictionary containing common chemical species' names
    CONDITIONS_SPECIES_PATH = os.path.join(Config.ROOT_DIR, '../dict/species.txt')
    # Threshold factor to filter out very big conditions regions (poor detections)
    CONDITIONS_MAX_AREA_FRACTION = 1.5
    # Maximum distance between conditions and the nearest arrow (assumes smaller image dimension is 1024 px)
    CONDITIONS_ARROW_MAX_DIST = 75
    # Maximum allowed distance difference between a labels and the first and second-closest diagram for pair reassignment
    DIAG_LABEL_MAX_REASSIGNMENT_DISTANCE = 75


class ProcessorConfig(Config):
    # Image binarisation thresholds
    BIN_THRESH = [70, 255]
    CANNY_THRESH = [50, 100]


class OCRConfig(Config):
    # Minimum area to perform character-wise OCR when poor outcome obtained
    PIECEWISE_OCR_THRESH_AREA = 100 #TODO: This value should be appropriate for commas, dots etc - optimise


class SchemeConfig(Config):
    # Overlap needed between a probing line and the diagram bbox to classify a diagram as part of a reaction step
    MIN_PROBING_OVERLAP_FACTOR = 0.35
    MAX_GROUP_DISTANCE = 50 # TODO: Adjust this value (maybe make a coefficient out of this)


class GlobalRGroupCache:
    def __init__(self):
        self.r_groups = set()
        self.r_group_variants = {}

    def update_variants(self, variant_dict):
        for k, v in variant_dict.items():
            if self.r_group_variants.get(k):
                self.r_group_variants[k].extend(v)
            else:
                self.r_group_variants[k] = v
                
class GlobalTextCache:
    def __init__(self):
        self.diag_chars = []
        
    def extend(self, iterable):
        self.diag_chars.extend(iterable)
    
    def append(self, v):
        self.diag_chars.append(v)
    
    def __iter__(self):
        return iter(self.diag_chars)

global_r_group_cache = GlobalRGroupCache()
global_text_cache = GlobalTextCache()
