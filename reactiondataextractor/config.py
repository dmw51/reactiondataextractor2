import os

import cv2



class Config:
    FIGURE = None
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_PATH = None


class ExtractorConfig(Config):
    # ARROW_DETECTOR_PATH = os.path.join(Config.ROOT_DIR, 'models/ml_models/arrow_detector.hdf5')
    # ARROW_CLASSIFIER_PATH = os.path.join(Config.ROOT_DIR, 'models/ml_models/arrow_classifier.h5')
    ARROW_DETECTOR_PATH = os.path.join(Config.ROOT_DIR, 'models/ml_models/torch_arrow_detector.pt')
    ARROW_CLASSIFIER_PATH = os.path.join(Config.ROOT_DIR, 'models/ml_models/torch_arrow_classifier.pt')
    ARROW_IMG_SHAPE = [64, 64]
    SOLID_ARROW_THRESHOLD = None  # Set dynamically based on the length of a single-bond line
    SOLID_ARROW_MIN_LENGTH = None  # Set dynamically based on the length of a single-bond line
    CURLY_ARROW_CNT_MODE = cv2.RETR_EXTERNAL
    CURLY_ARROW_CNT_METHOD = cv2.CHAIN_APPROX_SIMPLE
    CURLY_ARROW_MIN_AREA_FRACTION = 0.005
    CURLY_ARROW_CNT_AREA_TO_BBOX_AREA_RATIO = 0.3
    ARROW_DETECTOR_THRESH = 0.99

    # UNIFIED_EXTR_MODEL_WT_PATH = os.path.join(Config.ROOT_DIR, 'models/ml_models/unified_detection/weights.h5')
    UNIFIED_EXTR_MODEL_WT_PATH = os.path.join(Config.ROOT_DIR, 'models/ml_models/unified_detection/model_best_15Mar_diou.pth')
    # UNIFIED_EXTR_MODEL_WT_PATH = os.path.join(Config.ROOT_DIR,
    #                                           'models/ml_models/unified_detection/model_final 2000_bg_0.pth')
    UNIFIED_DIAG_FP_IOU_THRESH = 0.75
    UNIFIED_RECLASSIFY_DIST_THRESH_COEFF = 2
    UNIFIED_PRED_THRESH = 0.5
    UNIFIED_IOA_FILTER_THRESH = 0.9

    TILER_THRESH_AREA_PERCENTILE = 20
    TILER_MAX_TILE_DIMS = (600, 800) # Desired height of tile (excluding extension overlap betweem tiles)

    ARROW_DIAG_MAX_DISTANCE = 50 # Distance for classifying diagrams as part of a conditions region around the arrow
    DIAG_MIN_EXT = 50
    DIAG_MAX_AREA_FRACTION = 0.45 # Maximum diagram area relative to total image area
    CONDITIONS_SPECIES_PATH = os.path.join(Config.ROOT_DIR, 'dict/species.txt' )


class ProcessorConfig(Config):
    BIN_THRESH = [40, 255]
    CANNY_THRESH = [50, 100]

class OCRConfig(Config):
    PIECEWISE_OCR_THRESH_AREA = 100 #TODO: This value should be appropriate for commas, dots etc - optimise

class SchemeConfig(Config):
    SEARCH_DISTANCE_FACTOR = 0.9
    MIN_PROBING_OVERLAP_FACTOR = 0.5

class SegmentsConfig(Config):
    CROP_THRESH_INTER_AREA = 0.95  # What fraction of a connected component needs to be within a crop to belong to it
