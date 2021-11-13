import os

import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    FIGURE = None


class ExtractorConfig(Config):
    ARROW_DETECTOR_PATH = os.path.join(ROOT_DIR, 'models/ml_models/arrow_detector.hdf5')
    ARROW_CLASSIFIER_PATH = os.path.join(ROOT_DIR, 'models/ml_models/arrow_classifier.h5')
    ARROW_IMG_SHAPE = [64, 64]
    SOLID_ARROW_THRESHOLD = None  # Set dynamically based on the length of a single-bond line
    SOLID_ARROW_MIN_LENGTH = None  # Set dynamically based on the length of a single-bond line
    CURLY_ARROW_CNT_MODE = cv2.RETR_EXTERNAL
    CURLY_ARROW_CNT_METHOD = cv2.CHAIN_APPROX_SIMPLE
    CURLY_ARROW_MIN_AREA_FRACTION = 0.001
    CURLY_ARROW_CNT_AREA_TO_BBOX_AREA_RATIO = 0.3
    ARROW_DETECTOR_THRESH = 0.43

    UNIFIED_EXTR_MODEL_WT_PATH = os.path.join(ROOT_DIR, 'models/ml_models/unified_detection/weights.h5')
    UNIFIED_DIAG_FP_IOU_THRESH = 0.75
    UNIFIED_RECLASSIFY_DIST_THRESH_COEFF = 2

    CONDITIONS_SPECIES_PATH = os.path.join(ROOT_DIR, 'dict/species.txt' )


class ProcessorConfig(Config):
    BIN_THRESH = [40, 255]
    CANNY_THRESH = [50, 100]

class OCRConfig(Config):
    PIECEWISE_OCR_THRESH_AREA = 100 #TODO: This value should be appropriate for commas, dots etc - optimise