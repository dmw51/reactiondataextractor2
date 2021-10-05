class Config:
    pass


class ExtractorConfig(Config):
    SOLID_ARROW_THRESHOLD = None  # Set dynamically based on the length of a single-bond line
    SOLID_ARROW_MIN_LENGTH = None  # Set dynamically based on the length of a single-bond line
    CURLY_ARROW_MIN_AREA_FRACTION = 0.008
    CURLY_ARROW_CNT_AREA_TO_BBOX_AREA_RATIO = 0.3

class ProcessorConfig(Config):
    BIN_THRESH = [40, 255]
    CANNY_THRESH = [50, 100]