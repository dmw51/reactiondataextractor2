import os
from enum import Enum, auto
from abc import ABC, abstractmethod
import imageio as imageio
import numpy as np

import cv2
from scipy.stats import mode

from .models.segments import Rect, Figure
from . import config

class GlobalFigureMixin:
    """If no `figure` was passed to an initializer, use the figure stored in config
    (set at the beginning of extraction)"""
    def __init__(self, fig):
        if fig is None:
            self.fig = config.Config.FIGURE

class Processor(ABC, GlobalFigureMixin):

    class COLOR_MODE(Enum):
        GRAY = auto()
        RGB = auto()

    def __init__(self, enabled=True, fig=None):
        self._enabled = enabled
        self.fig = fig
        super().__init__(self.fig)
        self._img = self.fig.img if self.fig else None

    @abstractmethod
    def process(self):
        pass

    @property
    def img(self):
        return self._img


class ImageReader(Processor):


    def __init__(self, filepath, color_mode):
        assert color_mode in self.COLOR_MODE, "Color_mode must be one of ImageColor.COLORMODE enum members"
        assert os.path.exists(filepath), "Could not open file - Invalid path was entered"
        self.filepath = filepath
        self.color_mode = color_mode
        _, self.ext = os.path.splitext(filepath)
        super().__init__(enabled=True)

    def process(self):

        if self.color_mode == self.COLOR_MODE.GRAY:
            img = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)

        elif self.color_mode == self.COLOR_MODE.RGB:
            img = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None and self.ext == '.gif':   # Ensure this special case is treated
            img = imageio.mimread(self.filepath)
            img = img[0]
            assert len(img.shape) == 2  #
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_img = img
        bg_value = mode(img.ravel())[0][0]
        if bg_value in range(250, 256):
            img = np.invert(img)
        self.fig = Figure(img=img, raw_img=raw_img)
        return self.fig


class TextLineRemover(Processor):
    WIDTH_THRESH_FACTOR = 0.3

    def __init__(self, img, enabled=True):
        self.selem = np.concatenate((np.zeros((2, 6)), np.ones((2, 6)), np.zeros((2, 6))), axis=0).astype(np.uint8)

        self.top_roi = Rect(0, 0, self.img.shape[0] // 5, self.img.shape[1] // 5)
        self.bottom_roi = Rect(int(self.img.shape[0] * 4 / 5), 0,
                               self.img.shape[0], int(self.img.shape[1] // 5))

        super().__init__(enabled=enabled)

    def process(self):
        if not self._enabled:
            return self.img
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img = 255 - img
        img = cv2.dilate(img, self.selem, iterations=6)
        ret, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        crop_top = []
        crop_bottom = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            print(cv2.boundingRect(cnt))
            #             print(self.bottom_roi)
            if w > self.WIDTH_THRESH_FACTOR * self.img.shape[1]:
                print('width okay')

                #                 print(f'{x}, {y}')
                if self.top_roi.contains_point((x, y)):
                    crop_top.append((x, y, x + w, y + h))
                elif self.bottom_roi.contains_point((x, y)):
                    crop_bottom.append((x, y, x + w, y + h))

        # Crop the whole images down to the first bottom text line
        print(f'crop_bottom: {crop_bottom}')
        if crop_bottom:
            crop_bottom_boundary = min([coords[1] for coords in crop_bottom])
            print(crop_bottom_boundary)
            self._img = self._img[:crop_bottom_boundary, :]
        if crop_top:
            crop_top_boundary = max([coords[1] for coords in crop_top])
            self._img = self._img[crop_top_boundary:, :]
        return self._img


class EdgeExtractor(Processor):

    def __init__(self, fig, bin_thresh=None):
        super().__init__(enabled=True, fig=fig)
        if len(self.img.shape) == 2:
            self.color_mode = self.COLOR_MODE.GRAY
        elif len(self.img.shape) == 3 and self.img.dtype == np.uint8:
            self.color_mode = self.COLOR_MODE.RGB
        self.bin_thresh = bin_thresh if bin_thresh else config.ProcessorConfig.BIN_THRESH



    def process(self):
        if self.color_mode == self.COLOR_MODE.GRAY:

            #TODO: Make sure the background is consistent - this makes contour search reliable
            ## bg should be 0

            ret, thresh = cv2.threshold(self.img, *self.bin_thresh, cv2.THRESH_BINARY)
            return Figure(thresh, raw_img=self.img)
        elif self.color_mode == self.COLOR_MODE.RGB:

            img = cv2.GaussianBlur(self.img, (3, 3), 1)
            return cv2.Canny(img, *config.ProcessorConfig.CANNY_THRESH)


class Isolator(Processor):

    def __init__(self, fig, to_isolate, isolate_mask):
        super().__init__(fig=fig)
        self.to_isolate = to_isolate
        self.isolate_mask = isolate_mask

    def _isolate_panel(self):
        #TODO
        pass

    def _isolate_mask(self):
        rows, cols = zip(*self.to_isolate.pixels)
        mask = np.zeros_like(self.fig.img, dtype=np.bool)
        isolated = np.zeros_like(self.fig.img, dtype=np.uint8)
        mask[rows, cols] = True
        isolated[mask] = self.fig.img[mask]
        return Figure(img=isolated, raw_img=self.fig.img)

    def process(self):
        if self.isolate_mask:
            return self._isolate_mask()
        else:
            return self._isolate_panel()




