import os
from copy import deepcopy
from enum import Enum, auto
from abc import ABC, abstractmethod
import imageio as imageio
import numpy as np

import cv2
from scipy.stats import mode

from configs.figure import GlobalFigureMixin
from reactiondataextractor.models.segments import Figure
from reactiondataextractor.configs import config


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
        """The main processing method"""
        pass

    @property
    def img(self):
        return self._img


class ImageReader(Processor):

    def __init__(self, filepath, color_mode):
        config.Config.IMG_PATH = filepath
        assert color_mode in self.COLOR_MODE, "Color_mode must be one of ImageColor.COLORMODE enum members"
        assert os.path.exists(filepath), "Could not open file - Invalid path was entered"
        self.filepath = filepath
        self.color_mode = color_mode
        _, self.ext = os.path.splitext(filepath)
        super().__init__(enabled=True)

    def process(self):
        """Reads an image into an np.ndarray from .png, .jpg/.jpeg etc formats, as well as .gif format (used by some
        journals)"""
        if self.color_mode == self.COLOR_MODE.GRAY:
            img = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)
            img_detectron = cv2.imread(self.filepath)
        elif self.color_mode == self.COLOR_MODE.RGB:
            img = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_detectron = cv2.imread(self.filepath)

        if img is None and self.ext == '.gif':   # Ensure this special case is treated
            img = imageio.mimread(self.filepath)
            img = img[0]
            # img_detectron = img
            # img_detectron = cv2.cvtColor(img_detectron, cv2.COLOR_GRAY2BGR)
            img, img_detectron = self._convert_gif(img)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.adjust_bg_value(img)
        img_detectron = self.adjust_bg_value(img_detectron, desired=255)
        self.fig = Figure(img=img, raw_img=img, img_detectron=img_detectron)  # Important that all used imgs have bg value 0 hence raw_img==img
        return self.fig

    def _convert_gif(self, img):
        img_detectron = img
        if len(img_detectron.shape) == 2:
            img_detectron = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR )
        elif len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
            img_detectron = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if self.color_mode == self.COLOR_MODE.GRAY:
            if len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY )
            elif len(img.shape) == 3 and img.shape[-1] == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif self.color_mode == self.COLOR_MODE.RGB:
            if len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB )
            elif len(img.shape) == 2:  #Gray
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img, img_detectron

    def adjust_bg_value(self, img, desired=0):
        """Flips the image if the background colour does not match the desired background colour
        :param img: image to be processed
        :type img: np.ndarray
        :param desired: the expected value associated with background (0 or 255)
        :type desired: int"""
        bg_value = mode(img.ravel())[0][0]

        if desired == 0:
            if bg_value in range(250, 256):
                img = np.invert(img)
        elif desired == 255:
            if bg_value in range(0, 10):
                img = np.invert(img)
        return img


class ImageScaler(Processor):
    """Processor used for scaling an image"""

    def __init__(self, fig, resize_min_dim_to, enabled=True):

        self.resize_min_dim_to = resize_min_dim_to
        # self.resize_max_dim_to = resize_max_dim_to
        super().__init__(fig=fig, enabled=enabled)

    def process(self):
        """Scales an image so that the smaller dimension has the desired length (specified inside __init__)"""
        min_dim = min(self.fig.img.shape)
        y_dim, x_dim = self.fig.img.shape
        scaling_factor = self.resize_min_dim_to/min_dim
        # max_dim = max(self.fig.img.shape[:2])
        # y_dim, x_dim = self.fig.img.shape
        # scaling_factor = self.resize_max_dim_to / max_dim
        img = cv2.resize(self.fig.img, (int(x_dim*scaling_factor), int(y_dim*scaling_factor)))
        self.fig._scaling_factor = scaling_factor
        self.fig.img = img
        return self.fig

class ImageNormaliser(Processor):
    """Processor used to normalise an image to a range between 0 and 1"""
    def __init__(self, fig, enabled=True):
        super().__init__(fig=fig, enabled=enabled)

    def process(self):
        """Normalises an image to range [0, 1]"""
        img = self.fig.img
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        self.fig.img = img
        return self.fig

# class TextLineRemover(Processor):
#     WIDTH_THRESH_FACTOR = 0.3
#
#     def __init__(self, img, enabled=True):
#         self.selem = np.concatenate((np.zeros((2, 6)), np.ones((2, 6)), np.zeros((2, 6))), axis=0).astype(np.uint8)
#
#         self.top_roi = Rect(0, 0, self.img.shape[0] // 5, self.img.shape[1] // 5)
#         self.bottom_roi = Rect(int(self.img.shape[0] * 4 / 5), 0,
#                                self.img.shape[0], int(self.img.shape[1] // 5))
#
#         super().__init__(enabled=enabled)
#
#     def process(self):
#         if not self._enabled:
#             return self.img
#         img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
#         img = 255 - img
#         img = cv2.dilate(img, self.selem, iterations=6)
#         ret, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
#         contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         print(len(contours))
#         crop_top = []
#         crop_bottom = []
#
#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             print(cv2.boundingRect(cnt))
#             #             print(self.bottom_roi)
#             if w > self.WIDTH_THRESH_FACTOR * self.img.shape[1]:
#                 print('width okay')
#
#                 #                 print(f'{x}, {y}')
#                 if self.top_roi.contains_point((x, y)):
#                     crop_top.append((x, y, x + w, y + h))
#                 elif self.bottom_roi.contains_point((x, y)):
#                     crop_bottom.append((x, y, x + w, y + h))
#
#         # Crop the whole images down to the first bottom text line
#         print(f'crop_bottom: {crop_bottom}')
#         if crop_bottom:
#             crop_bottom_boundary = min([coords[1] for coords in crop_bottom])
#             print(crop_bottom_boundary)
#             self._img = self._img[:crop_bottom_boundary, :]
#         if crop_top:
#             crop_top_boundary = max([coords[1] for coords in crop_top])
#             self._img = self._img[crop_top_boundary:, :]
#         return self._img


class Binariser(Processor):
    """Processor used for binarisation"""

    def __init__(self, fig, bin_thresh=None):
        super().__init__(enabled=True, fig=fig)
        if len(self.img.shape) == 2:
            self.color_mode = self.COLOR_MODE.GRAY
        elif len(self.img.shape) == 3 and self.img.dtype == np.uint8:
            self.color_mode = self.COLOR_MODE.RGB
        self.bin_thresh = bin_thresh if bin_thresh else config.ProcessorConfig.BIN_THRESH

    def process(self):
        """Binarises gray images"""
        if self.color_mode == self.COLOR_MODE.GRAY:
            ret, img = cv2.threshold(self.img, *self.bin_thresh, cv2.THRESH_BINARY)
            fig_copy = deepcopy(self.fig)
            fig_copy.img = img
            return fig_copy


class Isolator(Processor):
    """Processor class used for isolating individual connected components"""
    def __init__(self, fig, to_isolate, isolate_mask):
        super().__init__(fig=fig)
        self.to_isolate = to_isolate
        self.isolate_mask = isolate_mask


    def _isolate_panel(self):
        #TODO
        pass

    def _isolate_mask(self):
        rows, cols = zip(*self.to_isolate.pixels)
        # if self.use_raw_img:
        #     img = self.fig.raw_img
        # else:
        #     img = self.fig.img
        mask = np.zeros_like(self.img, dtype=np.bool)
        mask[rows, cols] = True
        isolated_cc_img = np.zeros_like(self.img, dtype=np.uint8)
        isolated_cc_img[mask] = self.img[mask]
        # if self.fig.scaling_factor:
        #     y_dim, x_dim = img.shape
        #     isolated_cc_img_resized = cv2.resize(isolated_cc_img_resized, (x_dim, y_dim))
        # rows, cols = np.where(isolated_cc_img_resized > 0)
        # mask = np.zeros_like(img, dtype=np.bool)
        # mask[rows, cols] = True
        # isolated_cc_img_orig = np.zeros_like(img)
        # isolated_cc_img_orig[mask] = img[mask]
        fig_copy = deepcopy(self.fig)
        # fig_copy.eager_cc_init = False
        fig_copy.img = isolated_cc_img
        return fig_copy

    def process(self):
        """Isolates connected components either by using their panels or pixel-wise masks"""
        if self.isolate_mask:
            return self._isolate_mask()
        else:
            return self._isolate_panel()
