import numpy as np

from models import geometry


def test_skimagetoopencvhoughlineadapter():
    opencv_lines = np.array([[[[1, 2, 3, 4]]], [[[2, 4, 6, 8]]]])
    skimage_lines = [line for line in geometry.OpencvToSkimageHoughLineAdapter(opencv_lines)]
    assert skimage_lines == [((1, 2), (3, 4)), ((2, 4), (6, 8))]


test_skimagetoopencvhoughlineadapter()
