"""
Arrow segmentation test.
"""
from reactiondataextractor.actions import estimate_single_bond
from reactiondataextractor.extractors.arrows import SolidArrowCandidateExtractor, CurlyArrowCandidateExtractor, ArrowExtractor
from reactiondataextractor.models.segments import Panel, Point
# from models.utils import imread
from reactiondataextractor.processors import ImageReader, EdgeExtractor
PATH_SOLID = 'test_images/rde_test.png'
reader = ImageReader(PATH_SOLID, ImageReader.COLOR_MODE.GRAY)
solid_fig = reader.process()
binarizer = EdgeExtractor(solid_fig)
solid_fig = binarizer.process()

# solid_fig = imread('rde_test.png')
# settings.main_figure.append(solid_fig)
estimate_single_bond(solid_fig)
extractor = SolidArrowCandidateExtractor(solid_fig, )
solid_arrows = extractor.extract()


PATH_CURLY = 'test_images/rde_test_curly.gif'
reader = ImageReader(PATH_CURLY, ImageReader.COLOR_MODE.GRAY)
curly_fig = reader.process()
binarizer = EdgeExtractor(curly_fig)
curly_fig = binarizer.process()

curly_extractor = CurlyArrowCandidateExtractor(curly_fig)
curly_arrows = curly_extractor.extract()
total_arrow_extractor = ArrowExtractor(curly_fig)
total_arrow_extractor.extract()

def test_number_arrows():
    assert len(solid_arrows) == 2

def test_panels():
    panels = [arrow.panel for arrow in solid_arrows]
    assert set(panels) == {Panel((185, 324, 210, 529)), Panel((163, 1164, 187, 1383))}

def test_line_slope():
    for arrow in solid_arrows:
        assert arrow.line.slope == 0

def test_line_intercept():
    for arrow in solid_arrows:
        assert arrow.line.intercept == arrow.pixels[0].row

def test_sides_inequality():
    for arrow in solid_arrows:
        assert len(arrow.prod_side) > len(arrow.react_side)


def test_sides_pixels():
    for arrow in solid_arrows:
        if arrow.left < 800:
            assert Point(197, 427) in arrow.prod_side
            assert Point(202, 488) in arrow.prod_side
            assert Point(197, 329) in arrow.react_side
            assert Point(198, 419) in arrow.react_side
        else:
            assert Point(175, 1277) in arrow.prod_side
            assert Point(175, 1314) in arrow.prod_side
            assert Point(176, 1168) in arrow.react_side
            assert Point(174, 1231) in arrow.react_side

def test_panels_curly():
    panels = [arrow.panel for arrow in curly_arrows]
    assert set(panels) == {Panel((393, 283, 517, 407)),
                      Panel((334, 62, 430, 158)),
                      Panel((165, 458, 204, 497)),
                      Panel((106, 46, 153, 93)),
                      Panel((46, 321, 147, 422)),
                      Panel((30, 96, 68, 134)),
                      Panel((25, 152, 156, 283))}

test_panels()
test_sides_pixels()
test_sides_inequality()
test_line_intercept()
test_line_slope()
test_panels_curly()
