from __future__ import division, absolute_import, print_function
import unittest
from random import randint

from math import cos, pi, sin
from svgpathtools import *
from svgpathtools.svg2paths import *
from os.path import join, dirname



class TestSVG2Paths(unittest.TestCase):
    def test_svg2paths_polygons(self):

        paths, _ = svg2paths(join(dirname(__file__), 'polygons.svg'))

        # triangular polygon test
        path = paths[0]
        path_correct = Path(Line(55.5+0j, 55.5+50j), 
                            Line(55.5+50j, 105.5+50j), 
                            Line(105.5+50j, 55.5+0j)
                            )
        self.assertTrue(path.isclosed())
        self.assertTrue(len(path)==3)
        self.assertTrue(path==path_correct)

        # triangular quadrilateral (with a redundant 4th "closure" point)
        path = paths[1]
        path_correct = Path(Line(0+0j, 0-100j),
                            Line(0-100j, 0.1-100j),
                            Line(0.1-100j, 0+0j),
                            Line(0+0j, 0+0j)  # result of redundant point
                            )
        self.assertTrue(path.isclosed())
        self.assertTrue(len(path)==4)
        self.assertTrue(path==path_correct)

    def test_svg2paths_ellipses(self):

        paths, _ = svg2paths(join(dirname(__file__), 'ellipse.svg'))

        # ellipse tests
        path_ellipse = paths[0]
        path_ellipse_correct = Path(Arc(50+100j, 50+50j, 0.0, True, False, 150+100j),
                                    Arc(150+100j, 50+50j, 0.0, True, False, 50+100j))
        self.assertTrue(len(path_ellipse)==2)
        self.assertTrue(path_ellipse==path_ellipse_correct)
        self.assertTrue(path_ellipse.isclosed())

        # circle tests
        paths, _ = svg2paths(join(dirname(__file__), 'circle.svg'))

        path_circle = paths[0]
        path_circle_correct = Path(Arc(50+100j, 50+50j, 0.0, True, False, 150+100j),
                                    Arc(150+100j, 50+50j, 0.0, True, False, 50+100j))
        self.assertTrue(len(path_circle)==2)
        self.assertTrue(path_circle==path_circle_correct)
        self.assertTrue(path_circle.isclosed())

    def test_svg2paths_transform_path_translate(self):
        width = 100
        path = Path(Line(0 + 0j, 0 + width*1j),
                    Line(0 + width*1j, width + width*1j),
                    Line(width + width*1j, 0 + 0j)
                    )
        x = randint(-1000, 1000)
        y = randint(-1000, 1000)
        d_strings = transform_path((1, 0, 0, 1, x, y), path.d())
        assert d_strings == "M {:.1f},{:.1f} L {:.1f},{:.1f} L {:.1f},{:.1f} L {:.1f},{:.1f}"\
            .format(x, y, x, y+width, x+width, y+width, x, y)

    def test_svg2paths_transform_path_scale(self):
        width = 100
        path = Path(Line(0 + 0j, 0 + width*1j),
                    Line(0 + width*1j, width + width*1j),
                    Line(width + width*1j, 0 + 0j)
                    )
        x = randint(-1000, 1000)
        y = randint(-1000, 1000)
        d_strings = transform_path((x, 0, 0, y, 0, 0), path.d())
        assert d_strings == "M {:.1f},{:.1f} L {:.1f},{:.1f} L {:.1f},{:.1f} L {:.1f},{:.1f}"\
            .format(0, 0, 0, y*width, x*width, y*width, 0, 0)

    def test_svg2paths_transform_path_rotate(self):
        def c(a):
            return cos(a*pi/180.)

        def s(a):
            return sin(a*pi/180.)

        width = 100
        path = Path(Line(0 + 0j, 0 + width * 1j),
                    Line(0 + width * 1j, width + width * 1j),
                    Line(width + width * 1j, 0 + 0j)
                    )
        angle = randint(-180, 180)
        print(angle)
        d_strings = transform_path((c(angle), s(angle), -s(angle), c(angle), 0, 0), path.d())
        assert d_strings == "M {},{} L {},{} L {},{} L {},{}" \
            .format(0.0, 0.0, -width*s(angle), width*c(angle),
                    width*(c(angle)-s(angle)), width*(c(angle)+s(angle)), 0.0, 0.0)

    def test_svg2paths_transform_path_translate_relative(self):
        width = 100.
        path = "m 0,0 l {},{} l {},{} l {},{}".format(0, width, width, 0, -width, -width)
        x = randint(-1000, 1000)
        y = randint(-1000, 1000)
        d_strings = transform_path((1, 0, 0, 1, x, y), path)
        assert d_strings == "m {:.1f},{:.1f} l {},{} l {},{} l {},{}"\
            .format(x, y, 0., width, width, 0., -width, -width)

    def test_svg2paths_transform_path_translate_relative_no_lines(self):
        width = 100.
        path = "m 0,0 {},{} {},{} {},{}".format(0, width, width, 0, -width, -width)
        x = randint(-1000, 1000)
        y = randint(-1000, 1000)
        d_strings = transform_path((1, 0, 0, 1, x, y), path)
        assert d_strings == "m {:.1f},{:.1f} {},{} {},{} {},{}"\
            .format(x, y, 0., width, width, 0., -width, -width)

    def test_svg2paths_transform_path_translate_relative_arc(self):
        x = randint(-1000, 1000)
        y = randint(-1000, 1000)
        width = 25.0
        path = "m 0.0,0.0 a {},{} 0 0 1 -{},{} {},{} 0 0 1 -{},-{} {},{} 0 0 1 {},-{} " \
               "{},{} 0 0 1 {},{} z".format(*[width]*16)
        d_strings = transform_path((1, 0, 0, 1, x, y), path)
        assert d_strings == "m {:.1f},{:.1f} a {},{} 0 0 1 -{},{} {},{} 0 0 1 -{},-{} {},{} 0 0 1 {},-{} " \
               "{},{} 0 0 1 {},{} z".format(x, y, *[width]*16)