"""This submodule contains tools for creating path objects from SVG files.
The main tool being the svg2paths() function."""

# External dependencies
from __future__ import division, absolute_import, print_function

from re import findall
from xml.dom.minidom import parse
import tinycss
from os import path as os_path, getcwd
import re


from numpy import matmul, cos, sin, tan
from svgpathtools import wsvg, Line, QuadraticBezier, Path, CubicBezier

try:
    from freetype import Face
except ImportError:
    Face = None
# Internal dependencies
from .parser import parse_path


COORD_PAIR_TMPLT = re.compile(
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)' +
    r'(?:\s*,\s*|\s+|(?=-))' +
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)'
)

def combine_transforms(t1, t2):
    assert len(t1) == 6
    assert len(t2) == 6
    m1 = [[t1[0], t1[2], t1[4]], [t1[1], t1[3], t1[5]], [0, 0, 1]]
    m2 = [[t2[0], t2[2], t2[4]], [t2[1], t2[3], t2[5]], [0, 0, 1]]
    out = matmul(m1, m2)
    return out[0][0], out[1][0], out[0][1], out[1][1], out[0][2], out[1][2]


def get_transform(input_dict):
    """ Get the x/y transforms """
    if "transform" in input_dict:
        if input_dict["transform"].find("translate") == 0:
            numbers = input_dict["transform"].split("translate(")[1].split(")")[0].split(
                ",")
            if len(numbers) != 2:
                return 1, 0, 0, 1, 0, 0
            return 1, 0, 0, 1, float(numbers[0]), float(numbers[1])
        elif input_dict["transform"].find("matrix(") == 0:
            output = input_dict["transform"].replace("matrix(", "").replace(")", ""). \
                replace(",", " ").split(" ")
            return [float(x) for x in output]
        elif input_dict["transform"].find("scale(") == 0:
            output = input_dict["transform"].replace("scale(", "").replace(")", ""). \
                replace(",", " ").split(" ")
            return float(output[0]), 0, 0, float(output[1]), 0, 0
        elif input_dict["transform"].find("rotate(") == 0:
            output = input_dict["transform"].replace("rotate(", "").replace(")", ""). \
                replace(",", " ").split(" ")
            output = [float(x) for x in output]
            rotate = [[cos(output[0]), -sin(output[0]), 0],
                      [sin(output[0]), cos(output[0]), 1], [0, 0, 1]]
            if len(output) == 1:
                return rotate[0][0], rotate[1][0], rotate[0][1], rotate[1][1], rotate[0][
                    2], rotate[1][2]
            trans1 = [[1, 0, output[1]], [0, 1, output[2]], [0, 0, 1]]
            trans2 = [[1, 0, -output[1]], [0, 1, -output[2]], [0, 0, 1]]
            out = matmul(rotate, trans2)
            out = matmul(trans1, out)
            return out[0][0], out[1][0], out[0][1], out[1][1], out[0][2], out[1][2]
        elif input_dict["transform"].find("skewX(") == 0:
            out = float(input_dict["transform"].replace("skewX(", "").replace(")", ""))
            return 1, 0, tan(out), 1, 0, 0
        elif input_dict["transform"].find("skewY(") == 0:
            out = float(input_dict["transform"].replace("skewY(", "").replace(")", ""))
            return 1, tan(out), 0, 1, 0, 0
        else:
            return 1, 0, 0, 1, 0, 0
    else:
        return 1, 0, 0, 1, 0, 0


def transform_path_string(transform, path):
    path_parts = findall(r"([\w][\s\d\.\,\-]+|[zZ])", path)
    for i, part in enumerate(path_parts):
        numbers = findall(r"[\-\d\.]+", part)
        # arcs are handled differently because not all parameters are coordinates
        if part[0].lower() == 'a':
            # A rx ry x-axis-rotation large-arc-flag sweep-flag x y
            # there can be multiple arcs in a path
            path_parts[i] = ""
            for offset in range(0, int(len(numbers) / 7)):
                if offset == 0:
                    path_parts[i] += part[0]

                path_parts[i] += " " + transform_point(
                    "%s,%s" % (numbers[0 + offset * 7], numbers[1 + offset * 7]),
                    transform, format='str',
                    relative=True) + " " + \
                                 " ".join(numbers[2 + offset * 7:5 + offset * 7]) + " " + \
                                 transform_point("%s,%s" % (
                                 numbers[5 + offset * 7], numbers[6 + offset * 7]),
                                                 transform, format='str',
                                                 relative=part[0] == 'a')
        else:
            paired = ["%s,%s" % (numbers[j], numbers[j + 1]) for j in
                      range(0, len(numbers) - 1, 2)]
            # the relative move has to be handled differently
            if part[0] == "m":
                path_parts[i] = part[0] + " " + \
                                " ".join([transform_point(p, transform, format='str',
                                                          relative=j != 0)
                                          for j, p in enumerate(paired)])
            elif part[0].lower() == "z":
                path_parts[i] = part[0]
            else:
                relative = part[0].islower()
                path_parts[i] = part[0] + " " + \
                                " ".join([transform_point(p, transform, format='str',
                                                          relative=relative)
                                          for p in paired])
    # go through list, transform the points, and then rejoin the string
    return " ".join(path_parts)


def transform_path(transform, path):
    if isinstance(path, basestring):
        return transform_path_string(transform, path)
    # if not a string, it's probably a Path object
    segments = path._segments
    for segment in segments:
        if isinstance(segment, CubicBezier):
            segment.start = transform_point(segment.start, matrix=transform,
                                            format="complex")
            segment.end =   transform_point(segment.end, matrix=transform,
                                            format="complex")
            segment.control1 = transform_point(segment.control1, matrix=transform,
                                          format="complex")
            segment.control2 = transform_point(segment.control2, matrix=transform,
                                          format="complex")
        elif isinstance(segment, Line):
            segment.start = transform_point(segment.start, matrix=transform,
                                            format="complex")

            segment.end = transform_point(segment.end, matrix=transform,
                                          format="complex")
        else:
            raise ValueError("not sure how to handle {}".format(type(segment)))
    return Path(*segments)


def transform_point(point, matrix=(1, 0, 0, 1, 0, 0), format="float", relative=False):
    a, b, c, d, e, f = matrix
    if isinstance(point, complex):
        x, y = point.real, point.imag
    elif isinstance(point, int):
        x, y = point, 0
    elif isinstance(point, list):
        x, y = point
    else:
        point_parts = point.split(',')
        if len(point_parts) >= 2:
            # certain svg editors (like illustrator) express points as mathematical
            # operations
            x, y = [float(x) for x in point_parts]
        else:
            # probably got a letter describing the point, i.e., m or z
            return point
    # if the transform is relative, don't apply the translation
    if relative:
        x, y = a * x + c * y, b * x + d * y
    else:
        x, y = a * x + c * y + e, b * x + d * y + f
    if format == "float":
        return x, y
    elif format == "complex":
        return x + y*1j
    else:
        return "%s,%s" % (x, y)


def parse_style(element):
    parser = tinycss.make_parser('page3')
    stylesheet = parser.parse_stylesheet_bytes(element.childNodes[0].nodeValue)
    rules = {}
    for rule in stylesheet.rules:
        _decs = {}
        for dec in rule.declarations:
            _decs[dec.name] = dec.value[0].value
        rules["{}{}".format(*[d.value for d in rule.selector[0:2]])] = _decs
    return rules


def dom2dict(element):
    """Converts DOM elements to dictionaries of attributes."""
    if element.attributes is None:  # sometimes groups don't have any attributes
        return {}
    keys = list(element.attributes.keys())
    values = [val.value for val in list(element.attributes.values())]
    return dict(list(zip(keys, values)))


def combine_styles(domdict, style):
    # if there is already a style tag, ignore stylesheet
    if "style" in domdict:
        return domdict

    if "class" in domdict and ".{}".format(domdict["class"]) in style:
        domdict["style"] = ";".join([":".join([k,v]) for k,v in style[".{}".format(domdict["class"])].items()])

    if "id" in domdict and ".{}".format(domdict["id"]) in style:
        domdict["style"] = ";".join(
            [":".join([k, v]) for k, v in style[".{}".format(domdict["id"])].items()])

    return domdict


def ellipse2pathd(ellipse, group_transform=(1, 0, 0, 1, 0, 0)):
    """converts the parameters from an ellipse or a circle to a string for a 
    Path object d-attribute"""
    cx = float(ellipse.get('cx', None))
    cy = float(ellipse.get('cy', None))
    rx = ellipse.get('rx', None)
    ry = ellipse.get('ry', None)
    r = ellipse.get('r', None)

    if r is not None:
        rx = ry = float(r)
    else:
        rx = float(rx)
        ry = float(ry)
    cx = float(cx)
    cy = float(cy)
    transform = get_transform(ellipse)
    transform = combine_transforms(transform, group_transform)
    cx, cy = transform_point([cx, cy], transform)

    d = ''
    d += 'M' + str(cx - rx) + ',' + str(cy)
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(2 * rx) + ',0'
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(-2 * rx) + ',0'

    return d


def polyline2pathd(element, is_polygon=False, group_transform=(1, 0, 0, 1, 0, 0)):
    """converts the string from a polyline points-attribute to a string for a
    Path object d-attribute"""
    polyline_d = element["points"]
    transform = get_transform(element)
    transform = combine_transforms(transform, group_transform)

    points = COORD_PAIR_TMPLT.findall(polyline_d)
    closed = (float(points[0][0]) == float(points[-1][0]) and
              float(points[0][1]) == float(points[-1][1]))

    # The `parse_path` call ignores redundant 'z' (closure) commands
    # e.g. `parse_path('M0 0L100 100Z') == parse_path('M0 0L100 100L0 0Z')`
    # This check ensures that an n-point polygon is converted to an n-Line path.
    if is_polygon and closed:
        points.append(points[0])
    points = [transform_point(p, transform) for p in points]
    d = 'M' + 'L'.join('{0} {1}'.format(x,y) for x,y in points)
    if is_polygon or closed:
        d += 'z'
    return d


def polygon2pathd(polyline_d, group_transform=(1, 0, 0, 1, 0, 0)):
    """converts the string from a polygon points-attribute to a string 
    for a Path object d-attribute.
    Note:  For a polygon made from n points, the resulting path will be
    composed of n lines (even if some of these lines have length zero).
    """
    return polyline2pathd(polyline_d, True, group_transform)


def rect2pathd(rect, group_transform=(1, 0, 0, 1, 0, 0)):
    """Converts an SVG-rect element to a Path d-string.
    
    The rectangle will start at the (x,y) coordinate specified by the 
    rectangle object and proceed counter-clockwise."""
    x0, y0 = float(rect.get('x', 0)), float(rect.get('y', 0))
    w, h = float(rect["width"]), float(rect["height"])
    x1, y1 = x0 + w, y0
    x2, y2 = x0 + w, y0 + h
    x3, y3 = x0, y0 + h

    transform = get_transform(rect)
    transform = combine_transforms(transform, group_transform)

    x0, y0 = transform_point([x0, y0], transform)
    x1, y1 = transform_point([x1, y1], transform)
    x2, y2 = transform_point([x2, y2], transform)
    x3, y3 = transform_point([x3, y3], transform)

    d = ("M{} {} L {} {} L {} {} L {} {} z"
         "".format(x0, y0, x1, y1, x2, y2, x3, y3))
    return d


def text2pathd(text, group_transform=(1, 0, 0, 1, 0, 0)):
    attributes = dom2dict(text)
    if "font-size" in attributes:
        font_size = float(attributes["font-size"])
    elif "style" in attributes:
        if attributes["style"].find("font-size") >= 0:
            font_size = attributes["style"].split("font-size:")[1].split(";")[0]
            font_size = float(font_size.replace("px", ""))
        else:
            font_size = 12
    else:
        font_size = 12
    if "x" in attributes:
        x_global_offset = float(attributes["x"])
    else:
        x_global_offset = 0
    if "y" in attributes:
        y_global_offset = float(attributes["y"])
    else:
        y_global_offset = 0
    if hasattr(text.childNodes[0], "data"):
        text_string = text.childNodes[0].data
    else:
        flow_para = text.getElementsByTagName('flowPara')
        if flow_para:
            text_string = flow_para[0].childNodes[0].data
    # strip newline characters from the string, they aren't rendered in svg
    text_string = text_string.replace("\n", "").replace("\r", "")

    def tuple_to_imag(t):
        return t[0] + t[1] * 1j

    # keep fonts with repository, as dealing with importing fonts across platforms is a
    # nightmare
    foldername = os_path.dirname(os_path.abspath(__file__))
    face = Face(os_path.join(foldername, 'Vera.ttf'))

    face.set_char_size(48 * 64)
    scale = font_size / face.size.height
    outlines = []
    current_x = 0
    transform = get_transform(text)
    transform = combine_transforms(transform, group_transform)
    x_global_offset, y_global_offset = transform_point([x_global_offset, y_global_offset],
                                                       transform)
    for i, letter in enumerate(text_string):
        face.load_char(letter)
        outline = face.glyph.outline
        if i != 0:
            kerning = face.get_kerning(text_string[i - 1], text_string[i])
            kerning_x = kerning.x
        else:
            kerning_x = 0

        if text_string[i] == ' ':
            # a space is usually 30% of the widest character, capital W
            char_width = face.size.max_advance * 0.3
            char_height = 0
            char_offset = 0
        else:
            char_width = outline.get_bbox().xMax
            char_offset = face.size.height - outline.get_bbox().yMax
            char_height = outline.get_bbox().yMax

        outline_dict = {}
        current_x += kerning_x
        outline_dict["points"] = [(scale * (p[0] + current_x) + x_global_offset,
                                   scale * (
                                       char_offset + char_height - p[
                                           1]) + y_global_offset)
                                  for p in outline.points]
        outline_dict["contours"] = outline.contours
        outline_dict["tags"] = outline.tags
        outlines.append(outline_dict)
        current_x += char_width

    paths = []
    for outline in outlines:
        start, end = 0, 0
        for i in range(len(outline["contours"])):
            end = outline["contours"][i]
            points = outline["points"][start:end + 1]
            points.append(points[0])
            tags = outline["tags"][start:end + 1]
            tags.append(tags[0])

            segments = [[points[0], ], ]
            for j in range(1, len(points)):
                segments[-1].append(points[j])
                if tags[j] and j < (len(points) - 1):
                    segments.append([points[j], ])
            for segment in segments:
                if len(segment) == 2:
                    paths.append(Line(start=tuple_to_imag(segment[0]),
                                      end=tuple_to_imag(segment[1])))
                elif len(segment) == 3:
                    paths.append(QuadraticBezier(start=tuple_to_imag(segment[0]),
                                                 control=tuple_to_imag(segment[1]),
                                                 end=tuple_to_imag(segment[2])))
                elif len(segment) == 4:
                    C = ((segment[1][0] + segment[2][0]) / 2.0,
                         (segment[1][1] + segment[2][1]) / 2.0)

                    paths.append(QuadraticBezier(start=tuple_to_imag(segment[0]),
                                                 control=tuple_to_imag(segment[1]),
                                                 end=tuple_to_imag(C)))
                    paths.append(QuadraticBezier(start=tuple_to_imag(C),
                                                 control=tuple_to_imag(segment[2]),
                                                 end=tuple_to_imag(segment[3])))
            start = end + 1

    path = Path(*paths)
    return path.d()


def svg2paths(svg_file_location,
              return_svg_attributes=False,
              convert_circles_to_paths=True,
              convert_ellipses_to_paths=True,
              convert_lines_to_paths=True,
              convert_polylines_to_paths=True,
              convert_polygons_to_paths=True,
              convert_rectangles_to_paths=True):
    """Converts an SVG into a list of Path objects and attribute dictionaries.

    Converts an SVG file into a list of Path objects and a list of
    dictionaries containing their attributes.  This currently supports
    SVG Path, Line, Polyline, Polygon, Circle, and Ellipse elements.

    Args:
        svg_file_location (string): the location of the svg file
        return_svg_attributes (bool): Set to True and a dictionary of
            svg-attributes will be extracted and returned.  See also the
            `svg2paths2()` function.
        convert_circles_to_paths: Set to False to exclude SVG-Circle
            elements (converted to Paths).  By default circles are included as
            paths of two `Arc` objects.
        convert_ellipses_to_paths (bool): Set to False to exclude SVG-Ellipse
            elements (converted to Paths).  By default ellipses are included as
            paths of two `Arc` objects.
        convert_lines_to_paths (bool): Set to False to exclude SVG-Line elements
            (converted to Paths)
        convert_polylines_to_paths (bool): Set to False to exclude SVG-Polyline
            elements (converted to Paths)
        convert_polygons_to_paths (bool): Set to False to exclude SVG-Polygon
            elements (converted to Paths)
        convert_rectangles_to_paths (bool): Set to False to exclude SVG-Rect
            elements (converted to Paths).

    Returns:
        list: The list of Path objects.
        list: The list of corresponding path attribute dictionaries.
        dict (optional): A dictionary of svg-attributes (see `svg2paths2()`).
    """
    if os_path.dirname(svg_file_location) == '':
        svg_file_location = os_path.join(getcwd(), svg_file_location)

    doc = parse(svg_file_location)

    return svgdoc2paths(doc, return_svg_attributes=return_svg_attributes,
                        convert_circles_to_paths=convert_circles_to_paths,
                        convert_ellipses_to_paths=convert_ellipses_to_paths,
                        convert_lines_to_paths=convert_lines_to_paths,
                        convert_polylines_to_paths=convert_polylines_to_paths,
                        convert_polygons_to_paths=convert_polygons_to_paths,
                        convert_rectangles_to_paths=convert_rectangles_to_paths)


def svgdoc2paths(doc,
                 return_svg_attributes=False,
                 convert_circles_to_paths=True,
                 convert_ellipses_to_paths=True,
                 convert_lines_to_paths=True,
                 convert_polylines_to_paths=True,
                 convert_polygons_to_paths=True,
                 convert_rectangles_to_paths=True,
                 convert_text_to_paths=True,
                 transform=(1, 0, 0, 1, 0, 0),
                 style={}):
    """Converts an SVG into a list of Path objects and attribute dictionaries.

    Converts an SVG file into a list of Path objects and a list of
    dictionaries containing their attributes.  This currently supports
    SVG Path, Line, Polyline, Polygon, Circle, and Ellipse elements.

    Args:
        svg_file_location (string): the location of the svg file
        return_svg_attributes (bool): Set to True and a dictionary of
            svg-attributes will be extracted and returned.  See also the
            `svg2paths2()` function.
        convert_circles_to_paths: Set to False to exclude SVG-Circle
            elements (converted to Paths).  By default circles are included as
            paths of two `Arc` objects.
        convert_ellipses_to_paths (bool): Set to False to exclude SVG-Ellipse
            elements (converted to Paths).  By default ellipses are included as
            paths of two `Arc` objects.
        convert_lines_to_paths (bool): Set to False to exclude SVG-Line elements
            (converted to Paths)
        convert_polylines_to_paths (bool): Set to False to exclude SVG-Polyline
            elements (converted to Paths)
        convert_polygons_to_paths (bool): Set to False to exclude SVG-Polygon
            elements (converted to Paths)
        convert_rectangles_to_paths (bool): Set to False to exclude SVG-Rect
            elements (converted to Paths).

    Returns:
        list: The list of Path objects.
        list: The list of corresponding path attribute dictionaries.
        dict (optional): A dictionary of svg-attributes (see `svg2paths2()`).
    """
    # check for style tags
    if not style:
        style_tags = [node for node in doc.childNodes if node.nodeName == 'style']

        if len(style_tags) > 0:
            style = parse_style(style_tags[0])
    # first check for group transforms
    groups = [node for node in doc.childNodes if
              node.nodeName == 'g' or node.nodeName == 'svg']
    output = [[], [], []]
    for group in groups:
        gt = get_transform(dom2dict(group))
        group_transform = combine_transforms(transform, gt)
        group_output = svgdoc2paths(group, return_svg_attributes=return_svg_attributes,
                                    convert_circles_to_paths=convert_circles_to_paths,
                                    convert_ellipses_to_paths=convert_ellipses_to_paths,
                                    convert_lines_to_paths=convert_lines_to_paths,
                                    convert_polylines_to_paths=convert_polylines_to_paths,
                                    convert_polygons_to_paths=convert_polygons_to_paths,
                                    convert_rectangles_to_paths=convert_rectangles_to_paths,
                                    transform=group_transform, style=style)

        for i in range(len(group_output)):
            output[i] = output[i] + group_output[i]

    # TODO: make this preserve the order of the shapes
    d_strings = []
    attribute_dictionary_list = []
    for el in doc.childNodes:
        # Use minidom to extract path strings from input SVG
        if el.nodeName == 'path':
            domdict = dom2dict(el)
            domdict = combine_styles(domdict, style)
            path_transform = get_transform(domdict)
            path = parse_path(domdict['d'])
            d_strings += [transform_path(combine_transforms(path_transform, transform),
                                         path)]
            attribute_dictionary_list += [domdict]
        # Use minidom to extract polyline strings from input SVG, convert to
        # path strings, add to list
        elif el.nodeName == 'polyline' and convert_polylines_to_paths:
            plin = dom2dict(el)
            d_strings += [polyline2pathd(plin, transform)]
            attribute_dictionary_list += [plin]
        # Use minidom to extract polygon strings from input SVG, convert to
        # path strings, add to list
        elif el.nodeName == 'polygon' and convert_polygons_to_paths:
            pgon = dom2dict(el)
            d_strings += [polygon2pathd(pgon, transform)]
            attribute_dictionary_list += [pgon]
        elif el.nodeName == 'line' and convert_lines_to_paths:
            def tlp(l, part):
                # transform line part
                return str(float(l[part]) + transform[part[0] == 'y'])

            line = dom2dict(el)
            d_strings += [('M' + tlp(line, 'x1') + ' ' + tlp(line, 'y1') +
                           'L' + tlp(line, 'x2') + ' ' + tlp(line, 'y2'))]
            attribute_dictionary_list += [line]
        elif el.nodeName == 'ellipse' and convert_ellipses_to_paths:
            ellipse = dom2dict(el)
            d_strings += [ellipse2pathd(ellipse, transform)]
            attribute_dictionary_list += [ellipse]
        elif el.nodeName == 'circle' and convert_circles_to_paths:
            circle = dom2dict(el)
            d_strings += [ellipse2pathd(circle, transform)]
            attribute_dictionary_list += [circle]
        elif el.nodeName == 'rect' and convert_rectangles_to_paths:
            rectangle = dom2dict(el)
            d_strings += [rect2pathd(rectangle, transform)]
            attribute_dictionary_list += [rectangle]
        elif el.nodeName in ['text', 'flowRoot'] and Face is not None:
            d_strings += [text2pathd(el)]
            attribute_dictionary_list += [dom2dict(el)]

    if return_svg_attributes:
        svg_attributes = dom2dict(doc.getElementsByTagName('svg')[0])
        path_list = [parse_path(d) for d in d_strings]
        return path_list + output[0], attribute_dictionary_list + output[
            1], svg_attributes + output[2]
    else:
        path_list = [parse_path(d) if isinstance(d, basestring) else d for d in d_strings]
        return path_list + output[0], attribute_dictionary_list + output[1]


def svg2paths2(svg_file_location,
               return_svg_attributes=True,
               convert_circles_to_paths=True,
               convert_ellipses_to_paths=True,
               convert_lines_to_paths=True,
               convert_polylines_to_paths=True,
               convert_polygons_to_paths=True,
               convert_rectangles_to_paths=True):
    """Convenience function; identical to svg2paths() except that
    return_svg_attributes=True by default.  See svg2paths() docstring for more
    info."""
    return svg2paths(svg_file_location=svg_file_location,
                     return_svg_attributes=return_svg_attributes,
                     convert_circles_to_paths=convert_circles_to_paths,
                     convert_ellipses_to_paths=convert_ellipses_to_paths,
                     convert_lines_to_paths=convert_lines_to_paths,
                     convert_polylines_to_paths=convert_polylines_to_paths,
                     convert_polygons_to_paths=convert_polygons_to_paths,
                     convert_rectangles_to_paths=convert_rectangles_to_paths)
