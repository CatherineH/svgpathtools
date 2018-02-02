from xml.dom.minidom import parseString

from svgpathtools import svgdoc2paths, wsvg

example_text = '<svg>' \
               '  <rect x="100" y="100" height="200" width="200" style="fill:#0ff;" />' \
               '  <line x1="200" y1="200" x2="200" y2="300" />' \
               '  <line x1="200" y1="200" x2="300" y2="200" />' \
               '  <line x1="200" y1="200" x2="100" y2="200" />' \
               '  <line x1="200" y1="200" x2="200" y2="100" />' \
               '  <circle cx="200" cy="200" r="30" style="fill:#00f;" />' \
               '  <circle cx="200" cy="300" r="30" style="fill:#0f0;" />' \
               '  <circle cx="300" cy="200" r="30" style="fill:#f00;" />' \
               '  <circle cx="100" cy="200" r="30" style="fill:#ff0;" />' \
               '  <circle cx="200" cy="100" r="30" style="fill:#f0f;" />' \
               '  <text x="50" y="50" font-size="24">' \
               '   Testing SVG  </text></svg>'

doc = parseString(example_text)

paths, attributes = svgdoc2paths(doc)

wsvg(paths)