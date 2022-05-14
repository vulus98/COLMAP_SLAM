from enum import Enum
Extractors = Enum('Extractors', 'SuperPoint ORB')
Matchers = Enum('Matchers', 'SuperGlue OrbHamming OrbFlann')# "HAMMING"  # or FLANN
ImageSelectionMethod = Enum('ImageSelectionMethod', 'MAX_VISIBLE_POINTS_NUM MAX_VISIBLE_POINTS_RATIO MIN_UNCERTAINTY')