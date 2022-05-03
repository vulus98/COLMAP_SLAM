from enum import Enum
Extractors = Enum('Extractors', 'ORB SuperPoint')
Matchers = Enum('Matchers', 'OrbHamming OrbFlann SuperGlue')# "HAMMING"  # or FLANN
ImageSelectionMethod = Enum('ImageSelectionMethod', 'MAX_VISIBLE_POINTS_NUM MAX_VISIBLE_POINTS_RATIO MIN_UNCERTAINTY')