from enum import Enum
Extractors = Enum('Extractors', 'ORB SuperPoint')
Matchers = Enum('Matchers', 'OrbHamming OrbFlann SuperGlue')# "HAMMING"  # or FLANN