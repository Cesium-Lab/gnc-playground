import numpy as np

########################################
#               Angle
########################################

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 1 / DEG_TO_RAD

ARCSEC_TO_RAD = DEG_TO_RAD 

DEG_TO_ARCSEC = 3600
ARCSEC_TO_DEG = 1/3600

########################################
#               Time
########################################

UTC_TO_GPS = 18
"""Add to UTC to make GPS"""
GPS_TO_UTC = -18
"""Add to GPS to make UTC"""

DAY_TO_SEC = 86400
SEC_TO_DAY = 1/DAY_TO_SEC