import numpy as np

########################################
#               Angle
########################################

DEG2RAD = np.pi / 180.0
RAD2DEG = 1 / DEG2RAD

########################################
#               Time
########################################

UTC_TO_GPS = 18
"""Add to UTC to make GPS"""
GPS_TO_UTC = -18
"""Add to GPS to make UTC"""

SEC_TO_DAY = 86400
DAY_TO_SEC = 1/SEC_TO_DAY