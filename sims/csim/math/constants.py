import numpy as np

########################################
#               Angle
########################################

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 1 / DEG_TO_RAD



DEG_TO_ARCSEC = 3600
ARCSEC_TO_DEG = 1/3600

ARCSEC_TO_RAD = ARCSEC_TO_DEG * DEG_TO_RAD 
RAD_TO_ARCSEC = 1 / ARCSEC_TO_RAD 


########################################
#               Time
########################################

UTC_TO_GPS = 18
"""Add to UTC to make GPS"""
GPS_TO_UTC = -18
"""Add to GPS to make UTC"""

DAY_TO_SEC = 86400
SEC_TO_DAY = 1/DAY_TO_SEC

########################################
#               Force
########################################

LBF_TO_N = 4.44822
N_TO_LBF = 1/LBF_TO_N

KG_TO_LBM = 2.20462
LBM2KG = 1 / KG_TO_LBM