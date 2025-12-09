import numpy as np
from dataclasses import dataclass
from collections.abc import Callable

##################################################
#               EKF
##################################################


@dataclass
class EkfParams:
    """ Extended Kalman Filter Parameters
    """
    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    F: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray, np.ndarray], np.ndarray]
    H: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray

@dataclass
class State:
    x: np.ndarray  # Mean vector
    P: np.ndarray  # Covariance matrix


class EKF:
    """https://en.wikipedia.org/wiki/Extended_Kalman_filter"""

    def __init__(self, params: EkfParams):
        """"""
        self.f = params.f
        self.F = params.F
        self.h = params.h
        self.H = params.H
        self.Q = params.Q
        self.R = params.R

    def predict(self, u: )

