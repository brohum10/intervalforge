"""Small ridge-regression prediction-interval experiments with explicit limits."""

from .calibration import calibrate, threshold
from .data import Dataset
from .experiment import evaluate, experiment
from .model import fit

__version__ = "0.1.0"
__all__ = ["Dataset", "calibrate", "evaluate", "experiment", "fit", "threshold"]
