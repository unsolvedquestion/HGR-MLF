"""Legacy transformer utilities kept for compatibility.

These modules are not used by the main DocRE training pipeline, but they are
now importable and documented instead of being half-commented placeholders.
"""

from . import Beam
from . import Constants
from . import Layers
from . import Models
from . import Modules
from . import Optim
from . import SubLayers
from . import Translator

__all__ = [
    "Beam",
    "Constants",
    "Layers",
    "Models",
    "Modules",
    "Optim",
    "SubLayers",
    "Translator",
]
