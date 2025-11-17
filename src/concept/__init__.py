from importlib.metadata import version

from .model import ContrastiveModel
from .scConcept import scConcept

__all__ = ["ContrastiveModel", "scConcept"]

__version__ = version("scConcept")
