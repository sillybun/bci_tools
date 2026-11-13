"""LSL Viewer package."""

from .app import LslStreamCatalog, LslStreamRecord, TkLslViewer
from .version import VERSION_STRING

__all__ = ["LslStreamCatalog", "LslStreamRecord", "TkLslViewer"]
__version__ = VERSION_STRING
