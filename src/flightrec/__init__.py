"""FlightRec public API."""

from flightrec.data import IndexedDataset
from flightrec.recorder import FlightRecorder
from flightrec.storage import RunData, read_run

__all__ = ["FlightRecorder", "IndexedDataset", "RunData", "read_run"]
__version__ = "0.1.0"
