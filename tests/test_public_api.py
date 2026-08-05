from flightrec import FlightRecorder, IndexedDataset, RunData, read_run
from flightrec.analysis import InfluenceConfig, influence_on, self_influence


def test_documented_public_api_is_reexported():
    assert FlightRecorder is not None
    assert IndexedDataset is not None
    assert RunData is not None
    assert callable(read_run)
    assert InfluenceConfig is not None
    assert callable(influence_on)
    assert callable(self_influence)
