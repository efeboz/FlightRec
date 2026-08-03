"""Dataset helpers."""

from typing import Any, Generic, TypeVar

from torch.utils.data import Dataset

T = TypeVar("T")


class IndexedDataset(Dataset[tuple[Any, Any, int]], Generic[T]):
    """Wrap a map-style dataset and append a stable sample index."""

    def __init__(self, dataset: Dataset[T]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the wrapped dataset length."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any, int]:
        """Return ``(input, target, index)`` for one sample."""
        item = self.dataset[index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError("IndexedDataset requires items shaped like (input, target)")
        return item[0], item[1], index
