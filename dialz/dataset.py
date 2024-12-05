from dataclasses import dataclass
from typing import List


@dataclass
class DatasetEntry:
    """
    Represents a single entry in the dataset, consisting of a
    positive and a negative example.
    """

    positive: str
    negative: str


class Dataset:
    """
    A class to manage a dataset of positive and negative examples.
    """

    def __init__(self):
        """
        Initializes an empty dataset.
        """
        self.entries: List[DatasetEntry] = []

    def add_entry(self, positive: str, negative: str) -> None:
        """
        Adds a new DatasetEntry to the dataset.

        Args:
            positive (str): The positive example.
            negative (str): The negative example.
        """
        self.entries.append(DatasetEntry(positive=positive, negative=negative))

    def view_dataset(self) -> List[DatasetEntry]:
        """
        Returns the current dataset as a list of DatasetEntry objects.

        Returns:
            List[DatasetEntry]: The list of all entries in the dataset.
        """
        return self.entries

    def __str__(self) -> str:
        """
        Returns a string representation of the dataset for easy viewing.
        """
        return "\n".join(
            [
                f"Positive: {entry.positive}, Negative: {entry.negative}"
                for entry in self.entries
            ]
        )
