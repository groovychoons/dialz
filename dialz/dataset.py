from dataclasses import dataclass
from typing import List
import json
import os


@dataclass
class DatasetEntry:
    """
    Represents a single entry in the dataset, consisting of a positive
    and a negative example.
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

    def add_from_saved(self, saved_entries: List[dict]) -> None:
        """
        Adds entries from a pre-saved dataset.

        Args:
            saved_entries (List[dict]): A list of dictionaries, each containing
                                        "positive" and "negative" keys.
        """
        for entry in saved_entries:
            if "positive" in entry and "negative" in entry:
                self.add_entry(entry["positive"], entry["negative"])
            else:
                raise ValueError(
                    "Each entry must have 'positive' and \
                                 'negative' keys."
                )

    def view_dataset(self) -> List[DatasetEntry]:
        """
        Returns the current dataset as a list of DatasetEntry objects.

        Returns:
            List[DatasetEntry]: The list of all entries in the dataset.
        """
        return self.entries

    def save_to_file(self, file_path: str) -> None:
        """
        Saves the dataset to a JSON file.

        Args:
            file_path (str): The path to the file where the dataset will be \
                saved.
        """
        with open(file_path, "w") as file:
            json.dump([entry.__dict__ for entry in self.entries], file, indent=4)

    @classmethod
    def load_from_file(cls, file_path: str) -> "Dataset":
        """
        Loads a dataset from a JSON file.

        Args:
            file_path (str): The path to the JSON file containing the dataset.

        Returns:
            Dataset: A new Dataset instance loaded from the file.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
        dataset = cls()
        dataset.add_from_saved(data)
        return dataset

    @classmethod
    def load_corpus(cls, name: str) -> "Dataset":
        """
        Loads a default pre-saved corpus included in the package.

        Args:
            name (str): The name of the pre-saved corpus to load.

        Returns:
            Dataset: A new Dataset instance with the default corpus.

        Raises:
            FileNotFoundError: If the specified corpus does not exist.
        """
        # Assuming pre-saved corpora are stored in a 'corpus' folder in the
        # package
        base_path = os.path.join(os.path.dirname(__file__), "corpus")
        file_path = os.path.join(base_path, f"{name}.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Corpus '{name}' not found.")

        return cls.load_from_file(file_path)

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
