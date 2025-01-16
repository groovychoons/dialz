import json
import os

from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

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


    @staticmethod
    def _apply_chat_template(
        tokenizer, 
        system_role: str, 
        content1: str, 
        content2: str,
        add_generation_prompt: bool = True
    ) -> str:
        """
        Applies the chat template to the given content and returns the decoded output.
        """
        messages = []

        # Only add system message if system_role is non-empty
        if system_role:
            messages.append({"role": "system", "content": f"{system_role}{content1}."})

        messages.append({"role": "user", "content": content2})
        
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return tokenized


    @classmethod
    def create_dataset(
        cls, 
        model_name: str, 
        items: list, 
        prompt_type: str = "generic", 
        num_sents: int = 10,
        system_role: str = "Act as if you are extremely "
    ) -> "Dataset":

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        file_path = os.path.join(os.path.dirname(__file__), "corpus", f"{prompt_type}.json")
        with open(file_path, "r", encoding="utf-8") as file:
            variations = json.load(file)

        dataset = Dataset()

        for variation in variations[:num_sents]:
            # Use the helper function for both positive and negative
            positive_decoded = cls._apply_chat_template(tokenizer, system_role, items[0], variation)
            negative_decoded = cls._apply_chat_template(tokenizer, system_role, items[1], variation)

            # Add to dataset
            dataset.add_entry(positive_decoded, negative_decoded)

        return dataset


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
    def load_corpus(
        cls, 
        model_name: str, 
        name: str, 
        num_sents: int = 10
    ) -> "Dataset":
        """
        Loads a default pre-saved corpus included in the package,
        re-applies chat templates to each entry, and limits to num_sents.
        """
        base_path = os.path.join(os.path.dirname(__file__), "corpus")
        file_path = os.path.join(base_path, f"{name}.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Corpus '{name}' not found.")

        # 1. Load the raw data (list of dicts with "positive" and "negative")
        with open(file_path, "r", encoding="utf-8") as file:
            raw_entries = json.load(file)

        # 2. Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # 3. Create a new dataset to store the transformed entries
        processed_dataset = cls()

        # 4. Iterate through the first num_sents entries, apply templates
        for entry in raw_entries[:num_sents]:
            positive_transformed = cls._apply_chat_template(
                tokenizer,
                system_role="",
                content1="",
                content2=entry["positive"]
            )
            negative_transformed = cls._apply_chat_template(
                tokenizer,
                system_role="",
                content1="",
                content2=entry["negative"]
            )
            processed_dataset.add_entry(positive_transformed, negative_transformed)

        return processed_dataset


    def __str__(self) -> str:
        """
        Returns a string representation of the dataset for easy viewing.
        """
        return "\n".join(
            [
                f"Positive: {entry.positive}\nNegative: {entry.negative}"
                for entry in self.entries
            ]
        )
