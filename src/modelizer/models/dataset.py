import torch

from string import Template
from typing import Optional

from pandas import DataFrame

from modelizer.configs import VALIDATION_FRACTION
from modelizer.tokenizers.abstract import BaseTokenizer, abstractmethod, ABC


class TorchDataset(ABC, torch.utils.data.Dataset):
    """
    This is a base class for all custom Modelizer models.
    It inherits from torch.utils.data.Dataset.
    Do not create an instance of this class directly.
    """
    def __init__(self):
        super().__init__()
        assert type(self) is not TorchDataset, "TorchDataset is an abstract class and cannot be instantiated directly"

    @abstractmethod
    def __len__(self):
        """This method returns the size of the dataset."""
        raise NotImplementedError("__len__ method not implemented in the subclass")

    @abstractmethod
    def __getitem__(self, idx):
        """
        This method returns a sample from the dataset.
        :param idx: The index of the sample.
        :return: Either a single sample, a tuple, or a dictionary.
        """
        raise NotImplementedError("__getitem__ method not implemented in the subclass")

    @abstractmethod
    def collate_fn(self, batch):
        """
        This method is used to collate a list of samples into a batch.
        It is used by the DataLoader class to create batches.
        :param batch: A list of samples.
        :return: A batch of samples.
        """
        raise NotImplementedError("collate_fn method not implemented in the subclass")

    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        This method returns a DataLoader object for the dataset.
        :param batch_size: The number of samples to process in one iteration.
        :param shuffle: Shuffle the data before creating batches.
        :return: torch.utils.data.DataLoader object.
        """
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def get_dataloaders(self, batch_size: int, validation_fraction: Optional[float] = VALIDATION_FRACTION, shuffle: bool = True):
        """
        This method returns train and validation DataLoader objects for the dataset.
        :param batch_size: The number of samples to process per iteration.
        :param validation_fraction: The fraction of data to use for validation.
        :param shuffle: Shuffle the data before creating batches.
        :return: A tuple of train and validation DataLoader objects.
        """
        if validation_fraction is None or validation_fraction == 0 or validation_fraction > len(self):
            train_dataset = valid_dataset = self
        else:
            assert isinstance(validation_fraction, float), (f"validation_fraction must be a floating point number or None, "
                                                            f"got {type(validation_fraction)} = {validation_fraction} instead.")
            assert 0. < validation_fraction < 1.0, "validation_fraction must be between 0.0 and 1.0"
            valid_size = int(len(self) * validation_fraction)
            train_size = len(self) - valid_size
            assert valid_size > 0, f"Validation Fraction is too small, got {valid_size}"
            assert train_size > 0, f"Train Fraction is too small, got {train_size}"
            train_dataset, valid_dataset = torch.utils.data.random_split(self, [train_size, valid_size])

        train_sampler = valid_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                                   sampler=train_sampler, collate_fn=self.collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                                   sampler=valid_sampler, collate_fn=self.collate_fn)
        return train_loader, valid_loader


class TorchSeq2SeqDataset(TorchDataset):
    """This class is used to create a dataset for Encoder-Decoder models."""
    def __init__(self, dataframe: DataFrame,
                 source: str,
                 target: str,
                 source_tokenizer: BaseTokenizer,
                 target_tokenizer: BaseTokenizer,
                 instructions: Optional[Template] = None):
        assert source in dataframe.columns, f"Source column '{source}' not found in dataframe"
        assert target in dataframe.columns, f"Target column '{target}' not found in dataframe"
        super().__init__()
        self._src_data = dataframe[source].tolist()
        self._tgt_data = dataframe[target].tolist()
        self._src_tokenizer = source_tokenizer
        self._tgt_tokenizer = target_tokenizer
        if instructions is not None:
            for i, row in enumerate(self._src_data):
                self._src_data[i] = instructions.substitute({"source": source, "target": target, "input": row})

    def __len__(self):
        return len(self._src_data)

    def __getitem__(self, idx):
        input_encodings = self._src_tokenizer(self._src_data[idx])
        output_encodings = self._tgt_tokenizer(self._tgt_data[idx])

        return {
            "input_ids": input_encodings["input_ids"],
            "output_ids": output_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "output_mask": output_encodings["attention_mask"]
        }

    def collate_fn(self, batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "output_ids": torch.stack([item["output_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "output_mask": torch.stack([item["output_mask"] for item in batch]),
        }
