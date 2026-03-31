import torch

from pandas import DataFrame

from torch.utils.data import (
    Dataset,
    DataLoader,
    DistributedSampler,
    random_split,
)

from modelizer import configs
from modelizer.tokenizers.abstract import BaseTokenizer


class TorchDataset(Dataset):
    def __init__(self,
                 dataframe: DataFrame,
                 source: str,
                 target: str,
                 source_tokenizer: BaseTokenizer,
                 target_tokenizer: BaseTokenizer,
                 sample_count: int | float = 0):

        assert isinstance(dataframe, DataFrame) and not dataframe.empty, "DataFrame must be a non-empty pandas DataFrame"
        assert source in dataframe.columns, f"Source column {source} not found in DataFrame"
        assert target in dataframe.columns, f"Target column {target} not found in DataFrame"

        source_data = dataframe[source].tolist()
        target_data = dataframe[target].tolist()
        if sample_count > 0:
            if isinstance(sample_count, int):
                source_data = source_data[:sample_count]
                target_data = target_data[:sample_count]
            else:
                sample_size = int(len(source_data) * sample_count)
                source_data = source_data[:sample_size]
                target_data = target_data[:sample_size]
        assert len(source_data) == len(target_data), "Source and target data must have same length"
        assert len(source_data) > 0, "Source and target data must not be empty"

        self._source_data = [source_tokenizer(entry, return_tensors=True) for entry in source_data]
        self._target_data = [target_tokenizer(entry, return_tensors=True) for entry in target_data]

    def __len__(self):
        return len(self._source_data)

    def __getitem__(self, index):
        return self._source_data[index], self._target_data[index]

    @staticmethod
    def __create_batch__(data_entry):
        source_dicts, target_dicts = zip(*data_entry)
        return {
            'source': torch.stack([d['input_ids'] for d in source_dicts]),
            'target': torch.stack([d['input_ids'] for d in target_dicts]),
        }

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = False, pin_memory: bool = True, use_distributed_sampler: bool = False) -> DataLoader:
        sampler = DistributedSampler(self) if use_distributed_sampler else None
        return DataLoader(self, pin_memory=pin_memory, shuffle=shuffle, batch_size=batch_size, sampler=sampler, collate_fn=TorchDataset.__create_batch__)

    def get_dataloaders(self, validation_fraction: float, batch_size: int = 1, shuffle: bool = False,
                        pin_memory: bool = True, use_distributed_sampler: bool = False, seed: int = configs.SEED) -> tuple[DataLoader, DataLoader]:
        if validation_fraction is None or validation_fraction == 0:
            train_data = valid_data = self
        elif isinstance(validation_fraction, float):
            assert 0.0 < validation_fraction < 1.0, "Train Fraction must be between 0.0 and 1.0"
            valid_size = int(len(self) * validation_fraction)
            train_size = len(self) - valid_size
            assert valid_size > 0, f"Validation Fraction is too small, got {valid_size}"
            assert train_size > 0, f"Train Fraction is too small, got {train_size}"
            train_data, valid_data = random_split(self, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))
        else:
            raise ValueError(f"validation_fraction must be a floating point number or None, got {type(validation_fraction)} = {validation_fraction} instead.")

        if use_distributed_sampler:
            train_sampler, valid_sampler = DistributedSampler(train_data), DistributedSampler(valid_data)
            shuffle = False
        else:
            train_sampler = valid_sampler = None
        train_loader = DataLoader(train_data, pin_memory=pin_memory, shuffle=shuffle, collate_fn=self.__create_batch__, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(valid_data, pin_memory=pin_memory, shuffle=shuffle, collate_fn=self.__create_batch__, batch_size=batch_size, sampler=valid_sampler)
        return train_loader, valid_loader
