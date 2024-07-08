import torch

from pathlib import Path
from typing import Type
from itertools import chain
from collections import Counter

from pandas import read_csv
from torchtext.vocab import vocab, Vocab
from torch.utils.data import Dataset, DataLoader, random_split

from modelizer.utils import pickle_load, pickle_dump
from modelizer.generators.utils import PlaceholderProcessor
from modelizer.tokenizer.generic import AbstractTokenizer
from modelizer.tokenizer.mapping import TOKENIZERS_MAPPING
from modelizer.tokenizer.config import SPECIALS, SPECIAL_IDX, SEED


def load_data(filepath: str | Path) -> dict | None:
    filepath = Path(filepath).resolve() if isinstance(filepath, str) else filepath
    if filepath.exists():
        if "pickle" in filepath.suffix:
            data = pickle_load(filepath)
        elif "xlsx" in filepath.suffix:
            df = read_csv(filepath, index_col=0)
            data = {column: df[column].tolist() for column in df.columns}
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        assert isinstance(data, dict), f"Loaded data from {filepath} is not a dictionary"
        return data
    else:
        raise FileNotFoundError(f"Dataset file not found at {filepath.as_posix()}")


def create_vocab(mapping, column_name, filepath: Path | None):
    counter = Counter(chain(*mapping[column_name]))
    v = vocab(counter, specials=SPECIALS)
    v.set_default_index(SPECIAL_IDX["<UNK>"])
    if filepath is not None:
        torch.save(v, filepath.joinpath(f"{column_name}_vocab.pth"))
    return v


def load_vocabs(filepath: str | Path) -> dict[str, Vocab]:
    filepath = Path(filepath).resolve() if isinstance(filepath, str) else filepath
    filepath.mkdir(parents=True, exist_ok=True)
    vocabs = {vocab_file.stem.split("_voc", 1)[0]: torch.load(vocab_file.as_posix()) for vocab_file in filepath.glob("*_vocab.pth")}
    if not len(vocabs):
        vocabs = create_merge_vocab(filepath.parent.parent, filepath.name, "simplified" in filepath.name)
    assert len(vocabs), f"No vocabularies loaded"
    return vocabs


# this function should be used to save model-specific vocabs
def save_vocabs(vocabs: dict[str, Vocab], filepath: str | Path):
    filepath = Path(filepath).resolve() if isinstance(filepath, str) else filepath
    filepath.mkdir(parents=True, exist_ok=True)
    for data_type in vocabs:
        torch.save(vocabs[data_type], filepath.joinpath(f"{data_type}_vocab.pth"))


def create_merge_vocab(root_dir: str | Path, subject: str, simplified: bool) -> dict[str, Vocab]:
    root_dir = Path(root_dir).resolve() if isinstance(root_dir, str) else root_dir
    assert root_dir.is_dir(), f"Directory with datasets not found at {root_dir.as_posix()}"
    merged_data = None
    for data_dir in root_dir.glob("data_*"):
        data_dir = data_dir.joinpath(subject)
        if simplified:
            files = list(data_dir.glob("simplified.*"))
            if len(files):
                data = load_data(files[0])
            else:
                data = convert_to_simple_tokens(data_dir)
        else:
            data = load_data(list(data_dir.glob("dataset.*"))[0])
        if merged_data is None:
            merged_data = data
        else:
            for data_type in data:
                merged_data[data_type].extend(data[data_type])
    vocabs_dir = root_dir.joinpath("vocabs").joinpath(f"{subject}_simplified" if simplified else subject)
    vocabs_dir.mkdir(parents=True, exist_ok=True)
    vocabs = {data_type: create_vocab(merged_data, data_type, vocabs_dir) for data_type in merged_data}
    return vocabs


def convert_to_simple_tokens(dataset_dir: Path, processor_factory=PlaceholderProcessor):
    processor = processor_factory(dataset_dir.name)
    data = pickle_load(dataset_dir.joinpath("dataset.pickle"))
    assert data is not None, f"Dataset file not found at {dataset_dir.joinpath('dataset.pickle').as_posix()}"
    for data_type in data:
        data[data_type] = [processor.generalize_placeholders(data[data_type][i]) for i in range(len(data[data_type]))]
    pickle_dump(data, dataset_dir.joinpath("simplified.pickle"))
    return data


def convert_tokens_to_tensor(tokens: list, vocabulary: Vocab) -> torch.Tensor:
    return torch.cat([torch.tensor([SPECIAL_IDX["<SOS>"]], dtype=torch.long),
                      torch.tensor([vocabulary[t] for t in tokens], dtype=torch.long),
                      torch.tensor([SPECIAL_IDX["<EOS>"]], dtype=torch.long)], dim=0)


def __create_batch__(data_entry):
    source_data, target_data = zip(*data_entry)
    source_batch = torch.nn.utils.rnn.pad_sequence(source_data, padding_value=SPECIAL_IDX["<PAD>"])
    target_batch = torch.nn.utils.rnn.pad_sequence(target_data, padding_value=SPECIAL_IDX["<PAD>"])
    return source_batch, target_batch


class TorchDataset(Dataset):
    def __init__(self, data_filepath: str | Path | tuple, source: str, target: str,
                 vocabularies: dict[str, Vocab] | None, sample_count: int | float = 0,
                 tokenizers_mapping: dict[str, Type[AbstractTokenizer]] = TOKENIZERS_MAPPING):

        assert source != target, f"Source and Target can't be equal"
        if isinstance(data_filepath, tuple):
            data = None
            for filepath in [Path(filepath).resolve() if isinstance(filepath, str) else filepath.resolve() for filepath
                             in
                             data_filepath]:
                if data is None:
                    data = load_data(filepath)
                else:
                    loaded = load_data(filepath)
                    if loaded is not None:
                        data[source].extend(loaded[source])
                        data[target].extend(loaded[target])
        else:
            data_filepath = Path(data_filepath).resolve() if isinstance(data_filepath, str) else data_filepath.resolve()
            data = load_data(data_filepath)
        if data is None:
            if isinstance(data_filepath, tuple):
                for filepath in [filepath for filepath in data_filepath if "simplified" in filepath.name]:
                    filepath = Path(filepath).resolve() if isinstance(filepath, str) else filepath.resolve()
                    converted = convert_to_simple_tokens(filepath.parent)
                    if data is None:
                        data = converted
                    else:
                        data[source].extend(converted[source])
                        data[target].extend(converted[target])
            else:
                if "simplified" in data_filepath.name:
                    data = convert_to_simple_tokens(data_filepath.parent)
                else:
                    raise FileNotFoundError(f"Dataset file not found at {data_filepath.as_posix()}")
        if sample_count:
            if isinstance(sample_count, int):
                data = {data_type: data[data_type][:sample_count] for data_type in data}
            elif isinstance(sample_count, float):
                data = {data_type: data[data_type][:int(sample_count * len(data[data_type]))] for data_type in data}
            else:
                raise ValueError(f"Unsupported sample_count type: {type(sample_count)}")
        if not isinstance(data[source][0], list):
            tokenization_policy = int("simplified" not in data_filepath.name)
            tokenizers = {data_type: tokenizers_mapping[data_type]() for data_type in data if data_type in tokenizers_mapping}
            for data_type in tokenizers:
                tokenizers[data_type].set_tokenization_policy(tokenization_policy)
            if "latex" in [source, target]:
                tokenizers["latex"] = tokenizers_mapping["latex_expression"]() if "expression" in [source, target] else tokenizers_mapping["latex_mathml"]()
            assert source in tokenizers, f"No tokenizer defined for Source Data Type: {source}"
            assert target in tokenizers, f"No tokenizer defined for Target Data Type: {target}"
            data = {data_type: [tokenizers[data_type].feed(data[data_type][i]) for i in range(len(data[source]))] for data_type in tokenizers}

        assert source in data, f"Source column {source} not found in dataset"
        assert target in data, f"Target column {target} not found in dataset"
        assert len(data[source]) == len(data[target]), "Source and Target sequences must have the same length"

        if vocabularies is None:
            vocabularies = {data_type: create_vocab(data, data_type, data_filepath.parent) for data_type in data}

        data = {data_type: [convert_tokens_to_tensor(entry, vocabularies[data_type])
                            for entry in data[data_type]] for data_type in [source, target]}

        self.data = data
        self.source = source
        self.target = target
        self.vocabs = {data_type: vocabularies[data_type] for data_type in [source, target]}

    def get_vocabularies(self) -> tuple[Vocab, Vocab]:
        return self.vocabs[self.source], self.vocabs[self.target]

    def __len__(self):
        return len(self.data[self.source])

    def __getitem__(self, index):
        return self.data[self.source][index], self.data[self.target][index]

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = False, pin_memory: bool = True) -> DataLoader:
        return DataLoader(self, pin_memory=pin_memory, shuffle=shuffle, collate_fn=__create_batch__, batch_size=batch_size)


class TrainDataset(TorchDataset):
    def __init__(self, data_filepath: str | Path | tuple, source: str, target: str,
                 vocabularies: dict[str, Vocab] | None = None, sample_count: int | float = 0,
                 tokenizers_mapping: dict[str, Type[AbstractTokenizer]] = TOKENIZERS_MAPPING):
        super(TrainDataset, self).__init__(data_filepath, source, target, vocabularies, sample_count, tokenizers_mapping)
        for key in self.data.keys():
            if key not in [source, target]:
                del self.data[key]
                del self.vocabs[key]

    def get_dataloaders(self, train_fraction: float, test_fraction: float, batch_size: int = 1, shuffle: bool = False,
                        pin_memory: bool = True) -> tuple[DataLoader, DataLoader, DataLoader]:
        assert 0.0 <= test_fraction < 1.0, "Test Fraction must be between 0.0 <= Test Fraction < 1.0"
        assert 0.0 < train_fraction < 1.0, "Train Fraction must be between 0.0 < Train Fraction < 1.0"
        test_size = int(len(self) * test_fraction)
        train_size = int((len(self) - test_size) * train_fraction)
        assert train_size > 0.0, "Train Fraction is too small"
        valid_size = len(self) - train_size - test_size
        assert valid_size > 0.0, "Test Fraction is too small"
        train_data, valid_data, test_data = random_split(self, [train_size, valid_size, test_size],
                                                         generator=torch.Generator().manual_seed(SEED))
        train_loader = DataLoader(train_data, pin_memory=pin_memory, shuffle=shuffle, collate_fn=__create_batch__, batch_size=batch_size)
        valid_loader = DataLoader(valid_data, pin_memory=pin_memory, shuffle=shuffle, collate_fn=__create_batch__, batch_size=batch_size)
        test_loader = DataLoader(test_data, pin_memory=pin_memory, shuffle=shuffle, collate_fn=__create_batch__, batch_size=batch_size) if test_size > 0 else None
        return train_loader, valid_loader, test_loader


class RetrainingDataset(Dataset):
    def __init__(self, data: list[tuple[list[str], list[str]]], source_vocab: Vocab, target_vocab: Vocab):
        self.data = [(convert_tokens_to_tensor(src, source_vocab), convert_tokens_to_tensor(trg, target_vocab)) for src, trg in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = False, pin_memory: bool = True) -> DataLoader:
        return DataLoader(self, pin_memory=pin_memory, shuffle=shuffle, collate_fn=__create_batch__, batch_size=batch_size)
