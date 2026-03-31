from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from typing import (
    Sequence,
    Iterable,
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)

from random import (
    choice as random_choice,
    randrange as random_randrange,
)

from torch import (
    Tensor,
    tensor,
    dtype,
    long as torch_long,
    device as torch_device,
)

from modelizer.utils import Pickle
from modelizer.tokenizers.shared import BiMap


@dataclass
class TokenOperationStats:
    """Statistics for token operations"""
    token_id: int
    removal_successes: int = 0
    removal_attempts: int = 0
    duplication_successes: int = 0
    duplication_attempts: int = 0
    replacement_successes: int = 0
    replacement_attempts: int = 0
    insertion_successes: int = 0
    insertion_attempts: int = 0

    @property
    def removal_probability(self) -> float:
        return self.removal_successes / self.removal_attempts if self.removal_attempts > 0 else 0.0

    @property
    def duplication_probability(self) -> float:
        return self.duplication_successes / self.duplication_attempts if self.duplication_attempts > 0 else 0.0

    @property
    def replacement_probability(self) -> float:
        return self.replacement_successes / self.replacement_attempts if self.replacement_attempts > 0 else 0.0

    @property
    def insertion_probability(self) -> float:
        return self.insertion_successes / self.insertion_attempts if self.insertion_attempts > 0 else 0.0

    @property
    def overall_mutability_probability(self) -> float:
        """Token is mutable if any operation succeeds"""
        probabilities = [
            self.removal_probability,
            self.duplication_probability,
            self.replacement_probability,
            self.insertion_probability
        ]
        # Token is mutable if at least one operation has non-zero success rate
        return max(probabilities) if any(p > 0 for p in probabilities) else 0.0


@runtime_checkable
class StringProcessor(Protocol):
    def __call__(self, arg: str) -> str: ...


class BaseTokenizer(ABC):
    """This is the base class for all tokenizers."""
    def __init__(self, path: Optional[str | Path]):
        """
        Constructor for the BaseTokenizer class.
        :param path: The path to the tokenizer. If path is None, the tokenizer should not be saved to disk after training.
        """
        assert path is None or isinstance(path, (str, Path)), f"path can be a string or a pathlib.Path object or None, got {type(path)} instead."
        self.path = None if path is None else Path(path).resolve()
        self.name = "Noname" if path is None else self.path.stem
        self.tokenize = self.__call__
        self._torch_dtype: dtype = torch_long
        self._device = torch_device("cpu")
        self._bos_token = "<|sos|>"
        self._eos_token = "<|eos|>"
        self._pad_token = "<|pad|>"
        self._unk_token = "<|unk|>"
        self._cls_token = "<|cls|>"
        self._sep_token = "<|sep|>"
        self._mask_token = "<|mask|>"
        self._special_tokens = [self._bos_token, self._eos_token, self._pad_token, self._unk_token, self._cls_token, self._sep_token, self._mask_token]
        # Initialize the following attributes during training and save their state in save method
        self._max_sequence_length = 0
        self._token_to_id: BiMap[str, int] | None = None
        self._preprocessors: list[StringProcessor] = []
        self._postprocessors: list[StringProcessor] = []
        self.mutations_supported_operations = {
            "removal": [],
            "duplication": [],
            "replacement": [],
            "insertion": []
        }
        self.mutations_supported_tokens: set[int] = set()
        self.mutations_excluded_tokens: set[int] = set()

    @property
    def special_tokens(self) -> list[str]:
        return self._special_tokens

    @property
    def special_token_ids(self) -> set[int]:
        return set(filter(lambda x: x != -1, [self._token_to_id.get(token, -1) for token in self._special_tokens])) if self._token_to_id else set()

    @property
    def bos_token_id(self) -> int:
        return self._token_to_id.get(self._bos_token, -1)

    @property
    def eos_token_id(self) -> int:
        return self._token_to_id.get(self._eos_token, -1)

    @property
    def pad_token_id(self) -> int:
        return self._token_to_id.get(self._pad_token, -1)

    @property
    def unk_token_id(self) -> int:
        return self._token_to_id.get(self._unk_token, -1)

    @property
    def cls_token_id(self) -> int:
        return self._token_to_id.get(self._cls_token, -1)

    @property
    def sep_token_id(self) -> int:
        return self._token_to_id.get(self._sep_token, -1)

    @property
    def mask_token_id(self) -> int:
        return self._token_to_id.get(self._mask_token, -1)

    @property
    def special_tokens_mapping(self) -> dict[str, str]:
        return {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
        }

    @property
    def special_tokens_id_mapping(self) -> dict[str, int]:
        return {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "mask_token_id": self.mask_token_id,
        }

    @staticmethod
    def _validate_string_processor(processor: StringProcessor, kind: str) -> StringProcessor:
        if not callable(processor):
            raise TypeError(f"{kind} must be callable")
        try:
            sample = processor("test")
        except Exception as e:
            raise TypeError(f"{kind} must accept a single string argument") from e
        if not isinstance(sample, str):
            raise TypeError(f"{kind} must return a string")
        return processor

    def add_preprocessor(self, processor: StringProcessor):
        processor = self._validate_string_processor(processor, "preprocessor")
        self._preprocessors.append(processor)

    def add_postprocessor(self, processor: StringProcessor):
        processor = self._validate_string_processor(processor, "postprocessor")
        self._postprocessors.append(processor)

    def __str__(self):
        return f"{self.__class__.__name__} -> {self.__hash__()} | Name: {self.name} | Path: {self.path} | Vocab size: {self.vocab_size}"

    def __getstate__(self):
        """
        Method to serialize the object. Required for pickling. Reimplement in subclass if needed.
        :return: dictionary of object attributes
        """
        return {
            "path": self.path,
            "_max_sequence_length": self._max_sequence_length,
            "_torch_dtype": self._torch_dtype,
            "_token_to_id": self._token_to_id,
            "_bos_token": self._bos_token,
            "_eos_token": self._eos_token,
            "_pad_token": self._pad_token,
            "_unk_token": self._unk_token,
            "_cls_token": self._cls_token,
            "_sep_token": self._sep_token,
            "_mask_token": self._mask_token,
            "_special_tokens": tuple(self._special_tokens),
            "_preprocessors": Pickle.to_bytes(self._preprocessors),
            "_postprocessors": Pickle.to_bytes(self._postprocessors),
            "mutations_supported_operations": self.mutations_supported_operations,
            "mutations_supported_tokens": self.mutations_supported_tokens,
            "mutations_excluded_tokens": self.mutations_excluded_tokens
        }

    def __setstate__(self, state):
        """
        Method to deserialize the object. Required for pickling. Reimplement in subclass if needed.
        :param state: dictionary of object attributes
        """
        self.path = state["path"]
        self._max_sequence_length = state["_max_sequence_length"]
        self._torch_dtype = state["_torch_dtype"]
        self._token_to_id = state["_token_to_id"]
        self._bos_token = state["_bos_token"]
        self._eos_token = state["_eos_token"]
        self._pad_token = state["_pad_token"]
        self._unk_token = state["_unk_token"]
        self._cls_token = state["_cls_token"]
        self._sep_token = state["_sep_token"]
        self._mask_token = state["_mask_token"]
        self._special_tokens = list(state["_special_tokens"])
        self._preprocessors = Pickle.from_bytes(state["_preprocessors"])
        self._postprocessors = Pickle.from_bytes(state["_postprocessors"])
        self.mutations_supported_operations = state["mutations_supported_operations"]
        self.mutations_supported_tokens = state["mutations_supported_tokens"]
        self.mutations_excluded_tokens = state["mutations_excluded_tokens"]

    def __init_mutations_configuration__(self):
        self.mutations_supported_tokens.update(set(self._token_to_id.values()) - self.special_token_ids)
        for operation in self.mutations_supported_operations:
            existing = set(self.mutations_supported_operations[operation])
            new_tokens = self.mutations_supported_tokens - existing
            self.mutations_supported_operations[operation] = list(existing | new_tokens)
        self.mutations_excluded_tokens.update(self.special_token_ids)

    @property
    def data_tokens(self) -> set[str]:
        """
        Returns the set of data tokens (non-special tokens).
        :return: a set of data tokens
        """
        return set(self._token_to_id.keys()) - set(self._special_tokens) if self._token_to_id else set()

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def vocab(self) -> dict[str, int]:
        return self._token_to_id.forward

    @property
    def tokens(self):
        return list(self._token_to_id.keys()) if self._token_to_id else []

    @property
    def token_ids(self):
        return list(self._token_to_id.values()) if self._token_to_id else []

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def torch_dtype(self) -> dtype:
        return self._torch_dtype

    @torch_dtype.setter
    def torch_dtype(self, value: dtype):
        assert isinstance(value, dtype), f"value must be a torch dtype, got {type(value)} instead."
        self._torch_dtype = value

    @property
    def device(self) -> torch_device:
        return self._device

    @device.setter
    def device(self, value: torch_device):
        assert isinstance(value, torch_device), f"device must be a torch.device. Got {type(value)} instead."
        self._device = value

    @property
    def bos_token(self) -> str | None:
        return self._bos_token

    @property
    def eos_token(self) -> str | None:
        return self._eos_token

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @property
    def unk_token(self) -> str | None:
        return self._unk_token

    @property
    def cls_token(self) -> str | None:
        return self._cls_token

    @property
    def sep_token(self) -> str | None:
        return self._sep_token

    @property
    def mask_token(self) -> str | None:
        return self._mask_token

    @property
    def is_trained(self) -> bool:
        return self._token_to_id is not None

    @property
    def supports_vocab_stretching(self) -> bool:
        """Whether tokenizer supports adding new vocabulary entries after training."""
        return False

    @staticmethod
    def normalize_stretch_tokens(tokens: Sequence[Any] | Iterable[Any] | Any) -> list[Any]:
        if isinstance(tokens, (str, bytes)):
            tokens = [tokens]
        elif isinstance(tokens, Sequence | Iterable):
            tokens = list(tokens)
        else:
            tokens = [tokens]

        unique_tokens: list[Any] = []
        seen = set()
        for token in tokens:
            if token is None:
                continue
            token_key = (type(token), token)
            if token_key not in seen:
                seen.add(token_key)
                unique_tokens.append(token)
        return unique_tokens

    def stretch_vocabulary(self,
                           tokens: Sequence[Any] | Iterable[Any] | Any,
                           *,
                           save: bool = True) -> dict[Any, int]:
        """
        Extend tokenizer vocabulary with new tokens.
        Tokenizers with immutable vocabularies should keep the default behavior.
        :param tokens: single token or collection of tokens to add
        :param save: persist tokenizer after stretching when possible
        :return: mapping of newly added token -> token id
        """
        assert self.is_trained, "Tokenizer must be trained before stretching vocabulary."
        raise NotImplementedError(f"{self.__class__.__name__} does not support vocabulary stretching.")

    def _to_tensor(self, data: list[int] | list[list[int]]) -> Tensor:
        return tensor(data, dtype=self._torch_dtype, device=self.device)

    def __call__(self, data: str | Sequence[str | int | float] | Iterable[str | int | float] | Any,
                 truncation: bool = True,
                 padding: bool = True,
                 return_tensors: bool = True,
                 num_mutations: int = 0,
                 **kwargs) -> dict[str, list | Tensor]:
        """
        This method tokenizes the input data.
        :param data: data to be tokenized
        :param truncation: if True, truncates the input data to the maximum sequence length
        :param padding: if True, pads the input data to the maximum sequence length
        :param return_tensors: if True, returns the tokenized data as tensor
        :param num_mutations: number of mutations to apply to the tokenized data. If > 0, the tokens will be mutated. By default, is 0
                              To control which tokens can be injected update the `mutation_injectable_tokens` attribute with a list of token ids.
                              To control which tokens should not be mutated, update the `mutations_excluded_tokens` attribute with a set of token ids.
        :param kwargs: additional keyword arguments
        :return: a dictionary containing the tokenized input data, attention mask, etc.
        """
        for preprocessor in self._preprocessors:
            data = preprocessor(data)

        tokenized = self.__tokenize__(data, truncation, padding, **kwargs)
        if num_mutations > 0:
            tokenized["input_ids"] = self.mutate_tokens(tokenized["input_ids"], num_mutations)
            tokenized["attention_mask"] = self.__compute_mask__(tokenized["input_ids"])
        return {k: self._to_tensor(v) for k, v in tokenized.items()} if return_tensors else tokenized

    def tokenize_no_specials(self, data: str | Sequence[str | int | float] | Iterable[str | int | float] | Any,
                             truncation: bool = False,
                             num_mutations: int = 0,
                             to_string_tokens: bool = False,
                             **kwargs) -> list[int] | list[str]:
        """
        This method tokenizes the input data and filters out special tokens.
        :param data: data to be tokenized
        :param truncation: if True, truncates the input data to the maximum sequence length. By default, is False.
        :param num_mutations: number of mutations to apply to the tokenized data. If > 0, the tokens will be mutated. By default, is 0.
                              To control which tokens can be injected update the `mutation_injectable_tokens` attribute with a list of token ids.
                              To control which tokens should not be mutated, update the `mutations_excluded_tokens` attribute with a set of token ids.
        :param to_string_tokens: if True, converts the token ids back to string tokens. By default, is False.
        :param kwargs: additional keyword arguments for the tokenization process
        :return: a list of token ids or string tokens without special tokens
        """
        input_ids = self.__call__(data, truncation=truncation, padding=False, return_tensors=False, num_mutations=num_mutations, **kwargs)["input_ids"]
        filtered = [token_id for token_id in input_ids if token_id not in self.special_token_ids]
        if to_string_tokens:
            converted = [self.id_to_token(token_id) for token_id in filtered]
            filtered = [token for token in converted if token is not None]
        return filtered

    def mutate_tokens(self, tokens: list[int], num_mutations: int) -> list[int]:
        """
        This method mutates the given tokens by randomly deleting, inserting, or replacing tokens.
        It uses the `mutation_injectable_tokens` to select tokens for insertion or replacement,
        and `mutations_excluded_tokens` to avoid mutating certain tokens.
        If `num_mutations` is 0 or less, the tokens are returned unchanged.
        :param tokens: sequence of token ids to mutate
        :param num_mutations: number of mutations to apply to the tokens.
        :return: mutated token sequence
        """
        if not self.mutations_supported_tokens:
            raise ValueError("No mutable token ids available for mutation. Ensure that the tokenizer has mutable tokens defined.")
        if num_mutations <= 0:
            return tokens

        # Build O(1) lookup sets once per call to avoid O(n) list membership tests in the hot loop.
        removal_set = set(self.mutations_supported_operations["removal"])
        insertion_pool = self.mutations_supported_operations["insertion"]
        replacement_pool = self.mutations_supported_operations["replacement"]
        replacement_set = set(replacement_pool)
        duplication_set = set(self.mutations_supported_operations["duplication"])

        for _ in range(num_mutations):
            if len(tokens) > 2:  # Ensure there are enough tokens to mutate
                idx = random_randrange(1, len(tokens) - 1)
                match random_randrange(4):
                    case 0:  # Delete a token
                        if tokens[idx] not in self.mutations_excluded_tokens and tokens[idx] in removal_set:
                            tokens.pop(idx)
                    case 1:  # Insert a random token
                        if insertion_pool:
                            tokens.insert(idx, random_choice(insertion_pool))
                    case 2:  # Replace current token with a random token from the replacement pool
                        if tokens[idx] not in self.mutations_excluded_tokens and tokens[idx] in replacement_set:
                            tokens[idx] = random_choice(replacement_pool)
                    case 3:  # Duplicate a token
                        if tokens[idx] not in self.mutations_excluded_tokens and tokens[idx] in duplication_set:
                            tokens.insert(idx, tokens[idx])
        return tokens

    @abstractmethod
    def __tokenize__(self, data: str | Sequence[str | int | float] | Iterable[str | int | float] | Any,
                     truncation: bool = True, padding: bool = True, **kwargs) -> dict[str, list]:
        """
        This method implements the tokenization logic
        :param data: data to be tokenized
        :param truncation: if True, truncates the input data to the maximum sequence length
        :param padding: if True, pads the input data to the maximum sequence length
        :param kwargs: additional keyword arguments
        :return: a dictionary containing the tokenized input data, attention mask, etc.
        """
        raise NotImplementedError("__tokenize__ method not implemented in the subclass")

    def __compute_mask__(self, input_ids: list[int]) -> list[int]:
        """Helper method to compute attention mask from input ids. 0 if pad token, 1 otherwise. Can be overridden in subclass."""
        return [int(idx != self.pad_token_id) for idx in input_ids]

    @abstractmethod
    def train(self, *args, **kwargs):
        """This method trains the tokenizer. The exact signature and behavior of this method is up to the implementer."""
        raise NotImplementedError("train method not implemented in the subclass")

    def reconstruct(self, tokens: list[int] | Tensor) -> Any | list:
        """
        This method reconstructs data from the tokenized input.
        :param tokens: list of integers or a tensor containing the tokenized input
        :return: reconstructed data
        """
        assert self.is_trained, "Tokenizer must be trained before calling reconstruct."
        if not isinstance(tokens, Tensor) and all(isinstance(t, str) for t in tokens):
            tokens = [self.token_to_id(t) for t in tokens]
        result = self.__reconstruct__(tokens)
        for postprocessor in self._postprocessors:
            result = postprocessor(result)
        return result

    @abstractmethod
    def __reconstruct__(self, tokens: list[int] | Tensor) -> Any | list:
        """
        This method implements the reconstruction logic.
        :param tokens: list of integers or a tensor containing the tokenized input
        :return: reconstructed data
        """
        raise NotImplementedError("reconstruct method not implemented in the subclass")

    @abstractmethod
    def save(self, path: str | Path):
        """
        This method saves the tokenizer to the specified path.
        Use modelizer.utils.load_module function to re-initialize the tokenizer from the saved config.
        :param path: The path as string or pathlib.Path object to save the tokenizer to
        """
        raise NotImplementedError("save method not implemented in the subclass")

    def token_to_id(self, token: Any) -> int:
        """
        This method returns the id of the given token.
        :param token: the token to get the id of
        :return: the id of the token. If the token is not found or if tokenizer was not trained, returns -1
        """
        return int(self._token_to_id.get(token, -1)) if self._token_to_id else self.unk_token_id

    def id_to_token(self, token_id: int) -> Any | None:
        """
        This method returns the token corresponding to the given id.
        :param token_id: the id to get the token of
        :return: the token corresponding to the id. If the id is not found or if tokenizer was not trained, returns None
        """
        return self._token_to_id.get_key(token_id, None) if self._token_to_id else None

    def update_mutable_tokens(self, tokens: Iterable[TokenOperationStats] | Sequence[TokenOperationStats], success_rate: float = 0.5):
        tokens = list(tokens)
        excluded_tokens = []
        assert all(isinstance(t, TokenOperationStats) for t in tokens), "All elements in tokens must be instances of TokenOperationStats"
        for t in tokens:
            if t.overall_mutability_probability >= success_rate and t.token_id not in self.mutations_excluded_tokens:
                self.mutations_supported_tokens.add(t.token_id)
                # Use set-based deduplication to avoid growing the lists with duplicate ids
                for op, prob in (
                    ("duplication", t.duplication_probability),
                    ("removal", t.removal_probability),
                    ("replacement", t.replacement_probability),
                    ("insertion", t.insertion_probability),
                ):
                    if prob >= success_rate and t.token_id not in self.mutations_supported_operations[op]:
                        self.mutations_supported_operations[op].append(t.token_id)
            elif t.token_id in self.mutations_supported_tokens and t.overall_mutability_probability < success_rate:
                self.mutations_supported_tokens.remove(t.token_id)
                excluded_tokens.append(t.token_id)
                for op in ("duplication", "removal", "replacement", "insertion"):
                    op_list = self.mutations_supported_operations[op]
                    # Remove all occurrences (list may have duplicates from earlier appends)
                    self.mutations_supported_operations[op] = [x for x in op_list if x != t.token_id]
            else:
                excluded_tokens.append(t.token_id)
        self.mutations_excluded_tokens.update(excluded_tokens)
