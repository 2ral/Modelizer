from pathlib import Path
from typing import Any, Optional
from collections.abc import Sequence, Iterable

from modelizer.dependencies.sklearn import LabelEncoder
from modelizer.tokenizers.abstract import BaseTokenizer, Pickle, Tensor, BiMap


class EncoderTokenizer(BaseTokenizer):
    """Tokenizer class for encoding text data into numerical sequences using LabelEncoder."""
    def __init__(self, path: Optional[str | Path] = None):
        """
        Initializes the EncoderTokenizer class.
        :param path:  The path to the tokenizer. If path is None, the tokenizer should not be saved to disk after training.
        """
        super().__init__(path)
        if self.path is not None and self.path.joinpath("config.pkl").is_file():
            state = Pickle.load(self.path.joinpath("config.pkl"))
            self.name = state["name"]
            self._encoder = state["encoder"]
            self._max_sequence_length = state["max_sequence_length"]
            self._separator = state.get("separator", None)
            self._preprocessors = Pickle.from_bytes(state["preprocessors"])
            self._postprocessors = Pickle.from_bytes(state["postprocessors"])
            saved_vocab = state.get("token_to_id")
            if saved_vocab is not None:
                self._token_to_id = BiMap(saved_vocab)
            else:
                self._token_to_id = BiMap({cls: idx for idx, cls in enumerate(self._encoder.classes_)})
            self.mutations_supported_operations = state["mutations"]["operations"]
            self.mutations_supported_tokens = state["mutations"]["supported"]
            self.mutations_excluded_tokens = state["mutations"]["excluded"]
        else:
            self._encoder = self._separator = None

    @property
    def supports_vocab_stretching(self) -> bool:
        return True

    def stretch_vocabulary(self,
                           tokens: Sequence[Any] | Iterable[Any] | Any,
                           *,
                           save: bool = True) -> dict[Any, int]:
        assert self._encoder is not None and self._token_to_id is not None, "Tokenizer must be trained before stretching vocabulary."
        normalized_tokens = self.normalize_stretch_tokens(tokens)
        new_tokens = [token for token in normalized_tokens if token not in self._token_to_id]
        if not new_tokens:
            return {}

        next_id = max(self._token_to_id.values(), default=-1) + 1
        added: dict[Any, int] = {}
        for token in new_tokens:
            self._token_to_id[token] = next_id
            added[token] = next_id
            next_id += 1

        # Keep sklearn encoder classes in sync for compatibility with persisted state.
        self._encoder.fit(list(self._encoder.classes_) + new_tokens)
        self.__init_mutations_configuration__()
        if save and self.path is not None:
            self.save(self.path)
        return added

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length + 2

    def __getstate__(self):
        state = super().__getstate__()
        state["_encoder"] = Pickle.to_bytes(self._encoder)
        state["_max_sequence_length"] = self._max_sequence_length
        state["_separator"] = self._separator
        return state

    def __setstate__(self, state):
        self._encoder = Pickle.from_bytes(state["_encoder"])
        self._max_sequence_length = state["_max_sequence_length"]
        self._separator = state["_separator"]
        super().__setstate__(state)

    def train(self, data: Sequence[Any] | Iterable[Any], max_input_length: Optional[int] = None,
              separator: Optional[str] = None, *, legacy_padding_mode: bool = False, **_):
        """
        Trains the Encoder tokenizer on the given data.
        :param data: Sequence containing the training data as lists of strings
        :param max_input_length: the maximum input length for the tokenizer, by default is None, which means the maximum length of the training data
        :param separator: Optional separator to use for splitting the input data,
        :param legacy_padding_mode: if True, uses the legacy padding mode, where padding token == eos token.
        """
        if self._encoder is None:
            assert isinstance(data, Sequence | Iterable), f"data must be an Iterable or a Sequence, got {type(data)} instead."
            assert isinstance(max_input_length, int | None), f"max_input_length must be int or None, got {type(max_input_length)}, instead."
            if max_input_length is not None:
                assert max_input_length > 0, f"max_input_length must be greater than 0, got max_input_length={max_input_length} instead."
            if legacy_padding_mode:
                self._pad_token = self._eos_token
            data = list(data)
            self._separator = separator
            self._encoder = LabelEncoder()
            if all(isinstance(entry, str) for entry in data):
                if self._separator is not None:
                    data = [entry.replace(self._separator, " ").replace("  ", " ") for entry in data]
                data = [entry.split() for entry in data]
            self._max_sequence_length = max(len(tokens) for tokens in data) if max_input_length is None else max_input_length
            train_data = self._special_tokens + [t for tokens in data for t in tokens]
            self._encoder.fit(train_data)
            self._token_to_id = BiMap({cls: idx for idx, cls in enumerate(self._encoder.classes_)})
            self.__init_mutations_configuration__()
            self.save(self.path)
        else:
            raise ValueError("Encoder was already trained.")

    def __tokenize__(self, data: str | Any, truncation: bool = True, padding: bool = True, **_) -> dict[str, list]:
        """
        Encodes the input string into integer sequences.
        :param data: a string containing the input data
        :param truncation: if True, truncates the input data to the maximum sequence length
        :param padding: if True, pads the input data to the maximum sequence length
        :return: a dictionary containing the tokenized input data, attention mask.
        """
        assert self._encoder is not None, "Tokenizer must be trained before calling it."
        if len(data) > 0:
            if isinstance(data, str):
                if self._separator is not None:
                    # Split on the user-defined separator first, then on whitespace
                    data = data.replace(self._separator, " ")
                processed = data.replace(self.sep_token, f" {self.sep_token} ").split()
            else:
                processed = data
            processed = [token if token in self._token_to_id else self._unk_token for token in processed]

            if truncation and len(processed) > self._max_sequence_length:
                processed = processed[: self._max_sequence_length]

            processed = [self._bos_token] + processed + [self._eos_token]

            if padding and len(processed) < self.max_sequence_length:
                processed += [self._pad_token] * (self.max_sequence_length - len(processed))

            input_ids = [self.token_to_id(token) for token in processed]
            mask = self.__compute_mask__(input_ids)
        else:
            # Empty input: emit BOS + EOS (+ padding) so the output shape is consistent with non-empty inputs.
            processed = [self._bos_token, self._eos_token]
            if padding:
                processed += [self._pad_token] * (self.max_sequence_length - len(processed))
            input_ids = [self.token_to_id(token) for token in processed]
            mask = self.__compute_mask__(input_ids)
        return {"input_ids": input_ids, "attention_mask": mask}

    def __reconstruct__(self, tokens: list[int] | Tensor) -> str:
        """
        This method reconstructs a string from the tokenized input.
        :param tokens: list of integers or a tensor containing the tokenized input
        :return: reconstructed string
        """
        assert self._token_to_id is not None, "Tokenizer must be trained before calling it."
        tokens = tokens.flatten().tolist() if isinstance(tokens, Tensor) else tokens
        tokens = [token for token in tokens if token not in self.special_token_ids]
        values = [self.id_to_token(token) for token in tokens]
        values = [value for value in values if value is not None]
        values = [str(value) for value in values]
        if self._separator is not None:
            output = self._separator.join(values)
        else:
            output = " ".join(values)
        return output

    def save(self, path: str | Path):
        if path is None:
            print("path argument is None, so tokenizer is not saved to disk.")
        else:
            assert self._encoder is not None, "Tokenizer must be trained before saving it."
            path = Path(path).resolve()
            path.mkdir(parents=True, exist_ok=True)
            state = {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
                "name": self.name,
                "encoder": self._encoder,
                "separator": self._separator,
                "token_to_id": self._token_to_id.forward,
                "max_sequence_length": self._max_sequence_length,
                "preprocessors": Pickle.to_bytes(self._preprocessors),
                "postprocessors": Pickle.to_bytes(self._postprocessors),
                "mutations": {
                    "operations": self.mutations_supported_operations,
                    "supported": self.mutations_supported_tokens,
                    "excluded": self.mutations_excluded_tokens,
                }
            }
            Pickle.dump(state, path / "config.pkl")
