from pathlib import Path
from typing import Any, Optional
from json import loads as json_loads
from collections.abc import Sequence, Iterable

from transformers import PreTrainedTokenizerFast
from tokenizers.implementations.sentencepiece_bpe import SentencePieceBPETokenizer

from modelizer.tokenizers.dummy import DummyTokenizer
from modelizer.tokenizers.abstract import Tensor, BaseTokenizer, Pickle, BiMap


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, path: Optional[str | Path] = None):
        """
        Custom-trained SentencePiece tokenizer.
        :param path: path to the tokenizer directory. If the directory does not exist, the tokenizer needs to be trained.
        """
        super().__init__(path)
        self._tokenizer: Optional[PreTrainedTokenizerFast] = None
        self._sentence_model: Optional[SentencePieceBPETokenizer] = None
        self._added_tokens: list[str] = []

        if path is None:
            return
        config_filepath = Path(path).joinpath("config.pkl").resolve()
        sentence_filepath = Path(path).joinpath("sentencepiece").resolve()
        if not config_filepath.exists() or not sentence_filepath.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            return

        state = Pickle.load(self.path / "config.pkl")
        self.name = state["name"]
        self._special_tokens = state["special_tokens"]
        special_tokens_mapping = state.get("special_tokens_mapping", {})
        for name, token in special_tokens_mapping.items():
            setattr(self, f"_{name}", token)

        self._max_sequence_length = state["max_sequence_length"]
        self._added_tokens = list(state.get("added_tokens", []))
        self._preprocessors = Pickle.from_bytes(state["preprocessors"])
        self._postprocessors = Pickle.from_bytes(state["postprocessors"])

        mutations = state.get("mutations", {})
        self.mutations_supported_operations = mutations.get("operations", self.mutations_supported_operations)
        self.mutations_supported_tokens = mutations.get("supported", self.mutations_supported_tokens)
        self.mutations_excluded_tokens = mutations.get("excluded", self.mutations_excluded_tokens)

        self._sentence_model = self.__load_sentence_model__(sentence_filepath, state)
        self._tokenizer = self.__init_hf_tokenizer_from_spm__(self._sentence_model, self._max_sequence_length)
        if self._added_tokens:
            self._tokenizer.add_tokens(self._added_tokens)
        self._tokenizer.model_max_length = self._max_sequence_length
        self._token_to_id = BiMap(dict(self._tokenizer.get_vocab()))

    def __load_sentence_model_from_artifacts__(self, sentence_filepath: Path) -> SentencePieceBPETokenizer:
        """Best-effort recovery for malformed merges files by keeping only valid merge rules."""
        vocab_path = sentence_filepath.joinpath("vocab.json")
        merges_path = sentence_filepath.joinpath("merges.txt")
        vocab = json_loads(vocab_path.read_text(encoding="utf-8"))
        if not isinstance(vocab, dict) or not vocab:
            raise ValueError("vocab.json is empty or invalid")

        valid_merges: list[tuple[str, str]] = []
        invalid_lines = 0
        merges_lines = merges_path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(merges_lines, start=1):
            if idx == 1 and line.startswith("#version:"):
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                invalid_lines += 1
                continue
            left, right = parts
            if not left or not right:
                invalid_lines += 1
                continue
            if left not in vocab or right not in vocab:
                invalid_lines += 1
                continue
            if f"{left}{right}" not in vocab:
                invalid_lines += 1
                continue
            valid_merges.append((left, right))

        if not valid_merges:
            raise ValueError("No valid merge rules found in merges.txt")

        tokenizer = SentencePieceBPETokenizer(vocab=vocab, merges=valid_merges, unk_token=self._unk_token)
        tokenizer._salvage_info = {"valid_merges": len(valid_merges), "invalid_lines": invalid_lines}  # type: ignore[attr-defined]
        return tokenizer

    def __load_sentence_model__(self, sentence_filepath: Path, state: dict[str, Any]) -> SentencePieceBPETokenizer:
        """Load sentencepiece model with a serialized-state fallback for resiliency."""
        serialized_sentence_model = state.get("sentence_model")
        if serialized_sentence_model is not None:
            try:
                return Pickle.from_bytes(serialized_sentence_model)
            except Exception:
                # Fall back to vocab/merges files if older serialization is incompatible.
                pass

        try:
            return SentencePieceBPETokenizer.from_file(
                vocab_filename=sentence_filepath.joinpath("vocab.json").as_posix(),
                merges_filename=sentence_filepath.joinpath("merges.txt").as_posix(),
                unk_token=self._unk_token,
            )
        except Exception as file_error:
            try:
                return self.__load_sentence_model_from_artifacts__(sentence_filepath)
            except Exception as salvage_error:
                details = f"from_file_error={file_error}; salvage_error={salvage_error}"
                raise RuntimeError(
                    "Failed to load SentencePiece tokenizer artifacts. "
                    f"Expected valid files at '{sentence_filepath}'. "
                    "The merges file may be malformed and could not be salvaged. "
                    f"Details: {details}"
                ) from file_error

    def __getstate__(self):
        state = super().__getstate__()
        state["_max_sequence_length"] = self._max_sequence_length
        state["_sentence_model"] = Pickle.to_bytes(self._sentence_model)
        state["_added_tokens"] = tuple(self._added_tokens)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._max_sequence_length = state["_max_sequence_length"]
        serialized_sp = state.get("_sentence_model")
        if serialized_sp is None:
            vocab_dir = self.path.joinpath("sentencepiece")
            self._sentence_model = self.__load_sentence_model__(vocab_dir, state)
        else:
            self._sentence_model = Pickle.from_bytes(serialized_sp)
        self._tokenizer = self.__init_hf_tokenizer_from_spm__(self._sentence_model, self._max_sequence_length)
        self._added_tokens = list(state.get("_added_tokens", []))
        if self._added_tokens:
            self._tokenizer.add_tokens(self._added_tokens)
        self._token_to_id = BiMap(dict(self._tokenizer.get_vocab()))

    @property
    def supports_vocab_stretching(self) -> bool:
        return True

    def stretch_vocabulary(self,
                           tokens: Sequence[Any] | Iterable[Any] | Any,
                           *,
                           save: bool = True) -> dict[Any, int]:
        assert self._tokenizer is not None and self._token_to_id is not None, "Tokenizer must be trained before stretching vocabulary."

        normalized_tokens = self.normalize_stretch_tokens(tokens)
        normalized_tokens = [str(token) for token in normalized_tokens if str(token).strip()]
        new_tokens = [token for token in normalized_tokens if token not in self._token_to_id]
        if not new_tokens:
            return {}

        self._tokenizer.add_tokens(new_tokens)
        for token in new_tokens:
            if token not in self._added_tokens:
                self._added_tokens.append(token)

        self._token_to_id = BiMap(dict(self._tokenizer.get_vocab()))
        added = {token: self.token_to_id(token) for token in new_tokens if self.token_to_id(token) >= 0}
        self.__init_mutations_configuration__()
        if save and self.path is not None:
            self.save(self.path)
        return added

    def __tokenize__(self, data: str | Any,
                     truncation: bool = True,
                     padding: bool = True, **kwargs) -> dict[str, list]:
        """
        This method tokenizes the input data.
        :param data: data as string or list of strings to be tokenized
        :param truncation: if True, truncates the input data to the maximum sequence length
        :param padding: if True, pads the input data to the maximum sequence length
        :param kwargs: additional keyword arguments
        :return: a dictionary containing the tokenized input data, attention mask, etc.
        """
        assert self._tokenizer is not None and self._token_to_id is not None, "Tokenizer must be trained before calling it."
        kwargs.pop("return_tensors", None)
        # Prepend BOS / EOS manually, then disable automatic special-token insertion so
        # HuggingFace does not add a second copy of them.
        data = f"{self._tokenizer.bos_token}{data}{self._tokenizer.eos_token}"
        return dict(self._tokenizer(data, truncation=truncation, padding="max_length" if padding else False,
                                    max_length=self._max_sequence_length,
                                    add_special_tokens=False,
                                    return_tensors=None, **kwargs))

    def __reconstruct__(self, tokens: list[int] | Tensor) -> str:
        """
        This method reconstructs a string from the tokenized input.
        :param tokens: list of integers or a tensor containing the tokenized input
        :return: reconstructed string
        """
        assert self._tokenizer is not None and self._token_to_id is not None, "Tokenizer must be trained before calling it."
        tokens = tokens.flatten().tolist() if isinstance(tokens, Tensor) else tokens
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def train(self, data: Iterable[Any] | Sequence[Any], vocab_size: Optional[int] = None,
              max_input_length: Optional[int] = None, *, legacy_padding_mode: bool = False, **_):
        """
        Trains the tokenizer on the given data.
        :param data: Sequence containing the training data
        :param vocab_size: Vocabulary size for the tokenizer
        :param max_input_length: Maximum input length for the tokenizer. Default is None.
        :param legacy_padding_mode: if True, uses the legacy padding mode, where padding token == eos token.
        """
        if self._tokenizer is None:
            if legacy_padding_mode:
                self._pad_token = self._eos_token
            assert isinstance(data, Sequence | Iterable), f"data must be iterable or a sequence, got {type(data)} instead."
            data = list(data)

            if vocab_size is None:
                dummy_tokenizer = DummyTokenizer(silent=True)
                vocab_size = dummy_tokenizer.estimate_vocab_size(data)
            else:
                assert isinstance(vocab_size, int) and vocab_size > 0, f"If vocabulary size is provided it must be a positive integer, got vocab_size={vocab_size} instead."

            if max_input_length is None:
                dummy_tokenizer = DummyTokenizer(silent=True)
                max_input_length = dummy_tokenizer.estimate_max_length(data)

            self._sentence_model = SentencePieceBPETokenizer(unk_token=self._unk_token)
            self._sentence_model.train_from_iterator(iter(data), vocab_size=vocab_size, special_tokens=self._special_tokens, min_frequency=2)
            self._tokenizer = self.__init_hf_tokenizer_from_spm__(self._sentence_model, max_input_length)
            self._max_sequence_length = self._tokenizer.model_max_length
            self._token_to_id = BiMap(dict(self._tokenizer.get_vocab()))
            self.__init_mutations_configuration__()
            self.save(self.path)
        else:
            print("Tokenizer was already trained.")

    def __init_hf_tokenizer_from_spm__(self, tokenizer:  SentencePieceBPETokenizer, max_input_length: int):
        temp_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer._tokenizer,
            model_max_length=max_input_length,
            special_tokens=self._special_tokens,
            clean_up_tokenization_spaces=False,
            truncation=True,
            padding=True,
            truncation_side="right"
        )
        temp_tokenizer.add_special_tokens(self.special_tokens_mapping)
        tokens = {
            'bos_token': self._bos_token,
            'eos_token': self._eos_token,
            'pad_token': self._pad_token,
            'unk_token': self._unk_token,
            'cls_token': self._cls_token,
            'sep_token': self._sep_token,
            'mask_token': self._mask_token
        }

        for token_attr, token_value in tokens.items():
            setattr(temp_tokenizer, token_attr, token_value)
            setattr(temp_tokenizer, f"{token_attr}_id", tokenizer.token_to_id(token_value))
        return temp_tokenizer

    def save(self, path: str | Path):
        if path is None:
            print("path argument is None, so tokenizer is not saved to disk.")
        else:
            assert self._tokenizer is not None and self._sentence_model is not None and self._token_to_id is not None, "Tokenizer must be trained before calling it."
            path = Path(path).resolve()
            path.mkdir(parents=True, exist_ok=True)

            # Save tokenizer model files first, then persist config for an all-or-nothing-ish state.
            sentence_model_path = path.joinpath("sentencepiece")
            sentence_model_path.mkdir(parents=True, exist_ok=True)
            self._sentence_model.save_model(sentence_model_path.as_posix())

            state = {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
                "name": self.name,
                "type": self._sentence_model.__class__.__name__,
                "max_sequence_length": self._max_sequence_length,
                "added_tokens": tuple(self._added_tokens),
                "sentence_model": Pickle.to_bytes(self._sentence_model),
                "preprocessors": Pickle.to_bytes(self._preprocessors),
                "postprocessors": Pickle.to_bytes(self._postprocessors),
                "special_tokens_mapping": self.special_tokens_mapping,
                "special_tokens": self._special_tokens,
                "mutations": {
                    "operations": self.mutations_supported_operations,
                    "supported": self.mutations_supported_tokens,
                    "excluded": self.mutations_excluded_tokens,
                },
            }
            Pickle.dump(state, path / "config.pkl")
