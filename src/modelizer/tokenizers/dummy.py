from typing import Optional
from string import Template

from pandas import DataFrame
from transformers import AutoTokenizer


class DummyTokenizer:
    """
    A mock tokenizer for estimating vocabulary size and maximum sequence length.
    Do not use this tokenizer for any other purpose than estimating vocabulary size and maximum sequence length.
    """
    def __init__(self, name: str = "microsoft/codebert-base",
                 *,
                 hf_token: Optional[str] = None,
                 max_length: int = 250000,
                 silent: bool = False):
        """
        Attention: Do not use this tokenizer for any other purpose than estimating vocabulary size and maximum sequence length.
        :param name: checkpoint name from HuggingFace Hub or local filepath of the tokenizer
        :param hf_token: optional Huggingface API token to access private models. Default is None.
        :param max_length: optional maximum sequence length for the tokenizer, by default 250000 tokens
        """
        self.tokenize = self.__call__
        self._tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=hf_token, clean_up_tokenization_spaces=False)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._true_max_length = self._tokenizer.model_max_length
        self._tokenizer.model_max_length = self.max_length = max_length
        self.tokenize = self.__call__
        if not silent:
            print("Attention: Use DummyTokenizer only for getting the estimation of vocabulary size and maximum sequence length.")

    def __call__(self, text: str | list[str], **kwargs):
        return self._tokenizer(text, return_tensors=None, padding=False, truncation=False, **kwargs)

    def reset_max_length(self):
        """Resets the maximum sequence length to the original value."""
        self._tokenizer.model_max_length = self._true_max_length

    def estimate_max_length(self, text: str | list[str]) -> int:
        """
        Get the estimated maximum sequence length of the given text corpus.
        :param text: text corpus or list of texts
        :return: estimated maximum sequence length as integer
        """
        self._tokenizer.model_max_length = self.max_length
        if isinstance(text, str):
            input_ids = self._tokenizer(text, return_tensors=None, padding=False, truncation=False)["input_ids"]
        else:
            input_ids = [self._tokenizer(entry, return_tensors=None, padding=False, truncation=False)["input_ids"] for entry in text]

        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            lengths = [len(seq) for seq in input_ids]
        else:
            lengths = [len(input_ids)]

        max_length = max(lengths)
        self._tokenizer.model_max_length = self.max_length = max_length
        return max_length

    def estimate_vocab_size(self, text: str | list[str], factor: float | int = 1) -> int:
        """
        Get the estimated vocabulary size of the given text corpus.
        :param text: text corpus or list of texts
        :param factor: additional factor to the estimated vocabulary size, by default 1
        :return: estimated vocabulary size as integer
        """
        assert isinstance(factor, (float, int)), f"factor must be a float or an integer, got {type(factor)} instead."
        assert factor >= 1, f"factor must be greater than or equal to 1, got factor={factor} instead."
        if isinstance(text, str):
            ids = self._tokenizer(text, return_tensors="pt", padding=False, truncation=False)["input_ids"]
            vocab_size = round(ids.unique().numel() * factor)
        else:
            ids_set = set()
            for entry in text:
                ids = self._tokenizer(entry, return_tensors="pt", padding=False, truncation=False)["input_ids"]
                ids_set.update(ids.flatten().tolist())
            vocab_size = round(len(ids_set) * factor)
        return vocab_size

    def estimate_vocab_size2(self, dataframe: DataFrame, source: str, target: str,
                             instructions: Optional[Template] = None, factor: float | int = 1):
        train_data = dataframe[source].tolist() + dataframe[target].tolist()
        if not train_data:
            raise RuntimeError(f"Could not estimate vocabulary size from empty dataframe.")
        if instructions is not None:
            train_data[0] = instructions.safe_substitute({"source": source, "target": target, "input": train_data[0]})
        tokenized = self._tokenizer(train_data, return_tensors="pt", padding=False, truncation=False)["input_ids"]
        return round(tokenized.unique().numel() * factor)
