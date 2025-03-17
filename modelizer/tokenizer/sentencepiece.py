import tempfile
import sentencepiece as spm

from pathlib import Path
from modelizer.tokenizer.generic import AbstractTokenizer


class SentencePieceTokenizer(AbstractTokenizer):
    def __init__(self, model_path: str | Path, grammar: dict | None = None,
                 placeholders: list[str] | list[str] | tuple[str] = ()):
        super().__init__(grammar, placeholders)
        if isinstance(model_path, str):
            model_path = Path(model_path).resolve()
        assert model_path.is_file(), "Model path must point to a file"
        self.__tokenizer__ = spm.SentencePieceProcessor()
        self.__tokenizer__.load(model_path.as_posix())

    def feed(self, data):
        self.buffer = self.__tokenizer__.encode(data, out_type=str)

    def tokenize(self, data) -> list[str]:
        return self.__tokenizer__.encode(data, out_type=str)

    def reconstruct(self, tokens: list[str]) -> str:
        return self.__tokenizer__.decode(tokens)

    @staticmethod
    def train(data: str | list[str], name: str, vocab_size: int, model_type: str = "unigram",
              character_coverage: float = 1.0):
        if isinstance(data, list):
            data = "\n".join(data)
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                tmp.write(data)
                data_path = tmp.name
        elif Path(data).is_file():
            data_path = data
        else:
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                tmp.write(data)
                data_path = tmp.name

        spm.SentencePieceTrainer.train(
            input=data_path,
            model_prefix=name,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            bos_id=-1,
            eos_id=-1
        )
        if data_path != data:
            Path(data_path).unlink()
        print(f"Tokenizer trained. Files: {name}.model, {name}.vocab")
        return f"{name}.model"
