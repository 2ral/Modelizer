import torch

from time import time
from pathlib import Path
from math import exp as math_exp
from warnings import simplefilter
from abc import ABC, abstractmethod
from random import seed as random_seed
from gc import collect as garbage_collect

from tqdm import tqdm
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader

from modelizer.tokenizer.config import SPECIAL_IDX, SEED
from modelizer.transformer import Transformer as TransformerModel
from modelizer.utils import Logger, AbstractLogger, LoggingLevel, pickle_load, pickle_dump


__BASE_PARAMETERS__ = {
    "source": None,
    "target": None,
    "source_vocab_size": -1,  # initialized dynamically from model constructor
    "target_vocab_size": -1,  # initialized dynamically from model constructor
    "embedding_size": 256,
    "head_count": 8,
    # optimizer parameters #
    "optimizer": "AdamW",
    "learning_policy": None,
    "learning_rate": 0.0005,
    "weight_decay": 0.001,
    "b1": 0.9,
    "b2": 0.98,
    "eps": 1e-9,
    # other parameters #
    "epoch": 0,
    "model_type": None,
    "debug": False,
    "compile_model": False,
    "disable_backend": False,
    "free_cached_memory": False,
}


class AbstractLearner(ABC):

    def __init__(self, dataloaders: tuple[DataLoader, DataLoader, DataLoader | None] | None,
                 vocabularies: tuple[Vocab, Vocab], model, params, state_path: str | Path | None = None, logger: AbstractLogger | None = None):
        assert len(dataloaders) == 3 if dataloaders is not None else True, "DataLoaders must be a tuple of 3 elements (train, valid, test)"
        self.device = initialize_device(disable_backend=params.setdefault("disable_backend", False))
        self.params = params
        self.params["model_type"] = self.__class__
        if params.setdefault("compile_model", False):
            model = model.compile()
        self.model = model.to(self.device)
        self.model.apply(self.initialize_weights)
        self.clip = params.setdefault('clip_gradients', None)
        self.src_vocabulary, self.trg_vocabulary = vocabularies
        self.train_iter, self.valid_iter, self.test_iter = dataloaders if dataloaders is not None else (None, None, None)
        self.params.setdefault("epoch", 0)
        self.debug = params.setdefault("debug", False)
        self.logger = Logger(LoggingLevel.DEBUG if self.debug else LoggingLevel.INFO) if logger is None else logger
        self.free_cached_memory = params.setdefault("free_cached_memory", False)

        match params.setdefault("optimizer", "AdamW"):
            case "SGD":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=params.setdefault("learning_rate", 0.0005), weight_decay=params.setdefault("weight_decay", 0.001))
            case "Adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.setdefault("learning_rate", 0.0005), eps=params.setdefault("eps", 1e-9),
                                                  betas=(params.setdefault("b1", 0.9), params.setdefault("b2", 0.98)))
            case "SparseAdam":
                self.optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=params.setdefault("learning_rate", 0.0005), eps=params.setdefault("eps", 1e-9),
                                                        betas=(params.setdefault("b1", 0.9), params.setdefault("b2", 0.98)))
            case _:  # "AdamW"
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params.setdefault("learning_rate", 0.0005),
                                                   betas=(params.setdefault("b1", 0.9), params.setdefault("b2", 0.98)),
                                                   eps=params.setdefault("eps", 1e-9), weight_decay=params.setdefault("weight_decay", 0.001))

        if state_path is not None:
            self.model.load_state_dict(torch.load(state_path, map_location=self.device))
            self.model.eval()

    def initialize_scheduler(self):
        match self.params.setdefault("learning_policy", None):
            case "linear":
                scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=4)
            case "lambda":
                scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 0.65 ** self.params["epoch"])
            case "multiplicative":
                scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda _: 0.65 ** self.params["epoch"])
            case "step":
                scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=self.params.setdefault("weight_decay", 0.001))
            case "multi_step":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 5, 7], gamma=self.params.setdefault("weight_decay", 0.001))
            case "exponential":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.params.setdefault("weight_decay", 0.001))
            case "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=self.params.setdefault("learning_rate", 0.0005) / 10)
            case "cyclic":
                scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.params.setdefault("learning_rate", 0.0005) / 10, max_lr=self.params.setdefault("learning_rate", 0.0005), step_size_up=2)
            case "cyclic2":
                scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.params.setdefault("learning_rate", 0.0005) / 10, max_lr=self.params.setdefault("learning_rate", 0.0005), step_size_up=2, mode="triangular2")
            case _:
                scheduler = None
        return scheduler

    def train(self, n_epochs: int, path_template: str) -> dict | None:
        if self.train_iter is None:
            self.logger.warning(
                "Training data iterator is not defined. Pass appropriate iterator directly to 'train_epoch' method")
            return None
        elif self.valid_iter is None:
            self.logger.warning(
                "Validation data iterator is not defined. Pass appropriate iterator directly to 'valid_epoch' method")
            return None

        scheduler = self.initialize_scheduler()
        best_valid_loss = float("inf")
        self.logger.info(f"Backend: {self.device}")
        self.logger.info("Model Hyperparameters:")
        for name, value in self.params.items():
            self.logger.info(f"\t{name}: {value}")
        self.logger.info(f'\tTrainable Parameters: {self.count_parameters():,}')
        self.logger.info(f"\tTraining Set size: {len(self.train_iter)}\t"
                         f"Validation Set size: {len(self.valid_iter)}\t"
                         f"Test Set size: {0 if self.test_iter is None else len(self.test_iter)}")

        training_performance = {"train_loss": list(), "valid_loss": list()}
        for epoch in tqdm(range(n_epochs), desc="Training..."):
            start_time = time()
            current_epoch = epoch + 1
            self.params["epoch"] += 1
            train_loss = self.train_epoch(self.train_iter)
            valid_loss = self.valid_epoch(self.valid_iter)
            epoch_time = time() - start_time
            epoch_minutes = int(epoch_time / 60)
            epoch_seconds = int(epoch_time - (epoch_minutes * 60))

            training_performance["train_loss"].append(train_loss)
            training_performance["valid_loss"].append(valid_loss)

            self.logger.info(f'\tEpoch: {current_epoch} | Duration: {epoch_minutes}m {epoch_seconds}s')
            self.logger.info(f'\tTrain Loss: {train_loss:.6f} | '
                             f'Train Perplexity: {math_exp(train_loss):7.6f}')
            self.logger.info(f'\t Val. Loss: {valid_loss:.6f} |  '
                             f'Val. Perplexity: {math_exp(valid_loss):7.6f}')

            if scheduler is not None:
                scheduler.step()

            self.save_state(training_performance, path_template.format("checkpoint", "pth"))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), path_template.format("model", "pth"))
                pickle_dump(self.params, path_template.format("params", "pickle"))

        if self.test_iter is not None:
            self.logger.info("Testing...")
            test_loss = self.valid_epoch(self.test_iter)
            training_performance["test_loss"] = test_loss
            self.logger.info(f'\t Test Loss: {test_loss:.6f}  |  '
                             f'Test Perplexity: {math_exp(test_loss):7.6f}')
        self.logger.info("Training Performance:")
        for name, value in training_performance.items():
            self.logger.info(f"\t{name}: {value}")
        return training_performance

    def fine_tune(self, iterator: DataLoader, output_dir: str | Path | None, n_epochs: int = 1):
        self.logger.info(f"Fine Tuning {'1 pair' if len(iterator) == 1 else str(len(iterator)) + ' pairs'} of samples...")
        start_time = time()
        best_loss = float("inf")
        if isinstance(output_dir, (str, Path)):
            output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
            self.logger.info(f"Fine-tuned model will be save to {output_dir}")
        for epoch in range(n_epochs):
            self.params["epoch"] += 1
            loss = self.train_epoch(iterator)
            if loss < best_loss and output_dir is not None:
                best_loss = loss
                torch.save(self.model.state_dict(), output_dir.joinpath("model_tuned.pth"))
        self.model.eval()
        self.logger.info(f"Fine Tuning completed in {time() - start_time:.2f}")

    @abstractmethod
    def train_epoch(self, iterator: DataLoader) -> float:
        pass

    @abstractmethod
    def valid_epoch(self, iterator: DataLoader) -> float:
        pass

    @abstractmethod
    def translate(self, input_tokens: list[str], max_output_len: int = 0, eos: int = SPECIAL_IDX["<EOS>"]) -> list[str]:
        pass

    @staticmethod
    def initialize_weights(m: torch.nn.Module):
        for p in m.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @staticmethod
    def clear_caches():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        garbage_collect()

    def save_state(self, training_performance: dict | None, filepath: str | Path):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "params": self.params,
            "training_performance": training_performance
        }
        torch.save(checkpoint, filepath)

    def load_state(self, filepath: str | Path) -> dict:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.params = checkpoint["params"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.device = initialize_device(disable_backend=self.params.setdefault("disable_backend", False))
        self.model = self.model.to(self.device)
        self.model.eval()
        return checkpoint["training_performance"]

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class Learner(AbstractLearner):
    HYPERPARAMETERS = __BASE_PARAMETERS__.copy()
    HYPERPARAMETERS.update({
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "feedforward_size": 1024,
        "pos_encoding_size": 5000,
        "dropout": 0.1,
        "clip_gradients": 1.0,
    })

    def __init__(self, dataloaders: tuple[DataLoader, DataLoader, DataLoader | None] | None,
                 vocabularies: tuple[Vocab, Vocab], params: dict, state_path: str | Path | None = None, logger: AbstractLogger | None = None):
        if not params.setdefault("debug", False):
            simplefilter("ignore")

        params["source_vocab_size"] = len(vocabularies[0])
        params["target_vocab_size"] = len(vocabularies[1])
        model = TransformerModel(**params)
        super().__init__(dataloaders, vocabularies, model, params, state_path, logger)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_IDX["<PAD>"])

    def train_epoch(self, iterator: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.
        iterator_len = len(iterator)
        assert iterator_len > 0, "Training iterator is empty"

        for src, trg in tqdm(iterator, total=iterator_len) if self.debug else iterator:
            src, trg = src.to(self.device), trg.to(self.device)
            trg_input = trg[:-1, :]
            src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.__create_mask__(src, trg_input)
            output = self.model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)
            self.optimizer.zero_grad()
            trg_out = trg[1:, :].reshape(-1)
            output = output.reshape(-1, output.shape[-1])
            loss = self.criterion(output, trg_out)
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()

        if self.debug and torch.cuda.is_available():
            self.logger.info(f"Training Epoch"
                             f" | Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB"
                             f" | Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        elif self.debug and torch.backends.mps.is_available():
            self.logger.info(f"Training Epoch"
                             f" | Memory Allocated: {torch.mps.current_allocated_memory() / 1024 ** 3:.2f} GB"
                             f" | Memory Reserved: {torch.mps.driver_allocated_memory() / 1024 ** 3:.2f} GB")

        if self.free_cached_memory:
            self.clear_caches()

        return epoch_loss / iterator_len

    def valid_epoch(self, iterator: DataLoader) -> float:
        self.model.eval()
        epoch_loss = 0.
        iterator_len = len(iterator)
        assert iterator_len > 0, "Validation iterator is empty"

        for src, trg in tqdm(iterator, total=iterator_len) if self.debug else iterator:
            src, trg = src.to(self.device), trg.to(self.device)
            trg_input = trg[:-1, :]
            src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.__create_mask__(src, trg_input)
            output = self.model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)
            trg_out = trg[1:, :].reshape(-1)
            output = output.reshape(-1, output.shape[-1])
            epoch_loss += self.criterion(output, trg_out).item()

        if self.debug and torch.cuda.is_available():
            self.logger.info(f"Validation Epoch"
                             f" | Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB"
                             f" | Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        elif self.debug and torch.backends.mps.is_available():
            self.logger.info(f"Validation Epoch"
                             f" | Memory Allocated: {torch.mps.current_allocated_memory() / 1024 ** 3:.2f} GB"
                             f" | Memory Reserved: {torch.mps.driver_allocated_memory() / 1024 ** 3:.2f} GB")

        if self.free_cached_memory:
            self.clear_caches()

        return epoch_loss / iterator_len

    def __create_mask__(self, src, trg):
        src_mask = torch.zeros((src.shape[0], src.shape[0]), device=self.device).type(torch.bool)
        trg_mask = (torch.triu(torch.ones((trg.shape[0], trg.shape[0]), device=self.device)) == 1).transpose(0, 1)
        trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))

        # noinspection PyUnresolvedReferences
        src_padding_mask = (src == SPECIAL_IDX["<PAD>"]).transpose(0, 1)
        trg_padding_mask = (trg == SPECIAL_IDX["<PAD>"]).transpose(0, 1)
        return src_mask, trg_mask, src_padding_mask, trg_padding_mask

    def __encode_tokens__(self, input_tokens: list[str], max_output_len: int):
        assert max_output_len >= 0, "Max output length must be greater than or equal to 0 (0 means 1.25 length of input)"
        self.model.eval()
        src = torch.cat([torch.tensor([SPECIAL_IDX["<SOS>"]], dtype=torch.long),
                         torch.tensor([self.src_vocabulary[t] for t in input_tokens], dtype=torch.long),
                         torch.tensor([SPECIAL_IDX["<EOS>"]], dtype=torch.long)], dim=0)
        src_mask = torch.zeros((src.shape[0], src.shape[0]), device=self.device).type(torch.bool)
        src = src.view(-1, 1).to(self.device)
        output_shape = max_output_len if max_output_len else round(src.shape[0] * 1.25)
        encoder_outputs = self.model.encode(src, src_mask, self.device)
        return encoder_outputs, output_shape, src

    def translate(self, input_tokens: list[str], max_output_len: int = 0, eos: int = SPECIAL_IDX["<EOS>"]) -> list[str]:
        encoder_outputs, output_shape, src_tokens = self.__encode_tokens__(input_tokens, max_output_len)
        trg_tokens = torch.ones(1, 1).fill_(SPECIAL_IDX["<SOS>"]).type(torch.long).to(self.device)
        for t in range(output_shape):
            next_word = self.model.greedy_decode(trg_tokens, encoder_outputs, self.device)
            trg_tokens = torch.cat([trg_tokens, torch.ones(1, 1).type_as(src_tokens.data).fill_(next_word)], dim=0)
            if next_word == eos:
                break
        return [self.trg_vocabulary.lookup_token(t) for t in trg_tokens.flatten() if t not in SPECIAL_IDX.values()]

    def translate_best_of_k(self, input_tokens: list[str], max_output_len: int = 0, beam_size: int = 1, eos: int = SPECIAL_IDX["<EOS>"]) -> list[str]:
        assert beam_size > 1, "Beam size must be greater than 1. For greedy search use translate method"
        trg_tokens = self.__beam_search__(input_tokens, max_output_len, beam_size, eos)[0][0].flatten()
        return [self.trg_vocabulary.lookup_token(t) for t in trg_tokens if t not in SPECIAL_IDX.values()]

    def translate_top_k(self, input_tokens: list[str], max_output_len: int = 0, beam_size: int = 2, eos: int = SPECIAL_IDX["<EOS>"]) -> list[tuple[list[str], float]]:
        assert beam_size > 1, "Beam size must be greater than 1. For greedy search use translate method"
        trg_tokens = self.__beam_search__(input_tokens, max_output_len, beam_size, eos)
        return [([self.trg_vocabulary.lookup_token(t) for t in token_seq if t not in SPECIAL_IDX.values()], score) for token_seq, score in trg_tokens]

    def __beam_search__(self, input_tokens: list[str], max_output_len: int = 0, beam_size: int = 2, eos: int = SPECIAL_IDX["<EOS>"]):
        predictions = []
        closed_list = []
        encoder_outputs, output_shape, src_tokens = self.__encode_tokens__(input_tokens, max_output_len)
        open_list = [(torch.ones(1, 1).fill_(SPECIAL_IDX["<SOS>"]).type(torch.long).to(self.device), 0.0)]
        while len(open_list):
            trg, s = open_list.pop(0)
            scores, indices = self.model.beam_decode(trg, encoder_outputs, self.device, beam_size)
            for i in range(len(scores[0])):
                next_word = indices[0][i].item()
                new_seq = torch.cat([trg, torch.ones(1, 1).type_as(src_tokens.data).fill_(next_word)], dim=0)
                score = (s + scores[0][i].item()) / new_seq.shape[0]
                if next_word == eos:
                    predictions.append((new_seq, score))
                    break
                else:
                    if len(new_seq) < output_shape:
                        new_seq = new_seq.to(self.device)
                        open_list.append((new_seq, score))
                    else:
                        closed_list.append((new_seq, score))
            candidates = sorted(open_list, key=lambda x: x[1], reverse=True)
            closed_list.extend(candidates[beam_size:])
            open_list = candidates[:beam_size]

        return sorted(predictions, key=lambda x: x[1], reverse=True)[:beam_size] if len(predictions) \
            else sorted(closed_list, key=lambda x: x[1], reverse=True)[:beam_size]


def initialize_device(seed: int = SEED, disable_backend: bool = False) -> torch.device:
    random_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    if disable_backend:
        print("Backend disabled")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision('high')
        print("CUDA backend enabled")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.manual_seed(seed)
        print("MPS backend enabled")
    else:
        print("No supported backend found, using CPU")
    return device


def load_model(model_dir: str | Path, disable_backend: bool = False, load_tuned: bool = False, logger: AbstractLogger | None = None):
    root_dir = Path(model_dir).resolve() if isinstance(model_dir, str) else model_dir
    assert root_dir.exists(), f"Directory {root_dir.as_posix()} does not exist."
    params = pickle_load(root_dir.joinpath("params.pickle"))
    source_vocab = torch.load(root_dir.joinpath(f"{params['source']}_vocab.pth"))
    target_vocab = torch.load(root_dir.joinpath(f"{params['target']}_vocab.pth"))
    params["disable_backend"] = disable_backend
    original_model_filepath = root_dir.joinpath("model.pth")
    tuned_model_filepath = root_dir.joinpath("model_tuned.pth")
    if tuned_model_filepath.exists() and load_tuned:
        print("Loading the tuned model...")
        model_filepath = tuned_model_filepath
    else:
        print("Loading the baseline model...")
        model_filepath = original_model_filepath
    if model_filepath.exists():
        constructor = params.setdefault("model_type", Learner)
        model = constructor(None, (source_vocab, target_vocab), params, model_filepath, logger)
        print(f"Model successfully initialized.")
        return model
    else:
        raise FileNotFoundError(f"Model configuration file not found in {model_filepath.as_posix()}")


def read_model_log(data_dir: str, source: str | None = None, target: str | None = None) -> str:
    root_dir = Path(data_dir).resolve() if "modelizer" in data_dir else Path(data_dir, f"modelizer_{source}_{target}").resolve()
    assert root_dir.exists(), f"Directory {root_dir.as_posix()} does not exist."
    try:
        log_lines = root_dir.joinpath("debug.log").read_text().split("\n")
    except FileNotFoundError:
        log_lines = []
    else:
        for i, text_line in enumerate(log_lines):
            if "Test Set size" in text_line:
                log_lines = log_lines[:i]
                break
    return "\n".join(log_lines)
