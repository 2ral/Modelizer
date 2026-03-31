# This file implements the legacy model architecture from Modelizer v1.
# based on https://pytorch.org/tutorials/beginner/translation_transformer.html and adapted for Modelizer
import torch

from pathlib import Path
from copy import deepcopy
from threading import Event
from datetime import datetime
from typing import Any, Optional
from math import exp as math_exp
from concurrent.futures import CancelledError

from tqdm.auto import tqdm
from pandas import DataFrame

from modelizer import configs
from modelizer.tokenizers.abstract import BaseTokenizer
from modelizer.models.abstract import BaseConfig, BaseModel
from modelizer.models.legacy.transformer import Transformer
from modelizer.models.legacy.dataset import TorchDataset, DataLoader
from modelizer.utils import Logger, MemInfo, Pickle, TorchHelpers, load_module, get_time_diff


class LegacyConfig(BaseConfig):
    def __init__(self,
                 output_dir: str | Path,
                 source: str,
                 target: str,
                 backward: bool,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 embedding_size: int,
                 num_heads: int,
                 feedforward_size: int = 0,
                 enc_layers: int = 1,
                 dec_layers: int = 1,
                 dropout: float = 0.1,
                 source_max_len: int = 5000,
                 target_max_len: int = 5000,
                 optimizer: str = configs.OPTIMIZER,
                 scheduler: Optional[str] = configs.SCHEDULER,
                 learning_rate: float = configs.LEARNING_RATE,
                 weight_decay: float = configs.WEIGHT_DECAY,
                 b1: float = configs.B1,
                 b2: float = configs.B2,
                 eps: float = configs.EPS,
                 clip_grad: float = configs.CLIP_GRAD,
                 compile_model: bool = False,
                 force_cpu: bool = False,
                 total_save_limit: int = configs.TOTAL_SAVE_LIMIT,
                 wandb_token: Optional[str] = None,
                 seed: int = configs.SEED,
                 validation_fraction: Optional[float] = configs.VALIDATION_FRACTION,
                 shuffle_train_data: bool = True,
                 pin_dataloader_memory: bool = False,
                 free_cached_memory: bool = False,
                 use_distributed_sampler: bool = False,
                 reduce_spaces: bool = False,
                 metadata: Optional[dict[str, Any]] = None, **_):
        """
        Constructor for the LegacyConfig class.

        :param output_dir: The directory to save the model to.
        :param source: The source language or column name.
        :param target: The target language or column name.
        :param backward: If True, the model will be trained in the backward direction.
        :param source_vocab_size: Vocabulary size for the source tokenizer.
        :param target_vocab_size: Vocabulary size for the target tokenizer.
        :param embedding_size: Size of the embedding vectors.
        :param num_heads: Number of attention heads in the transformer.
        :param feedforward_size: Size of the feedforward network in the transformer.
        :param enc_layers: Number of encoder layers.
        :param dec_layers: Number of decoder layers.
        :param dropout: Dropout rate for the transformer.
        :param source_max_len: Maximum sequence length for the source.
        :param target_max_len: Maximum sequence length for the target.
        :param optimizer: Optimizer type to use during training.
        :param scheduler: Learning rate scheduler type.
        :param learning_rate: Initial learning rate.
        :param weight_decay: Weight decay (L2 penalty).
        :param b1: Beta1 parameter for Adam-based optimizers.
        :param b2: Beta2 parameter for Adam-based optimizers.
        :param eps: Epsilon value for optimizers.
        :param clip_grad: Maximum norm for gradient clipping.
        :param compile_model: If True, use torch.compile for model.
        :param force_cpu: If True, force the model to run on CPU.
        :param total_save_limit: The maximum number of model checkpoints to keep.
        :param wandb_token: (Optional) The Weights and Biases API token.
        :param seed: The seed value for reproducibility.
        :param validation_fraction: Fraction of data to use for validation (rest for Training). Could be None to use all data for training and validation. Default is configs.VALIDATION_FRACTION.
        :param shuffle_train_data: If True, shuffle training data.
        :param pin_dataloader_memory: If True, pin memory in DataLoader.
        :param free_cached_memory: If True, free cached memory after each epoch.
        :param use_distributed_sampler: If True, use distributed sampler in DataLoader.
        :param reduce_spaces: If True, the model will reduce spaces in the input data.
        :param metadata: Optional metadata dictionary to store additional information about the model.
        """
        super().__init__(output_dir, source, target, backward, False, validation_fraction, seed,
                         wandb_token, force_cpu, total_save_limit, free_cached_memory, reduce_spaces, metadata)
        self._source_vocab_size = source_vocab_size
        self._target_vocab_size = target_vocab_size
        self._embedding_size = embedding_size
        self._num_heads = num_heads
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._b1 = b1
        self._b2 = b2
        self._eps = eps
        self._clip_grad = clip_grad
        self._compile_model = compile_model
        self._feedforward_size = feedforward_size if feedforward_size > 0 else embedding_size
        self._enc_layers = enc_layers
        self._dec_layers = dec_layers
        self._dropout = dropout
        self._source_max_len = source_max_len
        self._target_max_len = target_max_len

        self._use_distributed_sampler = use_distributed_sampler
        self._shuffle_train_data = shuffle_train_data
        self._pin_dataloader_memory = pin_dataloader_memory

        self._model_state = None
        self._optimizer_state = None
        self._tokenizer_path = None
        self._tokenizer2_path = None
        self._report_epoch_progress = False

    def check_constraints(self):
        if self._feedforward_size == 0:
            self._feedforward_size = self._embedding_size * 2

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value

    @property
    def optimizer_state(self):
        return self._optimizer_state

    @optimizer_state.setter
    def optimizer_state(self, value):
        self._optimizer_state = value

    @property
    def tokenizer_path(self) -> str | None:
        return self._tokenizer_path

    @tokenizer_path.setter
    def tokenizer_path(self, value: str | Path | None):
        assert value is None or isinstance(value, str | Path), "Tokenizer path can be None or a string or pathlib.Path object"
        self._tokenizer_path = value

    @property
    def tokenizer2_path(self):
        return self._tokenizer2_path

    @tokenizer2_path.setter
    def tokenizer2_path(self, value):
        assert value is None or isinstance(value, str | Path), "Tokenizer path can be None or a string or pathlib.Path object"
        self._tokenizer2_path = value
        
    @property
    def source_vocab_size(self) -> int:
        return self._source_vocab_size
    
    @property
    def target_vocab_size(self) -> int:
        return self._target_vocab_size
    
    @property
    def embedding_size(self) -> int:
        return self._embedding_size
    
    @property
    def num_heads(self) -> int:
        return self._num_heads
    
    @property
    def optimizer(self) -> str:
        return self._optimizer
    
    @property
    def scheduler(self) -> str | None:
        return self._scheduler
    
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @property
    def weight_decay(self) -> float:
        return self._weight_decay
    
    @property
    def b1(self) -> float:
        return self._b1
    
    @property
    def b2(self) -> float:
        return self._b2
    
    @property
    def eps(self) -> float:
        return self._eps
    
    @property
    def clip_grad(self) -> float:
        return self._clip_grad
    
    @property
    def compile_model(self) -> bool:
        return self._compile_model
    
    @property
    def feedforward_size(self) -> int:
        return self._feedforward_size
    
    @property
    def enc_layers(self) -> int:
        return self._enc_layers
    
    @property
    def dec_layers(self) -> int:
        return self._dec_layers
    
    @property
    def dropout(self) -> float:
        return self._dropout
    
    @property
    def source_max_len(self) -> int:
        return self._source_max_len

    @property
    def target_max_len(self) -> int:
        return self._target_max_len

    @property
    def shuffle_train_data(self) -> bool:
        return self._shuffle_train_data

    @shuffle_train_data.setter
    def shuffle_train_data(self, value: bool):
        assert isinstance(value, bool), "shuffle_train_data must be a boolean"
        self._shuffle_train_data = value

    @property
    def use_distributed_sampler(self) -> bool:
        return self._use_distributed_sampler

    @use_distributed_sampler.setter
    def use_distributed_sampler(self, value: bool):
        assert isinstance(value, bool), "use_distributed_sampler must be a boolean"
        self._use_distributed_sampler = value

    @property
    def free_cached_memory(self) -> bool:
        return self._free_cached_memory

    @free_cached_memory.setter
    def free_cached_memory(self, value: bool):
        assert isinstance(value, bool), "free_cached_memory must be a boolean"
        self._free_cached_memory = value

    @property
    def pin_dataloader_memory(self) -> bool:
        return self._pin_dataloader_memory

    @property
    def report_epoch_progress(self) -> bool:
        return self._report_epoch_progress

    @report_epoch_progress.setter
    def report_epoch_progress(self, value: bool):
        assert isinstance(value, bool), "report_epoch_progress must be a boolean"
        self._report_epoch_progress = value

    def get_configurable_parameters(self, force_cpu: bool = False) -> dict[str, list[Any]]:
        config = deepcopy(configs.LEGACY_MODEL_PARAMETERS)
        if torch.cuda.is_available() and not force_cpu and not self.cross_platform_compatibility:
            config["optimizer"].extend([f"{opt}_8bit" for opt in config["optimizer"]])
        return config


class LegacyModel(BaseModel):
    def __init__(self, config: LegacyConfig, tokenizer: BaseTokenizer,
                 output_tokenizer: Optional[BaseTokenizer] = None, logger: Optional[Logger] = None, force_cpu: bool = False):
        assert tokenizer is not None, "At least input tokenizer must be provided"
        config.check_constraints()
        config.force_cpu = force_cpu
        super().__init__(config, tokenizer, output_tokenizer, logger)
        self.training_performance = {"train_loss": list(), "valid_loss": list()}
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = TorchHelpers.initialize_device(config.seed, logger=self.logger, force_cpu=config.force_cpu)

        self._model = Transformer(
            source_vocab_size=config.source_vocab_size,
            target_vocab_size=config.target_vocab_size,
            enc_layers=config.enc_layers,
            dec_layers=config.dec_layers,
            embedding_size=config.embedding_size,
            feedforward_size=config.feedforward_size,
            head_count=config.num_heads,
            dropout=config.dropout,
            source_max_len=config.source_max_len,
            target_max_len=config.target_max_len,
        )
        self._model.apply(BaseModel.initialize_weights)

        if config.compile_model and self.device.type == "mps":
            # torch.compile is not supported on MPS devices
            config._compile_model = False

        if config.compile_model:
            self._model = torch.compile(self._model)

        if config.model_state is not None:
            self._model.load_state_dict(config.model_state)

        self._model.to(self.device)
        if self._tokenizer is not None:
            self._tokenizer.device = self.device
        if self._output_tokenizer is not None:
            self._output_tokenizer.device = self.device

        match config.optimizer:
            case "sgd":
                # Explicit momentum=0.0 to avoid backends that require the argument
                self.optimizer = torch.optim.SGD(self._model.parameters(),
                                                 lr=config.learning_rate,
                                                 weight_decay=config.weight_decay,
                                                 momentum=0.0)
            case "adam":
                self.optimizer = torch.optim.Adam(self._model.parameters(),
                                                  lr=config.learning_rate,
                                                  weight_decay=config.weight_decay,
                                                  eps=config.eps,
                                                  betas=(config.b1, config.b2))
            case "rmsprop":
                self.optimizer = torch.optim.RMSprop(self._model.parameters(),
                                                     lr=config.learning_rate,
                                                     weight_decay=config.weight_decay,
                                                     eps=config.eps)
            case "adagrad":
                if torch.cuda.is_available() and not self.config.force_cpu:
                    from bitsandbytes import optim as bnb_optimizers
                else:
                    raise ImportError("Adagrad optimizer requires CUDA-enabled pytorch installation.")
                _model_params: Any = list(self._model.parameters())
                self.optimizer = bnb_optimizers.Adagrad(_model_params,
                                                        lr=config.learning_rate,
                                                        weight_decay=config.weight_decay,
                                                        eps=config.eps)
            case "paged_adamw":
                if torch.cuda.is_available() and not self.config.force_cpu:
                    from bitsandbytes import optim as bnb_optimizers
                else:
                    raise ImportError("PagedAdamW optimizer requires CUDA-enabled pytorch installation.")
                _model_params: Any = list(self._model.parameters())
                self.optimizer = bnb_optimizers.PagedAdamW(_model_params,
                                                           lr=config.learning_rate,
                                                           weight_decay=config.weight_decay,
                                                           eps=config.eps,
                                                           betas=(config.b1, config.b2))
            case "sgd_8bit":
                if torch.cuda.is_available() and not self.config.force_cpu:
                    from bitsandbytes import optim as bnb_optimizers
                else:
                    raise ImportError("SGD8bit optimizer requires CUDA-enabled pytorch installation.")
                momentum = config.b1 if config.b1 > 0 else 0.9
                if config.b1 <= 0:
                    self.logger.warning("SGD8bit requires momentum > 0; using momentum=0.9.")
                _model_params: Any = list(self._model.parameters())
                self.optimizer = bnb_optimizers.SGD8bit(_model_params,
                                                        lr=config.learning_rate,
                                                        weight_decay=config.weight_decay,
                                                        momentum=momentum)

            case "rmsprop_8bit":
                if torch.cuda.is_available() and not self.config.force_cpu:
                    from bitsandbytes import optim as bnb_optimizers
                else:
                    raise ImportError("RMSprop8bit optimizer requires CUDA-enabled pytorch installation.")
                _model_params: Any = list(self._model.parameters())
                self.optimizer = bnb_optimizers.RMSprop8bit(_model_params,
                                                            lr=config.learning_rate,
                                                            weight_decay=config.weight_decay,
                                                            eps=config.eps)
            case "adam_8bit":
                if torch.cuda.is_available() and not self.config.force_cpu:
                    from bitsandbytes import optim as bnb_optimizers
                else:
                    raise ImportError("Adam8bit optimizer requires CUDA-enabled pytorch installation.")
                _model_params: Any = list(self._model.parameters())
                self.optimizer = bnb_optimizers.Adam8bit(_model_params,
                                                         lr=config.learning_rate,
                                                         weight_decay=config.weight_decay,
                                                         eps=config.eps,
                                                         betas=(config.b1, config.b2))
            case "adagrad_8bit":
                if torch.cuda.is_available() and not self.config.force_cpu:
                    from bitsandbytes import optim as bnb_optimizers
                else:
                    raise ImportError("Adagrad8bit optimizer requires CUDA-enabled pytorch installation.")
                _model_params: Any = list(self._model.parameters())
                self.optimizer = bnb_optimizers.Adagrad8bit(_model_params,
                                                            lr=config.learning_rate,
                                                            weight_decay=config.weight_decay,
                                                            eps=config.eps)
            case "paged_adamw_8bit":
                if torch.cuda.is_available() and not self.config.force_cpu:
                    from bitsandbytes import optim as bnb_optimizers
                else:
                    raise ImportError("PagedAdamW8bit optimizer requires CUDA-enabled pytorch installation.")
                _model_params: Any = list(self._model.parameters())
                self.optimizer = bnb_optimizers.PagedAdamW8bit(_model_params,
                                                               lr=config.learning_rate,
                                                               weight_decay=config.weight_decay,
                                                               eps=config.eps,
                                                               betas=(config.b1, config.b2))
            case "adamw_8bit":
                if torch.cuda.is_available() and not self.config.force_cpu:
                    from bitsandbytes import optim as bnb_optimizers
                else:
                    raise ImportError("AdamW8bit optimizer requires CUDA-enabled pytorch installation.")
                _model_params: Any = list(self._model.parameters())
                self.optimizer = bnb_optimizers.AdamW8bit(_model_params,
                                                          lr=config.learning_rate,
                                                          weight_decay=config.weight_decay,
                                                          eps=config.eps,
                                                          betas=(config.b1, config.b2))

            case _:  # "By default, AdamW is used"
                self.optimizer = torch.optim.AdamW(self._model.parameters(),
                                                   lr=config.learning_rate,
                                                   weight_decay=config.weight_decay,
                                                   eps=config.eps,
                                                   betas=(config.b1, config.b2))

        if config.optimizer_state is not None:
            self.optimizer.load_state_dict(config.optimizer_state)

        match config.scheduler:
            case "linear":
                self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=4)
            case "lambda":
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 0.65 ** self.config.epoch)
            case "multiplicative":
                self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda _: 0.65 ** self.config.epoch)
            case "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=self.config.weight_decay)
            case "multi_step":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 5, 7], gamma=self.config.weight_decay)
            case "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.weight_decay)
            case "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=self.config.learning_rate / 10)
            case "cyclic":
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.learning_rate / 10, max_lr=self.config.learning_rate, step_size_up=2)
            case "cyclic2":
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.config.learning_rate / 10, max_lr=self.config.learning_rate, step_size_up=2, mode="triangular2")
            case _:
                self.scheduler = None

        self.__optimizer_step__ = self.__optimizer_compiled_step__ if config.compile_model else self.__optimizer_basic_step__
        self._logger.info(f"{self.__class__.__name__} got initialized. Total Parameters: {sum(p.numel() for p in self._model.parameters())}")
        self.config.trainable_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        ignore_index = -100 if self.output_tokenizer is None or self.output_tokenizer.pad_token_id < 0 else self.output_tokenizer.pad_token_id
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.__bos_token_id__ = self._output_tokenizer.token_to_id(self._output_tokenizer.bos_token)
        self.__eos_token_id__ = self._output_tokenizer.token_to_id(self._output_tokenizer.eos_token)
        self._model.eval()

    @torch.compile(fullgraph=False)
    def __optimizer_compiled_step__(self):
        self.optimizer.step()

    def __optimizer_basic_step__(self):
        self.optimizer.step()

    def __create_mask__(self, src, tgt):
        batch_size, src_seq_len = src.shape[0], src.shape[1]
        _, tgt_seq_len = tgt.shape[0], tgt.shape[1]
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device, dtype=torch.bool)
        tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=self.device, dtype=torch.bool), diagonal=1)
        src_padding_mask = (src == self.tokenizer.pad_token_id)
        tgt_padding_mask = (tgt == self.output_tokenizer.pad_token_id)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def train(self, dataframe: DataFrame, num_epochs: int, batch_size: int = 1, *, stop_event: Optional[Event] = None, show_progress: bool = True, **_) -> tuple[float, float]:
        if stop_event is not None and isinstance(stop_event, Event):
            self.__stop_event__ = stop_event
        self.config.batch_size = batch_size
        self.config.report_epoch_progress = show_progress
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            error_msg = "dataframe must be a non-empty Pandas DataFrame object."
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(num_epochs, int) or num_epochs < 1:
            error_msg = "Number of epochs must an integer that is greater than 0."
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(batch_size, int) or batch_size < 1:
            error_msg = "Batch size must be an integer that is greater than 0."
            self._logger.error(error_msg)
            raise ValueError(error_msg)

        self.__check_both_tokenizers__(self.config.tokenizer_path, self.config.tokenizer2_path, dataframe)

        self._logger.info("Preparing DataLoader...")
        dataset = TorchDataset(dataframe, self.config.source, self.config.target, self.tokenizer, self.output_tokenizer)
        train_dataloader, valid_dataloader = dataset.get_dataloaders(self.config.validation_steps_or_fraction, batch_size=batch_size,
                                                                     pin_memory=self.config.pin_dataloader_memory,
                                                                     use_distributed_sampler=self.config.use_distributed_sampler,
                                                                     shuffle=self.config.shuffle_train_data, seed=self.config.seed)
        self._logger.info("Training started...")
        start_time = datetime.now()
        best_train_loss = float("inf")
        best_valid_loss = float("inf")
        epochs_no_improve = 0

        for epoch in tqdm(range(num_epochs), desc="Training...") if show_progress else range(num_epochs):
            e_start_time = datetime.now()
            current_epoch = epoch + 1
            epochs_no_improve += 1
            self.config.epoch += 1

            if self.__stop_event__.is_set():
                raise CancelledError()

            train_loss = self.train_epoch(train_dataloader)

            if self.__stop_event__.is_set():
                raise CancelledError()

            valid_loss = self.valid_epoch(valid_dataloader)
            self.training_performance["train_loss"].append(train_loss)
            self.training_performance["valid_loss"].append(valid_loss)
            self.logger.info(f'\tEpoch: {current_epoch} | Duration: {get_time_diff(e_start_time)}')
            self.logger.info(f'\tTrain Loss: {train_loss:.6f} | Train Perplexity: {math_exp(train_loss):7.6f}')
            self.logger.info(f'\t Val. Loss: {valid_loss:.6f} | Valid Perplexity: {math_exp(valid_loss):7.6f}')

            if self.scheduler is not None:
                self.scheduler.step()

            if self.auto_saves_enabled:
                if valid_loss < best_valid_loss:
                    self.save_model("model.pth")
                    best_valid_loss = valid_loss
                    epochs_no_improve = 0
                elif self.config.total_save_limit > 1:
                    self.save_model(f"checkpoint_{self.checkpoint_id}")
            elif valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0

            if train_loss < best_train_loss:
                best_train_loss = train_loss

            if epochs_no_improve > 2:
                self._logger.info(f"Early stopping triggered at {current_epoch} after {epochs_no_improve} epochs without improvement.")
                break

        self._model.eval()
        self.logger.info(f"Training completed in {get_time_diff(start_time)}")
        self._logger.info(f"Best Train Loss: {best_train_loss} | Best Eval Loss: {best_valid_loss}")
        return best_train_loss, best_valid_loss

    def __get_output__(self, entries):
        src, tgt = entries["source"].to(self.device), entries["target"].to(self.device)
        tgt_input = tgt[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.__create_mask__(src, tgt_input)
        output = self._model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[:, 1:].reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        return output, tgt_out

    def train_epoch(self, iterator: DataLoader) -> float:
        iterator_len = len(iterator)
        assert iterator_len > 0, "Training iterator is empty"
        self._model.train()
        epoch_loss = 0.
        successful_batches = 0

        iterable = tqdm(iterator, total=iterator_len, leave=False,
                        miniters=10) if self.config.report_epoch_progress else iterator

        for entries in iterable:
            try:
                output, tgt = self.__get_output__(entries)
            except Exception as e:
                self._logger.warning(f"Skipping failing training batch due to error: {e!r}")
                continue
            else:
                successful_batches += 1

            self.optimizer.zero_grad()
            loss = self.criterion(output, tgt)
            loss.backward()
            if self.config.clip_grad:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.config.clip_grad)
            self.__optimizer_step__()
            epoch_loss += loss.item()

        if self.config.free_cached_memory:
            MemInfo.clean_cache()

        if successful_batches == 0:
            self._logger.error("All training batches failed in this epoch; returning inf loss.")
            return float("inf")

        self._logger.info(MemInfo.get_memory_usage_stats("Training phase"))
        return epoch_loss / iterator_len

    def valid_epoch(self, iterator: DataLoader) -> float:
        iterator_len = len(iterator)
        assert iterator_len > 0, "Validation iterator is empty"
        self._model.eval()
        epoch_loss = 0.
        successful_batches = 0
        iterable = tqdm(iterator, total=iterator_len, leave=False,
                        miniters=10) if self.config.report_epoch_progress else iterator

        for entries in iterable:
            try:
                with torch.no_grad():
                    output, tgt = self.__get_output__(entries)
            except Exception as e:
                self._logger.warning(f"Skipping failing validation batch due to error: {e!r}")
                continue
            else:
                successful_batches += 1

            epoch_loss += self.criterion(output, tgt).item()

        if self.config.free_cached_memory:
            MemInfo.clean_cache()

        if successful_batches == 0:
            self._logger.error("All validation batches failed in this epoch; returning inf loss.")
            return float("inf")

        self._logger.info(MemInfo.get_memory_usage_stats("Validation phase"))
        return epoch_loss / iterator_len

    @torch.inference_mode()
    def generate(self, input_data: Any, max_length: int = 256, **kwargs) -> Any:
        src_tokens = self._tokenizer(input_data)["input_ids"]
        src_mask = torch.zeros((src_tokens.shape[0], src_tokens.shape[0]), device=self.device).type(torch.bool)
        src_tokens = src_tokens.unsqueeze(0).to(self.device)
        output_shape = max_length if max_length > 0 else self._output_tokenizer.max_sequence_length
        encoder_outputs = self._model.encode(src_tokens, src_mask, self.device)
        max_gen_len = output_shape + 1
        generated = torch.full((1, max_gen_len), self._output_tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        generated[0, 0] = self._output_tokenizer.bos_token_id
        for t in range(1, max_gen_len):
            next_word = self._model.greedy_decode(generated[:, :t], encoder_outputs, self.device, beam_size=1)
            generated[0, t] = next_word
            if next_word == self._output_tokenizer.eos_token_id:
                break

        eos_pos = (generated[0] == self._output_tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            generated = generated[:, :eos_pos[0].item() + 1]
        else:
            generated = generated[:, :max_gen_len]
        return self._output_tokenizer.reconstruct(generated)

    def generate_best_of_k(self, input_data: Any, max_length: int = 256, beam_size: int = 1, eos: Optional[int] = None) -> str:
        assert beam_size > 1, "Beam size must be greater than 1. For greedy search use translate method."
        candidates = self.__beam_search__(input_data, max_length, beam_size, eos)
        if not candidates:
            return ""
        else:
            best_seq, _ = candidates[0]
            return self._output_tokenizer.reconstruct(best_seq)

    def generate_top_k(self, input_data: Any, max_length: int = 256, beam_size: int = 2, eos: Optional[int] = None) -> list[tuple[str, float]]:
        assert beam_size > 1, "Beam size must be greater than 1. For greedy search use translate method"
        candidates = self.__beam_search__(input_data, max_length, beam_size, eos)
        return [(self._output_tokenizer.reconstruct(seq), score) for seq, score in candidates]

    def __beam_search__(self, input_data: Any, max_length: int = 256, beam_size: int = 2, eos: Optional[int] = None):
        if eos is None:
            eos = self._output_tokenizer.eos_token_id

        src_tokens = self._tokenizer(input_data)["input_ids"]
        src_tokens = src_tokens.unsqueeze(0).to(self.device)
        src_mask = torch.zeros((src_tokens.size(1), src_tokens.size(1)), dtype=torch.bool, device=self.device)

        encoder_outputs = self._model.encode(src_tokens, src_mask, self.device)
        output_limit = max_length if max_length > 0 else self._output_tokenizer.max_sequence_length

        bos = torch.full((1, 1), self._output_tokenizer.bos_token_id, dtype=torch.long, device=self.device)
        open_list: list[tuple[torch.Tensor, float]] = [(bos, 0.0)]
        predictions: list[tuple[torch.Tensor, float]] = []
        closed_list: list[tuple[torch.Tensor, float]] = []

        while open_list:
            tgt, cum_score = open_list.pop(0)
            scores, indices = self._model.beam_decode(tgt, encoder_outputs, self.device, beam_size)

            if isinstance(scores, torch.Tensor):
                cand_scores = scores[0].tolist()
                cand_indices = indices[0].tolist()
            else:
                cand_scores = list(scores)
                cand_indices = list(indices)

            for cand_score, next_token in zip(cand_scores, cand_indices):
                next_token = int(next_token)
                token_tensor = torch.full((1, 1), next_token, dtype=tgt.dtype, device=self.device)
                new_seq = torch.cat([tgt, token_tensor], dim=1)  # append along sequence dimension
                new_cum_score = cum_score + float(cand_score)

                if next_token == eos or new_seq.size(1) >= output_limit:
                    predictions.append((new_seq.detach().clone(), new_cum_score))
                else:
                    open_list.append((new_seq, new_cum_score))

            open_list.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
            closed_list.extend(open_list[beam_size:])
            open_list = open_list[:beam_size]

        final = predictions if predictions else closed_list
        final = sorted(final, key=lambda x: x[1] / x[0].size(1), reverse=True)[:beam_size]
        return [(seq.cpu(), score / seq.size(1)) for seq, score in final]

    def save_model(self, filename: str = "model.pth"):
        if self.epochwise_checkpointing_enabled:
            filename = f"checkpoint_epoch{self.config.epoch}.pth"
        self.config.model_state = self._model.state_dict()
        self.config.optimizer_state = self.optimizer.state_dict()
        if self._output_tokenizer != self._tokenizer and self._output_tokenizer is not None:
            tokenizer_path = self.config.output_dir.joinpath("tokenizer2")
            self._output_tokenizer.save(tokenizer_path)
            self.config.tokenizer2_path = tokenizer_path.relative_to(self.config.output_dir)
        else:
            self.config.tokenizer2_path = None
        if self._tokenizer is not None:
            tokenizer_path = self.config.output_dir.joinpath("tokenizer")
            self._tokenizer.save(tokenizer_path)
            self.config.tokenizer_path = tokenizer_path.relative_to(self.config.output_dir)
        else:
            self.config.tokenizer_path = None
        save_filepath = self.config.output_dir / filename
        Pickle.dump(self.config, save_filepath)
        Pickle.dump({"module": self.__class__.__module__, "class": self.__class__.__name__}, self.config.output_dir / "config.pkl")
        self._logger.info(f"Model saved to {save_filepath}")

    @staticmethod
    def from_pretrained(filepath: str | Path, logger: Optional[Logger] = None) -> "LegacyModel":
        filepath = BaseModel.check_model_filepath(filepath, logger=logger)
        config: LegacyConfig = BaseConfig.from_pretrained(filepath, logger)
        if config.tokenizer_path is not None:
            tok_filepath = config.output_dir.joinpath(config.tokenizer_path)
            tokenizer = load_module(tok_filepath, logger)
        else:
            tokenizer = None
        if config.tokenizer2_path is not None:
            tok_filepath = config.output_dir.joinpath(config.tokenizer2_path)
            output_tokenizer = load_module(tok_filepath, logger)
        else:
            output_tokenizer = None
        return LegacyModel(config, tokenizer, output_tokenizer, logger)
