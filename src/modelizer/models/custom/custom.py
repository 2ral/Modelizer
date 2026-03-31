import torch
import torch.export

from pathlib import Path
from string import Template
from threading import Event
from datetime import datetime
from typing import Any, Optional
from math import exp as math_exp
from concurrent.futures import CancelledError

from tqdm.auto import tqdm

from modelizer import configs
from modelizer.models import dataset
from modelizer.utils import Logger, MemInfo, Pickle, TorchHelpers, get_time_diff
from modelizer.models.abstract import BaseTokenizer, BaseConfig, BaseModel, abstractmethod



class CustomConfig(BaseConfig):
    def __init__(self,
                 output_dir: str | Path,
                 source: str,
                 target: str,
                 backward: bool,
                 optimizer: str = configs.OPTIMIZER,
                 scheduler: Optional[str] = configs.SCHEDULER,
                 learning_rate: float = configs.LEARNING_RATE,
                 weight_decay: float = configs.WEIGHT_DECAY,
                 b1: float = configs.B1,
                 b2: float = configs.B2,
                 eps: float = configs.EPS,
                 clip_grad: float = configs.CLIP_GRAD,
                 validation_fraction: float = configs.VALIDATION_FRACTION,
                 checkpoint_interval: int = configs.CHECKPOINT_INTERVAL,
                 compile_model: bool = False,
                 force_cpu: bool = False,
                 reduce_memory_usage: bool = False,
                 free_cached_memory: bool = False,
                 total_save_limit: int = configs.TOTAL_SAVE_LIMIT,
                 wandb_token: Optional[str] = None,
                 instructions: Optional[Template] = None,
                 seed: int = configs.SEED,
                 reduce_spaces: bool = False,
                 metadata: Optional[dict[str, Any]] = None, **_):
        """
        This is the base class for all custom-trained Modelizer models.
        :param output_dir: Directory to save the model and tokenizer.
        :param source: Source type.
        :param target: Target type.
        :param backward: If True, the model will be trained in the backward direction.
        :param optimizer: Optimizer to use for training. Default is configs.OPTIMIZER.
                          Possible values are 'sgd', 'adam', 'adamw', 'adagrad', 'paged_adamw', 'rmsprop',
                          'sgd_8bit', 'adam_8bit', 'adamw_8bit', 'adagrad_8bit', 'paged_adamw_8bit', 'rmsprop_8bit'.
                          8bit versions only function on systems with CUDA-enabled PyTorch installation.
        :param scheduler: Learning rate scheduler to use for training. Default is configs.SCHEDULER.
                          Possible values: 'linear', 'cyclic', 'cosine', 'step', 'polynomial', None
        :param learning_rate: Learning rate for the optimizer. Default is configs.LEARNING_RATE.
        :param weight_decay: Weight decay for the optimizer. Default is configs.WEIGHT_DECAY.
        :param b1: Beta1 parameter for the optimizer. Default is configs.B1.
        :param b2: Beta2 parameter for the optimizer. Default is configs.B2.
        :param eps: Epsilon parameter for the optimizer. Default is configs.EPS.
        :param clip_grad: Gradient clipping value. Default is configs.CLIP_GRAD.
        :param validation_fraction: Training dataset fraction used to validate the model during training. Default is configs.VALIDATION_FRACTION.
        :param checkpoint_interval: Interval for saving checkpoints. Default is configs.CHECKPOINT_INTERVAL.
        :param compile_model: If True, compile the model before training. Default is False.
        :param force_cpu: If True, force the model to run on CPU. Default is False.
        :param reduce_memory_usage: If True, the model will reduce the parameter's precision to save memory. Default is False.
        :param free_cached_memory: If True, free cached memory after each epoch.
        :param total_save_limit: The maximum number of model checkpoints to keep. Default is configs.TOTAL_SAVE_LIMIT.
        :param wandb_token: (Optional) Weights and Biases API token. Default is None.
        :param seed: Random seed for reproducibility, Default is configs.SEED.
        :param reduce_spaces: If True, the model will reduce spaces in the input data.
        :param metadata: Optional metadata dictionary to store additional information about the model.
        """
        super().__init__(output_dir, source, target, backward, reduce_memory_usage, validation_fraction, seed,
                         wandb_token, force_cpu, total_save_limit, free_cached_memory, reduce_spaces, metadata)
        self._model_state = None
        self._optimizer_state = None
        self._tokenizer_path = None
        self._instructions = instructions
        self._dtype_configured = False
        # Train Configs
        self._compile_model = compile_model
        self._clip_grad = clip_grad
        self._checkpoint_interval = checkpoint_interval
        # Optimizer Configs
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._b1 = b1
        self._b2 = b2
        self._eps = eps

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value

    @property
    def instructions(self):
        return self._instructions

    @property
    def optimizer_state(self):
        return self._optimizer_state

    @optimizer_state.setter
    def optimizer_state(self, value):
        self._optimizer_state = value

    @property
    def tokenizer_path(self):
        return self._tokenizer_path

    @tokenizer_path.setter
    def tokenizer_path(self, value: str | Path | None):
        assert value is None or isinstance(value, str | Path), "Tokenizer path can be None or a string or pathlib.Path object"
        self._tokenizer_path = value

    @property
    def compile_model(self) -> bool:
        return self._compile_model

    @property
    def clip_grad(self) -> float:
        return self._clip_grad

    @property
    def checkpoint_interval(self) -> int:
        return self._checkpoint_interval

    @property
    def optimizer(self):
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

    @abstractmethod
    def get_configurable_parameters(self, force_cpu: bool = False) -> dict[str, list[Any]]:
        """
        This method that returns the configuration attributes as a dictionary.
        Implement it in the child class to specify the hyperparameters and possible values for the optimization.
        It should be used in hyperparameter optimization.
        :param force_cpu: If True, the model will be forced to run on CPU, so the selection of GPU-supported hyperparameters is not needed.
        :return: dict with class arguments names and list of possible values. By default, returns an empty dictionary.
        """
        raise NotImplementedError("get_configurable_parameters method not implemented in the subclass")


########################################################################################################################
#                                               Base Class for Custom Models                                           #
########################################################################################################################
class CustomModel(BaseModel):
    """Base class for all custom models."""

    def __init__(self, model, config: CustomConfig, tokenizer: Optional[BaseTokenizer] = None,
                 output_tokenizer: Optional[BaseTokenizer] = None, logger: Optional[Logger] = None, force_cpu: bool = False):
        config.force_cpu = bool(config.force_cpu or force_cpu)
        super().__init__(config, tokenizer, output_tokenizer, logger)
        self._model = model
        self._model.apply(BaseModel.initialize_weights)
        self.training_performance = {"train_loss": list(), "valid_loss": list()}
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = TorchHelpers.initialize_device(config.seed, logger=logger, force_cpu=config.force_cpu)

        if config.compile_model and self.device.type == "mps":
            # torch.compile is not supported on MPS devices
            config._compile_model = False

        if config.compile_model:
            self._model = torch.compile(self._model)
        if config.model_state is not None:
            self._model.load_state_dict(config.model_state)

        self._model = self._model.to(self.device)
        if self._tokenizer is not None:
            self._tokenizer.device = self.device
        if self._output_tokenizer is not None:
            self._output_tokenizer.device = self.device

        match config.optimizer:
            case "sgd":
                # Explicitly pass momentum=0.0 to avoid environments where SGD without
                # a momentum argument is rejected (some backends may require it).
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
                    self._logger.warning("SGD8bit requires momentum > 0; using momentum=0.9.")
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

            case _:  # "By default AdamW is used"
                self.optimizer = torch.optim.AdamW(self._model.parameters(),
                                                   lr=config.learning_rate,
                                                   weight_decay=config.weight_decay,
                                                   eps=config.eps,
                                                   betas=(config.b1, config.b2))

        if config.optimizer_state is not None:
            self.optimizer.load_state_dict(config.optimizer_state)

        match config.scheduler:
            case "linear":
                self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)
            case "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4)
            case "cyclic":
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                                   base_lr=self.config.learning_rate / 100,
                                                                   max_lr=self.config.learning_rate * 100)
            case "polynomial":
                self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer)
            case "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
            case _:
                self.scheduler = None

        self.__optimizer_step__ = self.__optimizer_compiled_step__ if config.compile_model else self.__optimizer_basic_step__
        self._logger.info(f"{self.__class__.__name__} got initialized. Total Parameters: {sum(p.numel() for p in self._model.parameters())}")
        self.config.trainable_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self._model.eval()

    @torch.compile(fullgraph=False)
    def __optimizer_compiled_step__(self):
        self.optimizer.step()

    def __optimizer_basic_step__(self):
        self.optimizer.step()

    @abstractmethod
    def __forge_dataset__(self, dataframe: dataset.DataFrame) -> dataset.TorchDataset:
        """
        This method is responsible for creating a TorchDataset from the given DataFrame. Do not call this method directly.
        :param dataframe: pandas DataFrame object
        :return: TorchDataset object
        """
        raise NotImplementedError("__forge_dataset__ method not implemented in the subclass")

    @abstractmethod
    def __forward__(self, sample) -> torch.Tensor:
        """
        This method is responsible for passing the input through the model and returning the loss. Do not call this method directly.
        :param sample: torch.Tensor | dict[str, torch.Tensor]
        """
        raise NotImplementedError("__forward__ method not implemented in the subclass")

    def train(self, dataframe: dataset.DataFrame, num_epochs: int, batch_size: int = 1, *,
              stop_event: Optional[Event] = None, show_progress: bool = True, **_) -> tuple[float, float]:
        if stop_event is not None and isinstance(stop_event, Event):
            self.__stop_event__ = stop_event
        self.config.batch_size = batch_size
        self.config.report_epoch_progress = show_progress
        if not isinstance(dataframe, dataset.DataFrame) or dataframe.empty:
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

        self.check_tokenizers(
            dataframe=dataframe,
            input_tokenizer_path=getattr(self.config, "tokenizer_path", None),
            output_tokenizer_path=getattr(self.config, "tokenizer2_path", None)
        )

        self._logger.info("Preparing DataLoader...")
        train_dataloader, valid_dataloader = self.__forge_dataset__(dataframe).get_dataloaders(
            batch_size=batch_size,
            shuffle=True,
            validation_fraction=self.config.validation_steps_or_fraction
        )

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
        self._logger.info(f"Training completed in {get_time_diff(start_time)}")
        self._logger.info(f"Best Train Loss: {best_train_loss} | Best Eval Loss: {best_valid_loss}")
        return best_train_loss, best_valid_loss


    def train_epoch(self, iterator: torch.utils.data.DataLoader) -> float:
        iterator_len = len(iterator)
        assert iterator_len > 0, "Training iterator is empty"
        self._model.train()
        epoch_loss = 0.
        successful_batches = 0
        iterable = tqdm(iterator, total=iterator_len, leave=False, miniters=10) if self.config.report_epoch_progress else iterator

        for entries in iterable:
            self.optimizer.zero_grad()
            try:
                t_loss = self.__forward__(entries)
            except Exception as e:
                self._logger.warning(f"Skipping failing training batch due to error: {e!r}")
                continue
            else:
                successful_batches += 1

            t_loss.backward()
            if self.config.clip_grad:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.config.clip_grad)
            self.__optimizer_step__()
            if self.scheduler is not None:
                self.scheduler.step()
            epoch_loss += t_loss.item()

        if self.config.free_cached_memory:
            MemInfo.clean_cache()

        if successful_batches == 0:
            self._logger.error("All training batches failed in this epoch; returning inf loss.")
            return float("inf")

        self._logger.info(MemInfo.get_memory_usage_stats("Training phase"))
        return epoch_loss / successful_batches

    def valid_epoch(self, iterator: torch.utils.data.DataLoader) -> float:
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
                    e_loss = self.__forward__(entries)
            except Exception as e:
                self._logger.warning(f"Skipping failing validation batch due to error: {e!r}")
                continue
            else:
                successful_batches += 1

            epoch_loss += e_loss.item()

        if self.config.free_cached_memory:
            MemInfo.clean_cache()

        if successful_batches == 0:
            self._logger.error("All validation batches failed in this epoch; returning inf loss.")
            return float("inf")

        self._logger.info(MemInfo.get_memory_usage_stats("Validation phase"))
        return epoch_loss / successful_batches

    def save_model(self, filename: str = "model.pth"):
        if self.epochwise_checkpointing_enabled:
            filename = f"checkpoint_epoch{self.config.epoch}.pth"

        model_to_save = self._model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        model_state = model_to_save.state_dict()
        optimizer_state = self.optimizer.state_dict()
        self.config.model_state = model_state
        self.config.optimizer_state = optimizer_state

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
