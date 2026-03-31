try:
    import wandb
except ImportError:
    wandb = None

from pathlib import Path
from string import Template
from threading import Event
from datetime import datetime
from abc import abstractmethod, ABC
from typing import Optional, Any, Union
from dataclasses import dataclass, field
from os import getenv as os_getenv, system as os_system

from tqdm import tqdm
from pandas import DataFrame

from modelizer.tokenizers.abstract import BaseTokenizer
from modelizer.utils import torch, load_module, Logger, LoggerConfig, Pickle


class TrialState:
    """Minimal enum-like namespace matching the optuna.trial.TrialState constants used by the codebase."""
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    PRUNED = "PRUNED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"

    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"TrialState.{self.value}"

    def __eq__(self, other):
        if isinstance(other, TrialState):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


@dataclass
class TrialRecord:
    """
    Trial descriptor.
    Attributes mirror the subset of ``optuna.trial.FrozenTrial``

    * ``number``     – sequential trial index
    * ``value``      – scalar objective value (eval loss); lower is better
    * ``params``     – ``dict[str, Any]`` of hyperparameter name → value
    * ``user_attrs`` – ``dict[str, Any]`` of extra metadata (e.g. peak memory)
    * ``state``      – one of the ``_TrialState`` sentinel strings
    """
    number: int
    value: float
    params: dict = field(default_factory=dict)
    user_attrs: dict = field(default_factory=dict)
    state: TrialState = field(default_factory=lambda: TrialState("COMPLETE"))

    def __post_init__(self):
        self.params = dict(self.params)
        self.user_attrs = dict(self.user_attrs)


@dataclass
class Hyperparameters:
    """
    Container for hyperparameter optimization results.

    ``trials`` may hold either :class:`TrialRecord` objects (produced by
    :class:`GeneticOptimizer`) or ``optuna.trial.FrozenTrial`` objects
    (produced by :class:`Optimizer`).  Both expose the same duck-type
    interface (``.number``, ``.value``, ``.params``, ``.user_attrs``).
    """
    trials: list  # list[TrialRecord | FrozenTrial]
    last_trial_id = -1


class BaseConfig(ABC):
    """This is the base class for all model configurations."""

    def __init__(self, output_dir: str | Path, source: str, target: str, backward: bool, reduce_memory_usage: bool,
                 validation_steps_or_fraction: int | float, seed: int, wandb_token: Optional[str] = None, force_cpu: bool = False,
                 total_save_limit: int = 1, free_cached_memory: bool = False, reduce_spaces: bool = False, metadata: Optional[dict[str, Any]] = None):
        """
        Constructor for the BaseConfig class. Do not create an instance of this class directly.
        :param output_dir: The directory to save the model to
        :param source: The source language
        :param target: The target language
        :param backward: If True, the model will be trained in the backward direction
        :param reduce_memory_usage: If True, the model will reduce parameter's precision to save memory
        :param validation_steps_or_fraction: The interval at which to validate the model during training.
                                    If an integer, it is interpreted as the number of steps between validations.
                                    If a float, it is interpreted as a fraction of the training data used for validation.
        :param seed: The seed value for reproducibility
        :param wandb_token: (Optional) The Weights and Biases API token
        :param force_cpu: If True, force the model to run on CPU. Default is False.
        :param total_save_limit: The maximum number of model checkpoints to keep. Default is configs.TOTAL_SAVE_LIMIT.
        :param free_cached_memory: If True, free cached memory after each epoch.
        :param reduce_spaces: If True, the model will reduce spaces in the input data.
        :param metadata: Optional metadata dictionary to store additional information about the model.
        """
        if not isinstance(output_dir, (str, Path)):
            raise TypeError("Output directory must be a string or pathlib.Path object")
        if not isinstance(seed, int):
            raise TypeError("Seed value must be an integer.")
        if not isinstance(backward, bool):
            raise TypeError("Backward must be a boolean value.")
        if not isinstance(source, str):
            raise TypeError("Source must be a string.")
        if not isinstance(target, str):
            raise TypeError("Target must be a string.")
        if source == target:
            raise ValueError("Source and target languages must be different")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self._source = source
        self._target = target
        self._backward = backward
        self._reduce_memory_usage = reduce_memory_usage
        self._force_cpu = force_cpu
        self._seed = seed
        self._epoch: int = 0
        self._batch_size: int = 1
        self._name = f"{self.source}_to_{self.target}_{self.__class__.__name__.replace('Config', '').lower()}"
        self._wandb_token = wandb_token
        self._validation_steps_or_fraction = validation_steps_or_fraction
        self._trainable_parameters: Optional[int] = None
        self._hyperparams: Optional[Hyperparameters] = None
        self._total_save_limit = total_save_limit
        self._free_cached_memory = free_cached_memory
        self._reduce_spaces = reduce_spaces
        self._model_class: type[BaseModel] | None = None
        self._memory_requirements: float = 0
        self._metadata = dict() if metadata is None else metadata
        self.kwargs = {
            "tokenizer_params": {},
            "output_tokenizer_params": {},
            "memory_statistics": {},
            "cross_platform_compatibility": False,
        }

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        assert isinstance(value, int), f"epoch must be an integer, got {type(value)} => {value} instead."
        assert value >= 0, f"epoch must be greater than or equal to 0, got {value} instead."
        self._epoch = value

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        assert isinstance(value, int), f"batch_size must be an integer, got {type(value)} => {value} instead."
        assert value >= 1, f"batch_size must be greater than or equal to 1, got {value} instead."
        self._batch_size = value

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str | Path):
        assert isinstance(value, (str, Path)), f"output_dir must be a string or pathlib.Path object, got {type(value)} => {value} instead."
        self._output_dir = Path(value)
        self._output_dir.mkdir(exist_ok=True, parents=True)

    @property
    def source(self) -> str:
        return self._source

    @property
    def target(self) -> str:
        return self._target

    @property
    def backward(self) -> bool:
        return self._backward

    @property
    def reduce_memory_usage(self) -> bool:
        return self._reduce_memory_usage

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def hyperparams(self) -> Hyperparameters:
        return self._hyperparams

    @hyperparams.setter
    def hyperparams(self, value: Hyperparameters):
        assert isinstance(value, Hyperparameters), f"hyperparams must be an instance of Hyperparameters, got {type(value)} => {value} instead."
        self._hyperparams = value

    @name.setter
    def name(self, value: Optional[str]):
        assert value is None or isinstance(value, str), f"name can be None or a string, got {type(value)} => {value} instead."
        self._name = value

    @property
    def wandb_token(self) -> str | None:
        return self._wandb_token

    @wandb_token.setter
    def wandb_token(self, value: Optional[str]):
        assert value is None or isinstance(value, str), f"wandb_token can be None or a string, got {type(value)} => {value} instead."
        self._wandb_token = value

    @property
    def validation_steps_or_fraction(self) -> int | float:
        return self._validation_steps_or_fraction

    @validation_steps_or_fraction.setter
    def validation_steps_or_fraction(self, value: int | float):
        assert isinstance(value, (int, float)), f"validation_interval must be a number, got {type(value)} => {value} instead."
        if isinstance(value, float):
            assert 0 <= value < 1.0, f"validation_interval as a fraction must be between 0.0 and 1.0, got {value} instead."
        else:
            assert value >= 0, f"validation_interval must be greater than 0, got {value} instead."
        self._validation_steps_or_fraction = value

    @property
    def trainable_parameters(self) -> int | None:
        return self._trainable_parameters

    @property
    def force_cpu(self) -> bool:
        return self._force_cpu

    @force_cpu.setter
    def force_cpu(self, value: bool):
        assert isinstance(value, bool), f"force_cpu must be a boolean, got {type(value)} => {value} instead."
        self._force_cpu = value

    @trainable_parameters.setter
    def trainable_parameters(self, value: int | None):
        assert value is None or isinstance(value, int), f"trainable_parameters must be a integer or None, got {type(value)} instead."
        self._trainable_parameters = value

    @property
    def total_save_limit(self) -> int:
        return self._total_save_limit

    @total_save_limit.setter
    def total_save_limit(self, value: int):
        assert isinstance(value, int), f"total_save_limit must be an integer, got {type(value)} instead."
        assert value >= 1, f"total_save_limit must be greater than or equal to 1, got {value} instead."
        self._total_save_limit = value

    @property
    def free_cached_memory(self) -> bool:
        return self._free_cached_memory

    @free_cached_memory.setter
    def free_cached_memory(self, value: bool):
        assert isinstance(value, bool), f"free_cached_memory must be a boolean, got {type(value)} instead."
        self._free_cached_memory = value

    @property
    def reduce_spaces(self) -> bool:
        return self._reduce_spaces

    @property
    def model_class(self) -> type["BaseModel"] | None:
        return self._model_class

    @model_class.setter
    def model_class(self, value: type["BaseModel"] | None):
        assert value is None or issubclass(value, BaseModel), f"model_class can be None or a BaseModel, got {type(value)} instead."
        self._model_class = value

    @property
    def validator_configuration_filepath(self) -> Path:
        return self.output_dir.joinpath("validator.pkl")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        assert isinstance(value, dict), f"metadata must be a dict, got {type(value)} => {value} instead."
        self._metadata.update(value)

    @property
    def memory_requirements(self):
        """Estimated memory requirements in MB required for model operation."""
        return self._memory_requirements

    @memory_requirements.setter
    def memory_requirements(self, value: float):
        assert isinstance(value, Union[float, int]), f"memory_requirements must be a floating point or integer number, got {type(value)} => {value} instead."
        assert value > 0, f"memory_requirements must be greater than 0, got {value} instead."
        if value > self._memory_requirements:
            self._memory_requirements = float(value)

    @property
    def cross_platform_compatibility(self) -> bool:
        return self.kwargs.get("cross_platform_compatibility", False)

    def make_cross_platform_compatible(self):
        """Reimplement this method in the child class for more complex actions if necessary."""
        self.kwargs["cross_platform_compatibility"] = True

    def __str__(self):
        attributes_str = "\n".join([f"{k}: {v}" for k, v in self.get_attributes().items()])
        return f"{self.__class__.__name__}:\n{attributes_str}"

    def __repr__(self):
        attributes_str = "\n".join([f"{k}: {v}" for k, v in self.get_attributes_raw().items()])
        return f"{self.__class__.__name__}: {self.__hash__()}\n{attributes_str}"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_wandb_token"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_attributes(self) -> dict:
        """
        This method returns the configuration attributes as a dictionary and strips off underscore in variable names.
        It should be used to initialize the model
        :returns: dict with class arguments
        """
        attributes = {k.lstrip('_'): v for k, v in vars(self).items()}
        attributes.pop('wandb_token', None)      # Remove sensitive information
        attributes.pop('model_state', None)      # Remove noise
        attributes.pop('optimizer_state', None)  # Remove noise
        attributes.pop("hyperparams", None)      # Remove noise

        if "instructions" in attributes and attributes["instructions"] is not None and isinstance(attributes["instructions"], Template):
            attributes["instructions"] = attributes["instructions"].template

        return attributes

    def get_attributes_raw(self) -> dict[str, Any]:
        """
        This method returns the configuration attributes as a dictionary.
        It should be used in hyperparameter optimization.
        :return: dict with class arguments
        """
        return dict(vars(self).items())

    @abstractmethod
    def get_configurable_parameters(self, force_cpu: bool = False) -> dict[str, list[Any]]:
        """
        This method that returns the configuration attributes as a dictionary.
        Implement it in the child class to specify the hyperparameters and possible values for the optimization.
        It should be used in hyperparameter optimization.
        :param force_cpu: If True, the model will be forced to run on CPU, so the selection of GPU-supported hyperparameters is not needed.
        :return: dict with class arguments names and list of possible values. By default, must return an empty dictionary.
        """
        raise NotImplementedError("get_configurable_parameters method not implemented in the subclass")

    def check_constraints(self):
        """
        Optional method that checks the constraints of the configuration.
        Implement it in the child class to specify the constraints for the hyperparameters.
        :return: None
        """
        pass

    def set_params_from_dict(self, params: dict[str, Any]):
        """
        This method sets the parameters of the configuration object based on a dictionary of parameters.
        :param params: dict with class arguments names and their values.
        """
        for key, value in params.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                setattr(self, f"_{key}", value)

    def set_next_params(self):
        """
        This method sets the parameters of the configuration object based on the results of hyperparameter search.
        """
        if self._hyperparams is not None:
            self._hyperparams.last_trial_id += 1
            if self._hyperparams.last_trial_id < len(self._hyperparams.trials):
                self.set_params(self._hyperparams.last_trial_id)
            else:
                raise ValueError("No more trials available.")
        else:
            raise TypeError("No trials available. Please run hyperparameter optimization first.")
        return self._hyperparams.last_trial_id

    def set_params(self, idx: int):
        if self._hyperparams is None:
            raise TypeError("Hyperparameters not set. Please run hyperparameter optimization first and initialize the hyperparams attribute.")
        trial = self._hyperparams.trials[idx]
        for key, value in trial.params.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                setattr(self, f"_{key}", value)
        user_attrs = getattr(trial, "user_attrs", {}) or {}
        peak_memory_usage = user_attrs.get("peak_memory_usage", 0)
        if peak_memory_usage > 0:
            self._memory_requirements = float(peak_memory_usage)

    def have_more_trials(self) -> bool:
        """
        This method checks if there are more trials available for hyperparameter optimization.
        :return: True if there are more trials available, False otherwise.
        """
        return self._hyperparams is not None and self._hyperparams.last_trial_id < len(self._hyperparams.trials) - 1

    @staticmethod
    def from_pretrained(filepath: str | Path, logger: Optional[Logger] = None):  # -> Subclass of BaseConfig
        filepath = Path(filepath).resolve()
        if not filepath.is_file():
            error_msg = f"Config file not found at {filepath}"
            if logger is not None:
                logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        config = Pickle.load(filepath)
        config.output_dir = filepath.parent
        return config


class BaseModel(ABC):
    # Registry for config->model mapping
    _model_registry: dict[str, type] = {}

    def __init__(self, config: BaseConfig,
                 tokenizer: Optional[BaseTokenizer] = None,
                 output_tokenizer: Optional[BaseTokenizer] = None,
                 logger: Optional[Logger | LoggerConfig] = None):
        """
        Constructor for the BaseModel class. Do not create an instance of this class directly.
        :param config: The model configuration object
        :param tokenizer: (Optional) tokenizer object for the input data
        :param output_tokenizer: (Optional) tokenizer object for the output data
        :param logger: (Optional) logger object for logging training information
        """
        self.__init_wandb__(config)
        if logger is None:
            self._logger = Logger(None)
        elif isinstance(logger, LoggerConfig):
            self._logger = Logger(logger)
        else:
            self._logger = logger

        if not isinstance(config, BaseConfig):
            error_msg = "Config must be an instance of the BaseConfig class"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

        self._config = config
        if self._config.model_class is None:
            self._config.model_class = type(self)
        else:
            assert self._config.model_class is type(self), (f"model_class must be an instance of {type(self)}. "
                                                            f"Wrong configuration used: {self._config.model_class}")

        self._tokenizer = tokenizer
        if output_tokenizer is not None:
            self._output_tokenizer = output_tokenizer
            self._max_sequence_length = self.output_tokenizer.max_sequence_length
        elif tokenizer is not None:
            self._output_tokenizer = tokenizer
            self._max_sequence_length = self._tokenizer.max_sequence_length
        else:
            self._max_sequence_length = None

        if tokenizer is None and output_tokenizer is not None:
                error_msg = "Output tokenizer specified without input tokenizer."
                self._logger.error(error_msg)
                raise ValueError(error_msg)

        self._auto_saves_enabled = True
        self._epochwise_checkpointing_enabled = False
        self._checkpoint_id = 1
        self.__started_wandb_session__ = False
        self.__stop_event__ = Event()

        # Setup additional tokenizer parameters
        if self._tokenizer is not None and "tokenizer" in self._config.kwargs and len(self._config.kwargs["tokenizer"]):
            for key, value in self._config.kwargs["tokenizer"].items():
                if hasattr(self._tokenizer, key):
                    setattr(self._tokenizer, key, value)
                else:
                    self._logger.warning(f"Tokenizer has no attribute '{key}'. Skipping setting this attribute.")
        if self._output_tokenizer is not None and "output_tokenizer" in self._config.kwargs and len(self._config.kwargs["output_tokenizer"]):
            for key, value in self._config.kwargs["output_tokenizer"].items():
                if hasattr(self._output_tokenizer, key):
                    setattr(self._output_tokenizer, key, value)
                else:
                    self._logger.warning(f"Output tokenizer has no attribute '{key}'. Skipping setting this attribute.")

    @classmethod
    def register_model(cls, config_class: type, model_class: type):
        """Register a custom model for a config class."""
        if not issubclass(config_class, BaseConfig):
            raise TypeError("config_class must extend BaseConfig")
        if not issubclass(model_class, BaseModel):
            raise TypeError("model_class must extend BaseModel")
        cls._model_registry[config_class.__name__] = model_class

    @classmethod
    def get_registered_model(cls, config_class_name: str):
        """Get registered model class by config class name."""
        return cls._model_registry.get(config_class_name)

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def checkpoint_id(self) -> int:
        self._checkpoint_id += 1
        if self._checkpoint_id >= self.config.total_save_limit:
            self._checkpoint_id = 1
        return self._checkpoint_id

    @property
    def auto_saves_enabled(self) -> bool:
        return self._auto_saves_enabled

    def enable_auto_saves(self):
        self._auto_saves_enabled = True

    def disable_auto_saves(self):
        self._auto_saves_enabled = False

    @property
    def epochwise_checkpointing_enabled(self) -> bool:
        return self._epochwise_checkpointing_enabled

    def enable_epochwise_checkpointing(self):
        self._epochwise_checkpointing_enabled = True

    def disable_epochwise_checkpointing(self):
        self._epochwise_checkpointing_enabled = False

    @property
    def stop_event(self) -> Event:
        return self.__stop_event__

    def filter_cls_token(self, output: torch.Tensor) -> torch.Tensor:
        cls_indices = (output == self._output_tokenizer.cls_token_id).nonzero(as_tuple=True)
        if cls_indices and cls_indices[0].numel() > 0:
            idx = cls_indices[0][0].item()
            if idx + 1 < output.size(0):
                return output[idx + 1:]
        return output

    def __del__(self):
        try:
            if self.__started_wandb_session__ and wandb is not None and wandb.run is not None:
                wandb.finish()
                self.__started_wandb_session__ = False
                self._logger.info("Logged out from Weights and Biases")
        except Exception:
            print("Error during Weights and Biases logout. This is not critical, but please check your W&B account for any issues.")

    def __init_wandb__(self, config: BaseConfig):
        if isinstance(config, BaseConfig) and config.wandb_token is not None and wandb is not None:
            if wandb.api.api_key is None:
                wandb.login(key=config.wandb_token)
            elif wandb.api.api_key != config.wandb_token:
                os_system("wandb logout")
                wandb.login(key=config.wandb_token)

            try:
                _ = wandb.Api()
            except Exception:
                logged_in = False
            else:
                logged_in = True

            if logged_in:
                if wandb.run is None:
                    try:
                        job_name = f"_{job_id}" if (job_id := os_getenv('SLURM_JOBID')) else ""
                        name = f"{self.__class__.__name__}_{config.source}_{config.target}_{datetime.now().strftime('%d%m%y%H%M')}" + job_name
                        config_copy = config.__dict__.copy()
                        config_copy.pop("_wandb_token")
                        wandb.init(project="Modelizer", name=name, config=config_copy, resume="allow")
                        print(f"Logged in to Weights and Biases and initialized project {name}")
                    except Exception:
                        print("Error during Weights and Biases login. This is not critical, but please check your W&B account for any issues.")
                    self.__started_wandb_session__ = True
                else:
                    try:
                        config_copy = config.__dict__.copy()
                        config_copy.pop("_wandb_token")
                        wandb.config.update(config_copy, allow_val_change=True)
                        print("Updated Weights and Biases config for the current run")
                    except Exception:
                        print("Error during Weights and Biases config update. This is not critical, but please check your W&B account for any issues.")

    def update_wandb(self, wandb_token: str | None):
        if self.config.wandb_token != wandb_token:
            self.config.wandb_token = wandb_token
            self.__init_wandb__(self.config)

    def __str__(self):
        attributes_str = "\n".join([f"{k}: {v}" for k, v in self.config.get_attributes().items()])
        in_tokenizer_str = f"None" if self._tokenizer is None else self._tokenizer.__class__.__name__
        out_tokenizer_str = f"None" if self._output_tokenizer is None else self._output_tokenizer.__class__.__name__
        tokenizers_str = f"Tokenizers: {in_tokenizer_str} and {out_tokenizer_str}"
        parameters_str = "Trainable parameters: Unknown" if self._config.trainable_parameters is None else f"Trainable parameters: {self._config.trainable_parameters}"
        return f"{self.__class__.__name__}:\n{tokenizers_str}\n{attributes_str}\n{parameters_str}"

    def __repr__(self):
        attributes_str = "\n".join([f"{k}: {v}" for k, v in self.config.get_attributes().items()])
        if self._tokenizer is None:
            in_tokenizer_str = f"None: None"
        else:
            in_tokenizer_str = f"{self._tokenizer.__class__.__name__}: {str(self._tokenizer.__hash__())}"
        if self._output_tokenizer is None:
            out_tokenizer_str = f"None: None"
        else:
            out_tokenizer_str = f"{self._output_tokenizer.__class__.__name__}: {str(self._output_tokenizer.__hash__())}"
        tokenizers_str = f"Tokenizers: {in_tokenizer_str} and {out_tokenizer_str}"
        parameters_str = "Trainable parameters: Unknown" if self._config.trainable_parameters is None else f"Trainable parameters: {self._config.trainable_parameters}"
        return f"{self.__class__.__name__}: {self.__hash__()}\n{tokenizers_str}\n{attributes_str}\n{parameters_str}"

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_tokenizer(self, tokenizer: BaseTokenizer):
        if not isinstance(tokenizer, BaseTokenizer):
            error_msg = "Input tokenizer must be an instance of BaseTokenizer."
            self._logger.error(error_msg)
            raise TypeError(error_msg)
        self._tokenizer = tokenizer

    @property
    def output_tokenizer(self):
        return self._output_tokenizer

    def set_output_tokenizer(self, tokenizer: BaseTokenizer):
        if not isinstance(tokenizer, BaseTokenizer):
            error_msg = "Output tokenizer must be an instance of BaseTokenizer."
            self._logger.error(error_msg)
            raise TypeError(error_msg)
        self._output_tokenizer = tokenizer

    @property
    def logger(self) -> Logger:
        return self._logger

    @logger.setter
    def logger(self, value: Logger):
        if value is None:
            self._logger = Logger(None)
        elif isinstance(value, Logger):
            self._logger = value
        else:
            error_msg = "Logger must be an instance of the Logger class"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

    @abstractmethod
    def train(self, dataframe: DataFrame, num_epochs: int, batch_size: int = 1, *,
              stop_event: Optional[Event] = None, show_progress: bool = True,  **kwargs) -> tuple[float, float]:
        """
        This method implements the training loop for the model.
        :param dataframe: training data as a pandas DataFrame
        :param num_epochs: the number of training iterations over the entire dataset
        :param batch_size: the number of samples to process in one iteration
        :param kwargs: additional keyword arguments for the model
        :param stop_event: Optional threading.Event object to signal early stopping
        :param show_progress: if True, shows a progress bar during training
        :return: tuple containing training loss and validation loss
        """
        raise NotImplementedError("train method not implemented in the subclass")

    def retrain(self, dataframe: DataFrame, num_epochs: int, batch_size: Optional[int] = None,
                save_state: bool = False, stop_event: Optional[Event] = None, **kwargs):
        """
        This method implements the retraining feature for self-repair / subject-driven backpropagation.
        :param dataframe: training data as a pandas DataFrame
        :param num_epochs: the number of training iterations over the entire dataset
        :param batch_size: Optional number of samples to process in one iteration. If none, uses the batch size from the config.
        :param save_state: if True, saves the model state after retraining
        :param stop_event: Optional threading.Event object to signal early stopping
        :return: tuple containing training loss and validation loss
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        self.disable_auto_saves()
        saved_output_dir = self.config.output_dir

        if save_state:
            next_id = len(list(saved_output_dir.glob("retrain_*"))) + 1
            self.config.output_dir = saved_output_dir / f"retrain_{next_id}"

        saved_interval = self.config.validation_steps_or_fraction
        if isinstance(saved_interval, int):
            self.config.validation_steps_or_fraction = len(dataframe) * num_epochs + 1
        else:
            self.config.validation_steps_or_fraction = 0

        self.train(dataframe, num_epochs, batch_size, stop_event=stop_event, **kwargs)

        if save_state:
            self.save_model()

        self.config.validation_steps_or_fraction = saved_interval
        self.config.output_dir = saved_output_dir
        self.enable_auto_saves()

    def test(self, dataframe: DataFrame,
             *,
             max_length: Optional[int] = 256,
             save_results: bool = True,
             output_dir: Optional[str | Path] = None,
             test_name: Optional[str] = None,
             to_dataframe: bool = True,
             **kwargs) -> Union[DataFrame, list[dict[str, Any]]]:
        """
        This method implements the testing loop for the model.
        :param dataframe: testing data as a pandas DataFrame
        :param max_length: the maximum number of tokens model is allowed to generate. If None, infer the maximum length per input
        :param save_results: if True, saves the test results to a file
        :param output_dir: Optionally the directory to save the test results to
        :param test_name: Optionally the name of the test
        :param to_dataframe: if True, returns the results as a pandas DataFrame, otherwise as a list of dictionaries
        :param kwargs: additional keyword arguments for the model
        :returns: pandas DataFrame containing the test results or a list of dictionaries
        """
        if dataframe.empty:
            error_msg = "Input dataframe is empty."
            self._logger.error(error_msg)
            raise ValueError(error_msg)

        results = []

        for row in tqdm(dataframe.itertuples(), total=len(dataframe), desc="Testing", unit="sample", miniters=10, leave=False):
            src = getattr(row, self.config.source)
            tgt = getattr(row, self.config.target)

            if max_length is None or max_length < 1:
                tgt_len = len(self._output_tokenizer(tgt, truncation=True, padding=False, return_tensors=False)["input_ids"])
            else:
                tgt_len = max_length

            output = self.generate(src, max_length=tgt_len, **kwargs)
            results.append({"Input": src, "Expected": tgt, "Predicted": output})

        if to_dataframe:
            results = DataFrame(results, columns=["Input", "Expected", "Predicted"])

        if save_results:
            if test_name is None:
                test_name = ""
            else:
                test_name += "_"
            if isinstance(output_dir, str | Path):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = self.config.output_dir
            filepath = output_dir.joinpath(f"evaluation_results_{test_name}{self.config.name}.csv")
            if to_dataframe:
                results.to_csv(filepath, index=False)
            else:
                Pickle.dump(results, filepath)

        return results

    @abstractmethod
    def generate(self, input_data: Any, max_length: int = 256, **kwargs) -> Any:
        """
        This method generates a response from the model given the input data.
        It is recommended to use @torch.inference_mode() decorator in the subclass
        :param input_data: a string containing the not tokenized input data
        :param max_length: the maximum number of tokens model is allowed to generate
        :param kwargs: additional keyword arguments for the model
        :return: prediction as a string
        """
        raise NotImplementedError("generate method not implemented in the subclass")

    @abstractmethod
    def save_model(self, filename: str = "model.pth"):
        """
        This method saves the model config and weights to the specified file.
        :param filename: The name of the file to save the model to
        """
        raise NotImplementedError("save_model method not implemented in the subclass")

    @staticmethod
    @abstractmethod
    def from_pretrained(filepath: str | Path, logger: Optional[Logger] = None) -> "BaseModel":
        """
        This method re-initializes an instance of the model class loaded from the specified file.
        The config file should have been previously saved using the save_model method.
        :param filepath: The path as string or pathlib.Path object to the file containing the model
        :param logger: Optional Logger object for logging
        :return: An instance of the model class loaded from the specified file
        """
        raise NotImplementedError("from_pretrained method not implemented in the subclass")

    def cleanup_checkpoints(self):
        checkpoints = list(self.config.output_dir.glob("checkpoint_*.pth"))
        if checkpoints:
            for checkpoint in checkpoints:
                checkpoint.unlink()
            self._logger.info("Checkpoints removed")

    def forge_dataframe(self, data: tuple[Any, Any] | list[tuple[Any, Any]]) -> DataFrame:
        """
        This method converts the input data into a pandas DataFrame.
        :param data: a tuple or list of tuples containing the input and target data
        :return: a pandas DataFrame containing the input and target data
        """
        if isinstance(data, tuple):
            data = [data]
        return DataFrame(data, columns=[self.config.source, self.config.target])

    @staticmethod
    def initialize_weights(m: torch.nn.Module):
        for p in m.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @staticmethod
    def check_model_filepath(filepath: str | Path,
                             filename: str = "model.pth",
                             checkpoints_pattern: Optional[str] = "checkpoint_*.pth",
                             logger: Optional[Logger] = None) -> Path:
        """
        This method inspects the provided filepath and checks if it exists.
        If it is a directory, it looks for the specified filename or the latest checkpoint file matching.
        :param filepath: a string or pathlib.Path object representing the path to the model file or directory
        :param filename: name of the model file to look for in the directory.
        :param checkpoints_pattern: pattern to match checkpoint files in the directory.
        :param logger: Optional Logger object for logging
        :return: pathlib.Path object representing the model file path
        :raises FileNotFoundError: if the specified file or checkpoints are not found in the
        """
        assert isinstance(filepath, str | Path), "Filepath must be a string or Path object."
        filepath = Path(filepath)
        if filepath.exists():
            if filepath.is_dir():
                candidate = filepath.joinpath(filename)
                if candidate.is_file():
                    filepath = candidate
                elif checkpoints_pattern is not None:
                    checkpoints = list(filepath.glob(checkpoints_pattern))
                    if checkpoints:
                        candidate = max(checkpoints, key=lambda cp: cp.stat().st_mtime)
                        if candidate.is_file():
                            filepath = candidate
                        else:
                            error_msg = f"Neither '{filename}' nor checkpoints found in directory in {filepath}"
                            if logger is not None:
                                logger.error(error_msg)
                            raise FileNotFoundError(error_msg)
                else:
                    error_msg = f"Neither '{filename}' nor checkpoints found in {filepath}"
                    if logger is not None:
                        logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
            else:
                assert filepath.name == filename or (checkpoints_pattern is not None and
                                                     (checkpoints_pattern.split("_", 1)[0]
                                                      in filepath.stem and filepath.suffix == ".pth")), \
                    f"Invalid model file name. Must be '{filename}' or follow '{checkpoints_pattern}' pattern"
        else:
            error_msg = f"Path does not exist: {filepath}"
            if logger is not None:
                logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        return filepath

    def __check_tokenizer__(self, tokenizer: BaseTokenizer, tokenizer_path: Optional[str | Path], dataframe: DataFrame, column: str,) -> BaseTokenizer:
        """
        This method checks if the tokenizer is set and if not, loads it from the specified path.
        If the tokenizer is not trained, it trains it on the source data.
        :param tokenizer: The BaseTokenizer subclass object to check
        :param tokenizer_path: Path to the pretrained tokenizer configuration file
        :param dataframe: DataFrame containing the source and target data
        :param column: The column name in the DataFrame to train the tokenizer on
        """
        if tokenizer is None:
            if tokenizer_path is not None:
                tokenizer = load_module(self.config.output_dir.joinpath(tokenizer_path), self._logger)
            else:
                raise ValueError("Tokenizer is not provided and no path to pretrained configuration of tokenizer is specified.")
        if not tokenizer.is_trained:
            tokenizer.train(dataframe[column].tolist())
        return tokenizer

    def __check_input_tokenizer__(self, tokenizer_path: Optional[str | Path], dataframe: DataFrame):
        """
        This method checks if the input tokenizer is set and if not, loads it from the specified path.
        If the tokenizer is not trained, it trains it on the source data.
        :param dataframe: DataFrame containing the source and target data
        :param tokenizer_path: Path to the pretrained tokenizer configuration file
        """
        try:
            self._tokenizer = self.__check_tokenizer__(self._tokenizer, tokenizer_path, dataframe, self.config.source)
        except ValueError as e:
            self._logger.error(f"Error checking input tokenizer: {e}")
            raise

    def __check_output_tokenizer__(self, tokenizer_path: Optional[str | Path], dataframe: DataFrame):
        """
        This method checks if the output tokenizer is set and if not, loads it from the specified path.
        If the tokenizer is not trained, it trains it on the target data.
        :param dataframe: DataFrame containing the source and target data
        :param tokenizer_path: Path to the pretrained tokenizer configuration file for output
        """
        try:
            self._output_tokenizer = self.__check_tokenizer__(self._output_tokenizer, tokenizer_path, dataframe, self.config.target)
        except ValueError as e:
            self._logger.error(f"Error checking output tokenizer: {e}")
            raise

    def __check_both_tokenizers__(self, input_tokenizer_path: Optional[str | Path], output_tokenizer_path: Optional[str | Path], dataframe: DataFrame):
        """
        This method checks if both input and output tokenizers are set and if not, loads them from the specified paths.
        :param input_tokenizer_path: Path to the pretrained tokenizer configuration file for input
        :param output_tokenizer_path: Path to the pretrained tokenizer configuration file for output
        :param dataframe: DataFrame containing the source and target data
        """
        if self._tokenizer is None and self._output_tokenizer is None:
            return  # Either both tokenizers are not yet set, or they are not required for the model
        elif self._tokenizer == self._output_tokenizer and self._tokenizer is not None and not self._tokenizer.is_trained:
            self._tokenizer.train(dataframe[self.config.source].tolist() + dataframe[self.config.target].tolist())
        else:
            self.__check_input_tokenizer__(input_tokenizer_path, dataframe)
            self.__check_output_tokenizer__(output_tokenizer_path, dataframe)

    def check_tokenizers(self, *, dataframe: DataFrame, **kwargs):
        """
        This method checks if the tokenizers are set and if not, loads them from the specified paths.
        The implementation of this method is recommended in the subclass. By default, is does nothing.
        If the tokenizers are not trained, it trains them on the source and target data.
        We suggest to call one of the following methods in the subclass:
        - __check_input_tokenizer__ for input tokenizer
        - __check_output_tokenizer__ for output tokenizer
        - __check_both_tokenizers__ for both input and output tokenizers
        - __check_tokenizer__ for a custom logic of checking the tokenizers
        :param dataframe: DataFrame containing the source and target data
        :param kwargs: Additional keyword arguments for the checking process, such as input_tokenizer_path and output_tokenizer_path.
        """
        pass
