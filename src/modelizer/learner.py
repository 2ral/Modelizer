from pathlib import Path
from typing import Optional
from pandas import DataFrame
from functools import lru_cache

from modelizer.metrics import compute_metrics
from modelizer.tokenizers import BaseTokenizer
from modelizer.models.optimizer import Optimizer
from modelizer.models.abstract import BaseConfig
from modelizer.configs import MODELIZER_GENERATOR_CACHE_SIZE
from modelizer.models.legacy import LegacyConfig, LegacyModel
from modelizer.models.custom import EncoderDecoderModel, EncoderDecoderConfig
from modelizer.utils import Logger, LoggerConfig, Pickle, MemoryTracker, MemInfo, load_module


class Modelizer:
    """The Modelizer - is a high-level interface for learning behavior from data using sequence-to-sequence models."""
    def __init__(self, config: BaseConfig | str | Path,
                 *,
                 logger: Optional[Logger | LoggerConfig] = None,
                 tokenizer: Optional[BaseTokenizer] = None,
                 output_tokenizer: Optional[BaseTokenizer] = None,
                 track_memory_usage: bool = False,
                 max_allowed_memory: int = 0,
                 wandb_token: Optional[str] = None,
                 force_cpu: bool = False,
                 **_):
        """
        Initializes the Modelizer class.
        :param config: Config object or path to the configuration file.
                       To load an existing model, pass the path to the model directory.
        :param logger: (Optional) Logger or LoggerConfig for logging training performance.
                       If LoggerConfig is provided, the Logger object will be created internally.
        :param tokenizer: (Optional) input tokenizer, only required by custom Encoder-Decoder and Decoder-only models
        :param output_tokenizer: (Optional) output tokenizer, only required by custom Encoder-Decoder models
        :param track_memory_usage: If True, tracks memory usage during model operations.
        :param max_allowed_memory: (Optional) Maximum allowed memory in MB for the model. Default is 0 (no limit).
        :param wandb_token: (Optional) Weights & Biases API key for experiment tracking.
        :param force_cpu: (Optional) If True, forces the model to use CPU even if GPU is available.
        """
        available_memory = MemInfo.get_available_memory()

        logger = Logger.forge(logger)
        if wandb_token is not None:
            logger.config.log_to_wandb = True

        self._init_args = {
            "config": config,
            "logger": logger,
            "tokenizer": tokenizer,
            "output_tokenizer": output_tokenizer,
        }

        if isinstance(config, BaseConfig):
            config.wandb_token = wandb_token
            config.force_cpu = force_cpu

        if isinstance(config, str | Path):
            self.engine = load_module(Path(config), logger, force_cpu)
            self.engine.update_wandb(wandb_token)
        elif isinstance(config, LegacyConfig):
            assert tokenizer is not None, "Tokenizer must be provided for legacy encoder-decoder model"
            self.engine = LegacyModel(config, tokenizer, output_tokenizer, logger, force_cpu)
        elif isinstance(config, EncoderDecoderConfig):
            assert tokenizer is not None, "Tokenizer must be provided for custom encoder-decoder model"
            self.engine = EncoderDecoderModel(config, tokenizer, output_tokenizer, logger, force_cpu)
        elif isinstance(config, BaseConfig):
            from modelizer.forger import forge_custom_model
            self.engine = forge_custom_model(config, tokenizer, output_tokenizer, logger, force_cpu)
        else:
            error_msg = "Invalid configuration type"
            logger.error(error_msg)
            raise ValueError(error_msg)

        assert hasattr(self, "engine"), "Model engine was not initialized properly"

        self._init_args["config"] = self.engine.config
        self._init_args["logger"] = self.engine.logger

        device = self.engine.device if hasattr(self.engine, "device") else None
        self._engine_type = type(self.engine)
        self.save_model = self.engine.save_model
        self.__generate__ = lru_cache(maxsize=MODELIZER_GENERATOR_CACHE_SIZE)(self.engine.generate)
        if track_memory_usage:
            self.memory_tracker = MemoryTracker(duration=None, interval=1, model_config=self.engine.config, device=device)
            self.memory_tracker.start_tracking()
        else:
            self.memory_tracker = None

        self._last_config_id = None
        self._max_allowed_memory = max_allowed_memory

        if available_memory < self.engine.config.memory_requirements:
            message = (f"Warning: Available memory ({available_memory:.2f} MB) is less than the model's "
                       f"peak recorded memory usage ({self.engine.config.memory_requirements:.2f} MB). "
                       "This may lead to out-of-memory errors during training or inference.")
            if logger.is_null_logger:
                print(message)
            else:
                logger.warning(message)
        message = f"After loading the model, used memory is {MemInfo.get_used_memory(device):.2f} MB. Remaining {MemInfo.get_available_memory(device):.2f} MB."
        if logger.is_null_logger:
            print(message)
        else:
            logger.info(message)

    @property
    def engine_type(self):
        return self._engine_type

    @property
    def memory_usage_statistics(self) -> dict[str, float]:
        """
        Returns memory usage statistics if memory tracking is enabled.
        :return: A dictionary containing memory usage statistics.
        """
        return self.memory_tracker.memory_usage_statistics if self.memory_tracker is not None else {}

    def __del__(self):
        if self.memory_tracker is not None:
            self.memory_tracker.stop_tracking()

    def __str__(self):
        engine_type, stats = str(self.engine).split(": ", 1)
        return f"{self.__class__.__name__} -> {engine_type}:\n{stats}"

    def __repr__(self):
        engine_type, stats = repr(self.engine).split(": ", 1)
        return f"{self.__class__.__name__}: {self.__hash__()} -> {engine_type}: {stats}"

    def generate(self, input_data: str | list[str], max_length: int = 256, **kwargs) -> str | list[str]:
        """
        Generate output from the input data.
        :param input_data: Vector of input data or a single string
        :param max_length: Maximum length of the output
        :param kwargs: Additional keyword arguments.
        :return: Generated output as a string or a vector
        """
        if isinstance(input_data, list):
            input_data = tuple(input_data)
        return self.__generate__(input_data=input_data, max_length=max_length, **kwargs)

    def train(self, *args, **kwargs):
        self.__generate__.cache_clear()
        return self.engine.train(*args, **kwargs)

    def retrain(self, *args, **kwargs):
        self.__generate__.cache_clear()
        return self.engine.retrain(*args, **kwargs)

    def finalize(self):
        if self.memory_tracker is not None:
            self.memory_tracker.stop_tracking()

    def optimize(self, trials: int,
                 test_data: DataFrame,
                 search_space: Optional[dict[str, list]] = None,
                 num_train_epochs: int = 4,
                 batch_size: int = 1,
                 reset: bool = True) -> BaseConfig:
        """
        Runs hyperparameter optimization using Optuna, initializes the model with the best configuration, and returns the best configuration.
        :param trials: The number of trials to run
        :param test_data: DataFrame containing the test data
        :param search_space: The search space for hyperparameter optimization as a dictionary
        :param num_train_epochs: The number of training epochs
        :param batch_size: The batch size for training
        :param reset: If True, resets the model to the best configuration
        Note that genetic optimization may not be compatible with all model types and may require additional configuration.
        :return: The best configuration
        """
        if self.memory_tracker is not None:
            was_tracking = self.memory_tracker.is_tracking
            self.memory_tracker.stop_tracking()
        else:
            was_tracking = False
        self.__erase_attrs__()
        force_cpu = getattr(self._init_args["config"], "force_cpu", False)
        if search_space is None:
            search_space = self._init_args["config"].get_configurable_parameters(force_cpu)

        assert hasattr(self._init_args["config"], "wandb_token"), "Config must have wandb_token attribute for optimization"
        optimizer = Optimizer(**self._init_args, max_allowed_memory=self._max_allowed_memory)
        config = optimizer.run(trials, search_space, test_data, num_train_epochs, batch_size, clean_cache=False)

        if reset:
            config.set_next_params()
            self.__reinitialize_attrs__(config)

        if self.memory_tracker is not None and was_tracking:
            self.memory_tracker.start_tracking()
        return config

    def __reset__(self, config: BaseConfig):
        try:
            self._last_config_id = config.set_next_params()
        except ValueError:
            if self._last_config_id == 0:
                raise RuntimeError("No more configurations to set in the provided config.")
            self._last_config_id = 0
            config.set_params(0)
        self.__erase_attrs__()
        self.__reinitialize_attrs__(config)

    def __erase_attrs__(self):
        if hasattr(self, "__generate__"):
            self.__generate__.cache_clear()
            del self.__generate__
        if hasattr(self, "save_model"):
            del self.save_model
        if hasattr(self, "engine"):
            del self.engine

    def __reinitialize_attrs__(self, config: BaseConfig):
        self.engine = self._engine_type(**{k: v for k, v in self._init_args.items() if k != "config"}, config=config)
        if self.memory_tracker is not None:
            was_tracking = self.memory_tracker.is_tracking
            self.memory_tracker.stop_tracking()
            self.memory_tracker = MemoryTracker(duration=None, interval=1, model_config=config,
                                                device=self.engine.device if hasattr(self.engine, "device") else None)
            if was_tracking:
                self.memory_tracker.start_tracking()
        self.__generate__ = lru_cache(maxsize=MODELIZER_GENERATOR_CACHE_SIZE)(self.engine.generate)
        self.save_model = self.engine.save_model

    def test(self, dataframe: DataFrame,
             evaluation_type: str,
             *,
             max_length: Optional[int] = 256,
             save_results: bool = True,
             get_metrics: bool = True,
             output_dir: Optional[str | Path] = None,
             test_name: Optional[str] = None,
             **kwargs) -> tuple[DataFrame, DataFrame]:
        """
        Evaluates the model and computes the metrics.
        :param dataframe: DataFrame containing the test data
        :param evaluation_type: a string indicating the evaluation type (e.g. "Synthetic", "Real", "Tuned").
        :param max_length: the maximum number of tokens model is allowed to generate
        :param save_results: if True, saves the test results to a file
        :param get_metrics: if True, computes the metrics
        :param output_dir: Optionally the directory to save the test results to.
                           Default is None, which saves to the current working directory.
        :param test_name: Optionally, the name of the test. Default is None, which resolves to "test_{evaluation_type}".
        :param kwargs: additional keyword arguments for the model generation facility
        :return: a tuple containing the test results and the metrics
        """
        assert isinstance(evaluation_type, str) and len(evaluation_type) > 0, "Evaluation type must be a non-empty string"
        output_dir = Path.cwd() if output_dir is None else Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        test_name = f"test_{evaluation_type}" if test_name is None else test_name
        results = self.engine.test(dataframe, max_length=max_length, save_results=False, to_dataframe=False, **kwargs)  # list[{"Input": src, "Expected": tgt, "Predicted": output}]
        results_df = DataFrame(results, columns=["Input", "Expected", "Predicted"])

        if save_results:
            Pickle.dump(results, output_dir / f"evaluation_results_{test_name}.pkl")
            results_df.to_csv(output_dir / f"evaluation_results_{test_name}.csv", index=False)

        converted_results = tuple(
            (
                tuple(self.engine.output_tokenizer.tokenize_no_specials(row[1], to_string_tokens=True)),
                tuple(self.engine.output_tokenizer.tokenize_no_specials(row[2], to_string_tokens=True)),
            )
            for row in results_df.itertuples(index=False, name=None)  # No name for tuple
        )

        if save_results:
            filepath = output_dir / f"evaluation_processed_{test_name}.pkl"
            Pickle.dump(converted_results, filepath)

        if get_metrics:
            metrics = compute_metrics(converted_results, self.engine.config.source, self.engine.config.target, evaluation_type)
            metrics_df = DataFrame([metrics])
            if save_results:
                metrics_df.to_csv(output_dir / f"evaluation_metrics_{test_name}.csv", index=False)
        else:
            metrics_df = DataFrame()

        return results_df, metrics_df
