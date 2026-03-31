import optuna
import inspect

from math import isinf, isnan
from time import sleep
from queue import Queue
from copy import deepcopy
from typing import Optional

from pandas import DataFrame

from modelizer.tokenizers.abstract import BaseTokenizer
from modelizer.models.abstract import BaseConfig, Hyperparameters
from modelizer.utils import MemInfo, Logger, MemoryTracker, StoppableThread

from modelizer.models.legacy import LegacyConfig
from modelizer.models.custom import EncoderDecoderConfig



class Optimizer:
    def __init__(self, config: BaseConfig,
                 tokenizer: Optional[BaseTokenizer] = None,
                 output_tokenizer: Optional[BaseTokenizer] = None,
                 logger: Optional[Logger] = None,
                 max_allowed_memory: int = 0, **_):
        assert isinstance(config, BaseConfig), "Invalid configuration type"
        assert tokenizer is None or isinstance(tokenizer, BaseTokenizer), "Invalid input tokenizer type"
        assert output_tokenizer is None or isinstance(output_tokenizer, BaseTokenizer), "Invalid output tokenizer type"
        self._logger = Logger.forge(logger)

        if isinstance(config, LegacyConfig):
            if tokenizer is None:
                error_msg = "At least Input Tokenizer must be provided for custom legacy model"
                self._logger.error(error_msg)
                raise ValueError(error_msg)
            from modelizer.models.legacy import LegacyModel
            self._factory = LegacyModel
        elif isinstance(config, EncoderDecoderConfig):
            if tokenizer is None:
                error_msg = "At least Input Tokenizer must be provided for custom encoder-decoder model"
                self._logger.error(error_msg)
                raise ValueError(error_msg)
            from modelizer.models.custom import EncoderDecoderModel
            self._factory = EncoderDecoderModel
        else:
            self._factory = None

        self._backup_wandb = getattr(config, "wandb_token", None)
        if self._factory is not None:
            self._config = deepcopy(config)
        else:
            self._config = config
            self._config.wandb_token = None

        if hasattr(self._config, "model_state"):
            self._config.model_state = None

        self._tokenizer = tokenizer
        self._output_tokenizer = output_tokenizer
        self.history = dict()
        self._search_space = None
        self._test_data = None
        self._max_allowed_memory = max_allowed_memory
        self._num_train_epochs = 0
        self._batch_size = 0

    def run(self, trials: int, search_space: dict[str, list], test_data: DataFrame, num_train_epochs: int = 4, batch_size: int = 1, clean_cache: bool = True):
        if self._factory is None:
            self._logger.info("No hyperparameter optimization is available for the selected model type.")
            return self._config
        for key in search_space.keys():
            assert hasattr(self._config, key), f"Parameter {key} undefined in model configuration"
        self._logger.info(f"Starting hyperparameter optimization with {trials} trials "
                          f"using {num_train_epochs} train epochs with batch_size={batch_size}")
        if self._max_allowed_memory > 0:
            self._logger.info(f"Forcing memory requirements to {self._max_allowed_memory} megabytes.")
        if clean_cache:
            self.history.clear()
        self._test_data = test_data
        self._batch_size = batch_size
        self._search_space = search_space
        self._num_train_epochs = num_train_epochs
        study = optuna.create_study(direction="minimize")
        study.optimize(self.__test__, n_trials=trials, show_progress_bar=True)
        trials = self.filter_and_sort_trials(study)
        best_params = trials[0].params
        params_str = "\n".join([f"{key}: {value}" for key, value in best_params.items()])
        self._logger.info(f"HyperParameter optimization completed. Best trial: {trials[0].number}\nBest parameters:\n{params_str}")
        self._config.hyperparams = Hyperparameters(trials=trials)
        if hasattr(self._config, "wandb_token"):
            self._config.wandb_token = self._backup_wandb
        return self._config

    @staticmethod
    def filter_and_sort_trials(study: optuna.Study | list[optuna.trial.FrozenTrial], max_inf: int = 2) -> list[optuna.trial.FrozenTrial]:
        """
        Sort trials using a custom multi-criteria sort and keep:
          - All valid (non-inf) trials.
          - At most `max_inf` trials where trial.value == inf.

        :param study: An Optuna Study or a list of trials to be processed.
        :param max_inf: Maximum number of trials with value == inf to keep. Defaults to 2.
        :return: A list of filtered and sorted trials.
        """

        trials = study.trials if hasattr(study, "trials") else study
        sorted_trials = sorted(trials, key=Optimizer.__sort_trial__)

        filtered_trials = []
        inf_count = 0

        for t in sorted_trials:
            safe_value = Optimizer.__safe_trial_value__(t)
            if isinf(safe_value):
                if inf_count < max_inf:
                    filtered_trials.append(t)
                    inf_count += 1
                continue
            filtered_trials.append(t)

        return filtered_trials

    @staticmethod
    def __safe_trial_value__(trial):
        value = getattr(trial, "value", None)
        if value is None:
            return float("inf")
        try:
            if isnan(value):
                return float("inf")
        except TypeError:
            return float("inf")
        return value

    @staticmethod
    def __sort_trial__(trial):
        user_attrs = getattr(trial, "user_attrs", {}) or {}
        safe_value = Optimizer.__safe_trial_value__(trial)
        return (
            safe_value,
            user_attrs.get("peak_memory_usage", 0),
            trial.params.get("num_layers", float("inf")),
            trial.params.get("enc_layers", float("inf")),
            trial.params.get("dec_layers", float("inf")),
            trial.params.get("num_heads", float("inf")),
            trial.params.get("enc_heads", float("inf")),
            trial.params.get("dec_heads", float("inf")),
            trial.params.get("embedding_size", float("inf")),
            trial.params.get("enc_embedding_size", float("inf")),
            trial.params.get("dec_embedding_size", float("inf")),
            trial.params.get("hidden_size", float("inf")),
            trial.params.get("enc_hidden_size", float("inf")),
            trial.params.get("dec_hidden_size", float("inf")),
            trial.params.get("feedforward_size", float("inf")),
        )

    def __test__(self, trial):
        trial_count = 1000
        config = deepcopy(self._config)
        params = {key: trial.suggest_categorical(key, value) for key, value in self._search_space.items()}
        param_key = frozenset(params.items())
        while param_key in self.history and trial_count > 0:
            params = {key: trial.suggest_categorical(key, value) for key, value in self._search_space.items()}
            param_key = frozenset(params.items())
            trial_count -= 1

        if param_key in self.history:
            self._logger.info(f"Trial {trial.number} already completed with loss: {self.history[param_key]}")
            return self.history[param_key]
        else:
            self.history[param_key] = float("inf")

        for key, value in params.items():
            try:
                setattr(config, key, value)
            except AttributeError:
                setattr(config, f"_{key}", value)

        tracker = None
        eval_loss = float("inf")
        peak_memory_consumption = 0
        try:
            MemInfo.clean_cache()
            supports = {"output_tokenizer": False, "force_cpu": False}
            try:
                params = inspect.signature(self._factory).parameters
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                for name in ("output_tokenizer", "force_cpu"):
                    if has_var_kw:
                        supports[name] = True
                    else:
                        param = params.get(name)
                        supports[name] = (param is not None and param.kind != inspect.Parameter.POSITIONAL_ONLY)
            except (TypeError, ValueError):
                pass

            kwargs = {}
            if supports["output_tokenizer"]:
                kwargs["output_tokenizer"] = self._output_tokenizer
            if supports["force_cpu"]:
                kwargs["force_cpu"] = getattr(config, "force_cpu", False)

            model = self._factory(config, self._tokenizer, **kwargs)

            if self._max_allowed_memory > 0:
                tracker = MemoryTracker(duration=None, interval=1, model_config=None,
                                        device=model.device if hasattr(model, "device") else None)
                tracker.start_tracking()
                training_results_queue = Queue(maxsize=1)

                def _train_runner():
                    try:
                        res = model.train(self._test_data, self._num_train_epochs, self._batch_size, show_progress=False)
                        training_results_queue.put(("ok", res))
                    except Exception as ex:
                        training_results_queue.put(("err", ex))

                train_thread = StoppableThread(target=_train_runner, name=f"train-trial-{trial.number}", daemon=True)
                train_thread.start()

                while train_thread.is_alive():
                    if tracker.peak_memory_usage > self._max_allowed_memory:
                        train_thread.request_stop()
                        break
                    sleep(1)
                train_thread.join()
                tracker.stop_tracking()
                peak_memory_consumption = tracker.peak_memory_usage
                if not training_results_queue.empty() and tracker.peak_memory_usage <= self._max_allowed_memory:
                    status, payload = training_results_queue.get()
                    if status == "err":
                        raise payload
                    else:
                        _, eval_loss = payload
                elif tracker.peak_memory_usage > self._max_allowed_memory:
                    eval_loss = float("inf")
                else:
                    raise RuntimeError("Training thread finished without producing results")
            else:
                model.disable_auto_saves()
                _, eval_loss = model.train(self._test_data, self._num_train_epochs, self._batch_size, show_progress=False)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            if tracker is not None:
                tracker.stop_tracking()
                peak_memory_consumption = tracker.peak_memory_usage
            self._logger.info(f"Trial {trial.number} failed with exception: {e}")
        else:
            if peak_memory_consumption > self._max_allowed_memory:
                self._logger.info(f"Trial {trial.number} exceeded memory requirements: "
                                  f"{peak_memory_consumption} > {self._max_allowed_memory}")
            else:
                self._logger.info(f"Trial {trial.number} completed with loss: {eval_loss}")
            self.history[param_key] = eval_loss
        finally:
            params_log = "\n".join(f"{key}: {value}" for key, value in params.items())
            self._logger.info(f"Trial {trial.number} configuration:\n{params_log}")
            try:
                trial.set_user_attr("peak_memory_usage", peak_memory_consumption)
            except Exception as e:
                self._logger.info(f"Failed to write user attributes for trial {trial.number}: {e}")
        return eval_loss
