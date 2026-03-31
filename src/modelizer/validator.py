from pathlib import Path
from inspect import signature, Parameter

from pandas import DataFrame
from typing import Optional, Sequence, Iterable, Any, Callable

from modelizer import configs
from modelizer.learner import Modelizer
from modelizer.utils import Logger, LoggerConfig, Pickle
from modelizer.generators.subjects import BaseSubject, ExecutionState
from modelizer.backpropagation import SequenceDebugger, MutationTester
from modelizer.metrics import Metrics, FeatureResults, ValidationResults


class ValidationConfig:
    """A configuration class for the Validator."""
    def __init__(self,
                 subject: BaseSubject,
                 timeout: int = 60,
                 debugging_cores: int = 1,
                 mutating_cores: int = 1,
                 mutation_trials: int = 50,
                 mutation_count: int = 5,
                 mutation_strategies: Optional[Sequence | Iterable] = None,
                 placeholders: Optional[Sequence[str]] = None,
                 supress_debugger_assertions: bool = True,
                 collect_only_passing_tests: bool = True,
                 early_stop_on_no_progress_trials: int = 5,
                 delta_debugging_mode: str = "+",
                 seed: int = configs.SEED,
                 post_formating: Callable[[Any], Any] = None):
        """
        Initializes the ValidationConfig class.
        :param subject: an instance of BaseSubject that can interact with the program under test
        :param timeout: time in seconds to wait for the execution to complete
        :param debugging_cores: the number of SequenceDebugger instances running in parallel
        :param mutating_cores: the number of mutation fuzzer instances running in parallel
        :param mutation_trials: the number of attempts fuzzer tries to come up with a valid input.
        :param mutation_count: the maximum number of sequential mutations to apply to the input data. Default is 5
        :param mutation_strategies: (Optional) Callable objects that implement mutations on the input data.
                                    Default is None, which uses the default mutation strategies.
        :param placeholders: (Optional) List of placeholders to use for mutation.
        :param supress_debugger_assertions: a flag to suppress debugger assertions.
        :param collect_only_passing_tests: a flag to collect only passing tests after mutation testing.
        :param early_stop_on_no_progress_trials: The number of consecutive trials with no progress after which to stop the repair process. Default is 5.
        :param delta_debugging_mode: The mode for delta debugging.
                                     Can be "+" for maximizing passing inputs
                                     or "-" for minimizing failing inputs
                                     or "+-" for both. Default is "+".
        :param seed: the seed for reproducibility of results.
        :param post_formating: (Optional) A callable object to be used as an encoding function for the program output / model input data.
        """
        assert isinstance(subject, BaseSubject), "subject must be a BaseSubject instance"
        assert delta_debugging_mode in SequenceDebugger.SUPPORTED_MODES, f"delta debugging mode can be one of {SequenceDebugger.SUPPORTED_MODES}, got {delta_debugging_mode} instead."
        self._seed = seed
        self._subject = subject
        self._timeout = timeout
        self._mutation_count = mutation_count
        self._mutation_trials = mutation_trials
        self._mutating_cores = mutating_cores
        self._debugging_cores = debugging_cores
        self._placeholders = placeholders
        self._mutation_strategies = mutation_strategies
        self._supress_debugger_assertions = supress_debugger_assertions
        self._collect_only_passing_tests = collect_only_passing_tests
        self._delta_debugging_mode = delta_debugging_mode
        self._early_stop_on_no_progress_trials = early_stop_on_no_progress_trials
        self._post_formating_func: Callable[[Any], Any] = lambda x: x if post_formating is None else post_formating

    @property
    def subject(self) -> BaseSubject:
        return self._subject

    @property
    def mutation_count(self) -> int:
        return self._mutation_count

    @property
    def mutation_trials(self) -> int:
        return self._mutation_trials

    @property
    def timeout(self) -> int:
        return self._timeout

    @property
    def placeholders(self) -> Sequence[str] | None:
        return self._placeholders

    @property
    def mutation_strategies(self) -> Sequence | Iterable | None:
        return self._mutation_strategies

    @property
    def supress_debugger_assertions(self) -> bool:
        return self._supress_debugger_assertions

    @property
    def collect_only_passing_tests(self) -> bool:
        return self._collect_only_passing_tests

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def early_stop_on_no_progress_trials(self) -> int:
        return self._early_stop_on_no_progress_trials

    @property
    def delta_debugging_mode(self) -> str:
        return self._delta_debugging_mode

    @delta_debugging_mode.setter
    def delta_debugging_mode(self, mode: str):
        assert mode in SequenceDebugger.SUPPORTED_MODES, f"delta debugging mode can be one of {SequenceDebugger.SUPPORTED_MODES}, got {mode} instead."
        self._delta_debugging_mode = mode

    @property
    def post_formating_func(self) -> Callable[[Any], Any]:
        return self._post_formating_func

    @post_formating_func.setter
    def post_formating_func(self, func: Callable[[Any], Any]):
        assert callable(func), "post_formating_func must be a callable function."
        positional_params = [
            p for p in signature(func).parameters.values()
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        ]
        required = [p for p in positional_params if p.default is Parameter.empty]
        assert len(required) == 1, "post_formating_func function must accept exactly one required positional argument."
        self._post_formating_func = func

    def __getstate__(self):
        """
        Returns the state of the ValidationConfig object for serialization.
        :return: A dictionary containing the state of the ValidationConfig object.
        """
        return {
            "subject": Pickle.to_bytes(self._subject),
            "timeout": self._timeout,
            "mutation_count": self._mutation_count,
            "mutation_trials": self._mutation_trials,
            "mutating_cores": self._mutating_cores,
            "debugging_cores": self._debugging_cores,
            "placeholders": self._placeholders,
            "mutation_strategies": self._mutation_strategies,
            "supress_debugger_assertions": self._supress_debugger_assertions,
            "collect_only_passing_tests": self._collect_only_passing_tests,
            "delta_debugging_mode": self._delta_debugging_mode,
            "post_formating_func": Pickle.to_bytes(self._post_formating_func),
        }

    def __setstate__(self, state):
        """
        Restores the state of the ValidationConfig object from a serialized state.
        :param state: A dictionary containing the state of the ValidationConfig object.
        """
        self._subject = Pickle.from_bytes(state["subject"])
        self._timeout = state["timeout"]
        self._mutation_count = state["mutation_count"]
        self._mutation_trials = state["mutation_trials"]
        self._mutating_cores = state["mutating_cores"]
        self._debugging_cores = state["debugging_cores"]
        self._placeholders = state["placeholders"]
        self._mutation_strategies = state["mutation_strategies"]
        self._supress_debugger_assertions = state["supress_debugger_assertions"]
        self._collect_only_passing_tests = state["collect_only_passing_tests"]
        self._delta_debugging_mode = state["delta_debugging_mode"]
        self._post_formating_func = Pickle.from_bytes(state["post_formating_func"])


class Validator:
    """The Validator - is a high-level interface for validating and repairing model predictions"""

    def __init__(self, model: Modelizer | Path | str, config: Optional[ValidationConfig | Path | str] = None,  *,
                 logger: Optional[Logger | LoggerConfig] = None, wandb_token: Optional[str] = None):
        """
        Initializes the Validator class.
        :param model: an instance of Modelizer or path to the model configuration file
        :param config: an instance of ValidationConfig containing the validation configuration, if None, it will be loaded from the model directory
        :param logger: optional logger for logging training performance
        :param wandb_token: optional Weights & Biases token for experiment tracking
        """

        if isinstance(model, Path | str):
            self.model = Modelizer(model, logger=logger, wandb_token=wandb_token)
        elif isinstance(model, Modelizer):
            self.model = model
            self.model.engine.update_wandb(wandb_token)
            self.model.engine.logger = Logger.forge(logger)
        else:
            logger = Logger.forge(logger)
            error_msg = f"Invalid 'model' type: {type(model)}. Expected a Modelizer instance or a path to a model configuration file."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if hasattr(self.model.engine.config, "validation_fraction"):
            self.model.engine.config.validation_fraction = None

        if isinstance(config, Path | str):
            self._config = Pickle.load(config)
        elif isinstance(config, ValidationConfig):
            self._config = config
            Pickle.dump(self._config, self.model.engine.config.validator_configuration_filepath)
        elif config is None and self.model.engine.config.validator_configuration_filepath.exists():
            self._config = Pickle.load(self.model.engine.config.validator_configuration_filepath)
        else:
            error_msg = f"Invalid 'config' type: {type(config)}. Expected a ValidationConfig instance or a path to a validation configuration file."
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._config.subject.pre_execution()
        self._metrics = Metrics()
        subject_tokenizer = self.model.engine.output_tokenizer if self.model.engine.config.backward else self.model.engine.tokenizer
        self._mutator = MutationTester(self._config.subject, subject_tokenizer,
                                       timeout=self._config.timeout, max_mutations=self._config.mutation_count,
                                       seed=self._config.seed, mutation_strategies=self._config.mutation_strategies,
                                       placeholders=self._config.placeholders, collect_only_passing_tests=self._config.collect_only_passing_tests)

        self._debugger = SequenceDebugger(self._config.subject, subject_tokenizer,
                                          supress_assertions=self._config.supress_debugger_assertions, timeout=self._config.timeout)

    @property
    def logger(self) -> Logger:
        return self.model.engine.logger

    @property
    def config(self) -> ValidationConfig:
        return self._config

    def __del__(self):
        self._config.subject.post_execution()

    def __process_and_tokenize_feature_data__(self, model_input, model_output):
        feature_tokenizer = self.model.engine.output_tokenizer
        program_input_tokenizer = self.model.engine.tokenizer
        monitored = self._config.subject.execute(model_input)
        monitored = self._config.post_formating_func(monitored)
        monitored = self._config.subject.get_encoder()(monitored)
        prog_inp_tokens = program_input_tokenizer(model_input, return_tensors=False)["input_ids"]
        monitored_tokens = feature_tokenizer(monitored, return_tensors=False)["input_ids"]
        feature_tokens = feature_tokenizer(model_output, return_tensors=False)["input_ids"]
        return monitored, monitored_tokens, prog_inp_tokens, feature_tokens

    def __save_repaired_model__(self, output_dir):
        if output_dir is not None:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            backup = self.model.engine.config.output_dir
            self.model.engine.config.output_dir = output_dir
            self.model.save_model("repaired.pth")
            self.model.engine.config.output_dir = backup
        else:
            self.model.save_model("repaired.pth")

    def __prepare_data__(self, program_input, debug: bool, mutate: bool):
        train_data = []
        if len(program_input) == 0:
            return train_data
        if debug:
            repaired = self._debugger.repair(program_input, mode=self._config.delta_debugging_mode)
            if repaired is not None and (
                    repaired[2] == ExecutionState.PASS or not self._config.collect_only_passing_tests):
                train_data.append(
                    (repaired[1], repaired[0]) if self.model.engine.config.backward else (repaired[0], repaired[1]))
        if mutate:
            train_data.extend(
                self._mutator.mutate(program_input, self.model.engine.config.backward, self._config.mutation_trials))
        return train_data

    def validate(self, model_input: Any, max_length: Optional[int] = None, **kwargs) -> ValidationResults:
        """
        Validates given data using Program under Test and Model.
        :param model_input: model input
        :param max_length: maximum number of tokens / concepts to generate
        :param kwargs: additional keyword arguments for the model generation facility
        :return: ValidationResults object
        """
        if max_length is None or max_length <= 0:
            max_length = self.model.engine.max_sequence_length
        model_input_tokens = self.model.engine.tokenizer(model_input, return_tensors=False)
        model_output = self.model.generate(model_input, max_length=max_length, **kwargs)
        model_output_tokens = self.model.engine.output_tokenizer(model_output, return_tensors=False)
        if self.model.engine.config.backward:
            truth = self._config.subject.execute(model_output)
            truth = self._config.post_formating_func(truth)
            truth = self._config.subject.get_encoder()(truth)
            truth_tokens = self.model.engine.tokenizer(truth, return_tensors=False)
        else:
            truth = self._config.subject.execute(model_input)
            truth = self._config.post_formating_func(truth)
            truth = self._config.subject.get_encoder()(truth)
            truth_tokens = self.model.engine.output_tokenizer(truth, return_tensors=False)

        return ValidationResults(self.model.engine.config.backward, model_input, model_input_tokens,
                                 model_output, model_output_tokens, truth, truth_tokens)

    def validate_set(self, model_inputs: Iterable[Any] | Sequence[Any],
                     max_length: Optional[int] = None,
                     stop_on_exception: bool = True, **kwargs) -> list[ValidationResults]:
        """
        Validates a set of data.
        :param model_inputs: Iterable or Sequence of model inputs
        :param max_length: max_length number of tokens / concepts to generate
        :param kwargs: additional keyword arguments for the model generation facility
        :param stop_on_exception: whether to stop when validation fails
        :return: list of ValidationResults objects
        """
        if max_length is None or max_length <= 0:
            max_length = self.model.engine.max_sequence_length
        try:
            dataset = set(model_inputs)
        except TypeError:
            dataset = model_inputs

        results: list[ValidationResults] = []
        for inp in dataset:
            try:
                result = self.validate(inp, max_length, **kwargs)
            except Exception as e:
                error_msg = f"Validation failed for input {inp}: {e}"
                self.model.engine.logger.error(error_msg)
                if stop_on_exception:
                    raise RuntimeError(error_msg)
            else:
                results.append(result)

        return results

    def validate_feature(self, model_input: Any, max_length: Optional[int] = None, **kwargs) -> FeatureResults:
        """
        Validates given data using Program under Test and Model.
        :param model_input: model input
        :param max_length: maximum number of tokens / concepts to generate
        :param kwargs: additional keyword arguments for the model generation facility
        :return: FeatureResults object
        """
        if max_length is None or max_length <= 0:
            max_length = self.model.engine.max_sequence_length
        model_output = self.model.generate(model_input, max_length=max_length, **kwargs)

        if self.model.engine.config.backward:
            monitored, monitored_tokens, prog_inp_tokens, feature_tokens = self.__process_and_tokenize_feature_data__(model_input, model_output)
            result = FeatureResults(program_input=model_output, features=model_input, monitored=monitored, is_backward=True,
                                    input_tokens=prog_inp_tokens, feature_tokens=feature_tokens, monitored_tokens=monitored_tokens)
        else:
            monitored, monitored_tokens, prog_inp_tokens, feature_tokens = self.__process_and_tokenize_feature_data__(model_input, model_output)
            result = FeatureResults(program_input=model_input, features=model_output, monitored=monitored, is_backward=False,
                                    input_tokens=prog_inp_tokens, feature_tokens=feature_tokens, monitored_tokens=monitored_tokens)
        return result

    def validate_feature_set(self, model_inputs: Iterable[Any] | Sequence[Any],
                             max_length: Optional[int] = None,
                             stop_on_exception: bool = True, **kwargs) -> list[FeatureResults]:
        """
        Validates a set of data.
        :param model_inputs: Iterable or Sequence of model inputs
        :param max_length: max_length number of tokens / concepts to generate
        :param kwargs: additional keyword arguments for the model generation facility
        :param stop_on_exception: whether to stop when validation fails
        :return: list of FeatureResults objects
        """
        if max_length is None or max_length <= 0:
            max_length = self.model.engine.max_sequence_length
        try:
            dataset = set(model_inputs)
        except TypeError:
            dataset = model_inputs

        results: list[FeatureResults] = []
        for inp in dataset:
            try:
                result = self.validate_feature(inp, max_length, **kwargs)
            except Exception as e:
                error_msg = f"Feature Validation failed for input {inp}: {e}"
                self.model.engine.logger.error(error_msg)
                if stop_on_exception:
                    raise RuntimeError(error_msg)
            else:
                if result is not None:
                    results.append(result)

        return results

    def repair(self, data: ValidationResults | FeatureResults, *, trials: int = 1, train_epochs: int = 1,
               debug: bool = True, mutate: bool = False, checkpoint: bool = False, strict: bool = True,
               output_dir: Optional[str | Path] = None) -> ValidationResults | FeatureResults | None:
        """
        Repairs the faulty model prediction.
        :param data: ValidationResult or FeatureResults containing the faulty model prediction
        :param trials: Number of trials to run for the repair
        :param train_epochs: Number of training epochs per trial
        :param debug: If True, the program input will be debugged before retraining
        :param mutate: If True, the program input will be mutated before retraining
        :param checkpoint: If True, the model state will be saved after the repair
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :param output_dir: Directory to save the repair results. By default, it uses the model's directory.
        :return: ValidationResult or FeatureResults object depending on the input containing the results of successful repair or last repair attempt
        """
        assert isinstance(trials, int) and trials > 0, "Number of trials must be an integer that is greater than 0."
        assert isinstance(train_epochs, int) and train_epochs > 0, "Number of training epochs must be an integer that is greater than 0."
        validator = self.validate_feature if isinstance(data, FeatureResults) else self.validate

        if data.is_equal(strict=strict):
            error_msg = "Input is valid and can't be repaired."
            self.model.engine.logger.error(error_msg)
            raise ValueError(error_msg)

        result = data
        initial_trials = trials
        source = self.model.engine.config.source
        target = self.model.engine.config.target

        train_data = [data.get_training_data()] if data.is_evaluable else list()
        if debug or mutate:
            train_data.extend(self.__prepare_data__(data.get_subject_input_tokenized(), debug, mutate))

        while (not result.is_equal(strict=strict)) and trials > 0:
            trials -= 1
            df = DataFrame(train_data, columns=[source, target])
            df = df.drop_duplicates()
            self.model.retrain(df, train_epochs, None, checkpoint)
            result_candidate = validator(data.get_model_input())
            if result_candidate is not None and result_candidate.is_evaluable:
                train_data.append(result_candidate.get_training_data())
                result = result_candidate
            if debug or mutate:
                train_data.extend(self.__prepare_data__(data.get_subject_input_tokenized(), debug, mutate))

        result.used_trials = initial_trials - trials

        if not checkpoint:
            self.__save_repaired_model__(output_dir)

        return result

    def repair_set(self, data: Iterable[ValidationResults | FeatureResults] | Sequence[ValidationResults | FeatureResults], *, trials: int = 1,
                   train_epochs: int = 1, debug: bool = True, mutate: bool = False, checkpoint: bool = False,
                   strict: bool = True, output_dir: Optional[str | Path] = None, early_stopping: bool = False) -> tuple[list[ValidationResults | FeatureResults], list[ValidationResults | FeatureResults], float]:
        """
        Repairs a set of faulty model predictions.
        :param data: Iterable or Sequence of ValidationResult or FeatureResults objects containing the faulty model predictions
        :param trials: Number of trials to run for the repair
        :param train_epochs: Number of training epochs per trial
        :param debug: If True, the program input will be debugged before retraining
        :param mutate: If True, the program input will be mutated before retraining
        :param checkpoint: If True, the model state will be saved after the repair by replacing the main model checkpoint
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :param output_dir: Directory to save the repair results. By default, it uses the model's directory.
        :param early_stopping: If True, stops the repair process if no progress is made in repairing inputs for 3 consecutive trials.
        :return: Tuple of following elements:
                 - List of ValidationResult or FeatureResults objects containing passes
                 - List of ValidationResult or FeatureResults objects containing failures
                 - Percentage of repaired inputs
        """
        assert isinstance(trials, int) and trials > 0, f"Number of trials must be an integer that is greater than 0, given {trials}"
        assert isinstance(train_epochs, int) and train_epochs > 0, f"Number of training epochs must be an integer that is greater than 0, given {train_epochs}"
        assert isinstance(debug, bool), f"Debug must be a boolean value, given {debug}"
        assert isinstance(mutate, bool), f"Mutate must be a boolean value, given {mutate}"
        assert isinstance(checkpoint, bool), f"Checkpoint must be a boolean value, given {checkpoint}"
        assert isinstance(strict, bool), f"Strict equality must be a boolean value, given {strict}"

        faulty_inputs = {d for d in data if not d.is_equal(strict=strict)}

        initial_fault_count = len(faulty_inputs)
        if initial_fault_count == 0:
            error_msg = "No faulty inputs found in the dataset."
            self.model.engine.logger.error(error_msg)
            raise ValueError(error_msg)

        is_feature_model = isinstance(next(iter(faulty_inputs)), FeatureResults)
        validator = self.validate_feature if is_feature_model else self.validate

        passing_inputs = []
        unrepairable_inputs = []
        initial_trials = trials
        no_progress_trials = 0
        source = self.model.engine.config.source
        target = self.model.engine.config.target

        train_data = {d.get_training_data() for d in faulty_inputs if d.is_evaluable}
        if debug or mutate:
            train_data |= {item for fi in faulty_inputs for item in
                           self.__prepare_data__(fi.get_subject_input_tokenized(), debug, mutate)}

        while faulty_inputs and trials > 0:
            before_faulty_inputs_count = len(faulty_inputs)
            self.model.engine.logger.info(f"Trying to repair {len(faulty_inputs)} faulty inputs")
            trials -= 1
            used_trials = initial_trials - trials
            df = DataFrame(train_data, columns=[source, target])
            df = df.drop_duplicates()
            self.model.engine.logger.info(f"Training model with {len(df)} repair samples")

            self.model.retrain(df, train_epochs, None, checkpoint)
            results = [validator(fi.get_model_input()) for fi in faulty_inputs]
            results = [r for r in results if r is not None]
            for r in results:
                r.used_trials = used_trials
            newly_passing_inputs = [r for r in results if r.is_equal(strict=strict) and r.is_evaluable]
            self.model.engine.logger.info(f"Successfully repaired {len(newly_passing_inputs)} inputs out of {len(results)}")
            passing_inputs.extend(newly_passing_inputs)
            faulty_inputs = [r for r in results if not r.is_equal(strict=strict) and r.is_evaluable]
            newly_unrepairable = [r for r in results if not r.is_evaluable]
            unrepairable_inputs.extend(newly_unrepairable)
            train_data = set([r.get_training_data() for r in results if r.is_evaluable])  # using results instead of faulty_inputs for more data
            if debug or mutate:
                train_data |= {item for fi in faulty_inputs for item in
                               self.__prepare_data__(fi.get_subject_input_tokenized(), debug, mutate)}
                train_data |= {item for fi in newly_unrepairable for item in
                               self.__prepare_data__(fi.get_subject_input_tokenized(), debug, mutate)}

            after_faulty_inputs_count = len(faulty_inputs) + len(newly_unrepairable)
            if early_stopping and after_faulty_inputs_count == before_faulty_inputs_count:
                no_progress_trials += 1
                if no_progress_trials >= self._config.early_stop_on_no_progress_trials:
                    self.model.engine.logger.info(f"No progress in repairing inputs for "
                                                  f"{self._config.early_stop_on_no_progress_trials} consecutive trials."
                                                  f" Stopping repair process after trial {used_trials}.")
                    break
            else:
                no_progress_trials = 0

        faulty_inputs.extend(unrepairable_inputs)

        if not checkpoint:
            self.__save_repaired_model__(output_dir)

        return passing_inputs, list(faulty_inputs), round((len(passing_inputs) / initial_fault_count) * 100, 2)
