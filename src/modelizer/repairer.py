import argparse

from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass
from os import getenv as os_getenv
from random import seed as random_seed, sample as random_sample
from typing import Optional, Iterable, Sequence, Any, Union, Tuple

from modelizer.configs import SEED
from modelizer.learner import Modelizer
from modelizer.generators import BaseSubject
from modelizer.validator import Validator, ValidationConfig
from modelizer.metrics import FeatureResults, ValidationResults
from modelizer.tokenizers.features import ForgingPolicy, FeatureTokenizer
from modelizer.utils import Logger, LoggerConfig, DataHandlers, Pickle, get_time_diff, retrieve_init_arguments


@dataclass
class RepairArguments:
    subject: str
    num_samples: int
    trials: int
    epochs: int
    max_length: Optional[int] = None
    root_dir: Optional[str | Path] = None
    repair_results_dir: Optional[Path] = None
    model_dir: Optional[Path] = None
    data_dir: Optional[Path] = None
    model_instance: Optional[Modelizer] = None
    subject_instance: Optional[BaseSubject] = None
    move: bool = False
    debug: bool = False
    mutate: bool = False
    feature_model: bool = False
    strict_evaluation: bool = True
    force_repairing: bool = False
    model_type = "generic"
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    test_split_name: Optional[str] = None
    seed: int = SEED
    wandb: Optional[str] = None
    forging_policy: Optional[str | ForgingPolicy] = None
    startup_arguments: Optional[str] = retrieve_init_arguments()

    def __post_init__(self):
        self.root_dir = DataHandlers.locate_temp_dir() if self.root_dir is None else Path(self.root_dir).resolve()
        self.repair_results_dir = DataHandlers.locate_temp_dir() / "repair_results" if self.move else self.root_dir / "repair_results"
        self.model_dir = self.root_dir / "models"
        self.data_dir = self.root_dir / "results"

        if self.subject is not None and len(self.subject) > 0:
            self.repair_results_dir = self.repair_results_dir / self.subject
            self.model_dir = self.model_dir / self.subject
            self.data_dir = self.data_dir / self.subject

        if self.forging_policy is not None:
            if isinstance(self.forging_policy, ForgingPolicy):
                assert self.forging_policy is not ForgingPolicy.UNKNOWN, "If forging_policy is specified, it cannot be UNKNOWN"
            elif isinstance(self.forging_policy, str):
                if len(self.forging_policy) > 0:
                    assert self.forging_policy in ForgingPolicy.valid_policies(), (f"Provided forging_policy='{self.forging_policy}' "
                                                                                   f"is not valid. Valid policies: {ForgingPolicy.valid_policies()}")
                else:
                    self.forging_policy = None
            else:
                raise TypeError("forging_policy must be either a string or an instance of ForgingPolicy")

        if self.model_instance is None:
            self.model_instance = Modelizer(self.model_dir)

        if self.feature_model:
            self.strict_evaluation = False

    def get_dataset_paths(self, folder: Optional[Path] = None, filename: Optional[str] = None) -> list[Path]:
        """
        Unified dataset path resolver.

        - If self.test_split_name is None:
          behaves like the old get_dataset_paths(folder, filename).

        - If self.test_split_name is not None:
          behaves like the old get_dataset_paths_custom_split(folder)
          and ignores 'filename'.
        """

        def try_to_find(work_dir: Path):
            file = None
            for d in [d for d in work_dir.iterdir() if d.is_dir()]:
                try:
                    found = self.get_dataset_paths(d, filename)
                except FileNotFoundError:
                    continue
                else:
                    if found:
                        file = found[0] if isinstance(found, list) else found
                        break
            if isinstance(file, Path) and file.suffix == ".zip":
                file = DataHandlers.unzip(file)
            return file

        filepaths_list: list[Path] = []
        folder = self.data_dir if folder is None else Path(folder)

        if self.test_split_name is not None:
            # pattern = evaluation_results + split_name: Optional[str] + forging_policy: Optional[str]
            retrieve_first_file = True
            base_pattern = "evaluation_results*"

            assert isinstance(self.test_split_name, str), f"expected test_split_name to be a string, given: {type(self.test_split_name)}"
            assert len(self.test_split_name) > 0, "if test_split_name is specified, it cannot be an empty string"

            if self.test_split_name == "merged":
                # if test_split_name is merged, retrieve all files with pattern=evaluation_results*policy*
                retrieve_first_file = False
            else:
                base_pattern += f"{self.test_split_name}*"

            if self.forging_policy is not None:
                assert isinstance(self.forging_policy, (str, ForgingPolicy)), f"expected forging_policy to be a string or ForgingPolicy, given: {type(self.forging_policy)}"
                policy_str = self.forging_policy if isinstance(self.forging_policy, str) else self.forging_policy.value
                base_pattern += f"{policy_str}*"

            zip_files = list(folder.glob("*.zip"))
            pkl_files = list(folder.glob(f"{base_pattern}.pkl"))
            csv_files = list(folder.glob(f"{base_pattern}.csv"))
            input_parts = [i for i in base_pattern.split("*") if len(i) > 0]

            if len(zip_files) > 0:
                unzipped = [DataHandlers.unzip(f) for f in zip_files]
                filepaths_list.extend([f for f in unzipped if f is not None and all(part in f.name for part in input_parts)])

            if len(filepaths_list) == 0:
                if len(pkl_files) > 0:
                    filepaths_list.extend([f for f in pkl_files if all(part in f.name for part in input_parts)])
                elif len(csv_files) > 0:
                    filepaths_list.extend([f for f in csv_files if all(part in f.name for part in input_parts)])

            if len(filepaths_list) == 0:
                raise FileNotFoundError(f"No dataset in the directory `{folder}` for pattern `{base_pattern}`")
            elif retrieve_first_file:
                filepaths_list = filepaths_list[:1]

        elif filename is None:
            zip_files = list(folder.glob("*.zip"))
            pkl_files = list(folder.glob("evaluation_results*.pkl"))
            csv_files = list(folder.glob("evaluation_results*.csv"))

            if len(zip_files) > 0:
                for zip_file in zip_files:
                    filepath = DataHandlers.unzip(zip_file)
                    if filepath is not None and "evaluation_results" in filepath.name and filepath.suffix in (".pkl", ".csv"):
                        filepaths_list.append(filepath)

            if len(filepaths_list) == 0:
                if len(pkl_files) > 0:
                    filepaths_list.extend(pkl_files)
                elif len(csv_files) > 0:
                    filepaths_list.extend(csv_files)
                else:
                    filepath = try_to_find(folder)
                    if filepath is not None:
                        filepaths_list.append(filepath)
                if len(filepaths_list) == 0:
                    raise FileNotFoundError(f"No dataset in the directory `{folder}`")

        else:
            filepath = folder / filename
            if filepath.is_file():
                if filepath.suffix == ".zip":
                    filepath = DataHandlers.unzip(filepath)
            else:
                filepath = try_to_find(folder)

            if filepath is None:
                raise FileNotFoundError(f"Dataset file `{filename}` not found in the directory `{folder}`")
            filepaths_list.append(filepath)

        return filepaths_list

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model_instance'] = None
        if self.subject_instance is not None:
            state['subject_instance'] = Pickle.to_bytes(self.subject_instance)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.model_instance is None:
            self.model_instance = Modelizer(self.model_dir)
        if self.subject_instance is not None and isinstance(self.subject_instance, bytes):
            self.subject_instance = Pickle.from_bytes(self.subject_instance)

    @staticmethod
    def forge(state: dict):
        arguments = RepairArguments(**state)
        if arguments.model_instance is None:
            arguments.model_instance = Modelizer(arguments.model_dir)
        if arguments.subject_instance is not None and isinstance(arguments.subject_instance, bytes):
            arguments.subject_instance = Pickle.from_bytes(arguments.subject_instance)

    @staticmethod
    def get_sample_arguments() -> "RepairArguments":
        """
        Create a reference TrainArguments instance with safe, minimal defaults.
        Useful for generating example YAML or bootstrapping configs.
        """
        sample = RepairArguments.__new__(RepairArguments)
        sample.subject = "demo_subject"
        sample.num_samples = 100
        sample.trials = 50
        sample.epochs = 5
        sample.max_length = 128
        sample.root_dir = "./.modelizer"  # example root dir; will be resolved on load
        sample.model_instance = None
        sample.subject_instance = None
        sample.move = False
        sample.debug = False
        sample.mutate = False
        sample.feature_model = False
        sample.force_repairing = False
        sample.task_id = None
        sample.seed = SEED
        sample.forging_policy = None
        sample.strict_evaluation = True
        sample.startup_arguments = retrieve_init_arguments()
        return sample

class Repairer:
    def __init__(self,
                 arguments: str | Path | RepairArguments,
                 validation_config: Optional[ValidationConfig] = None,):
        if isinstance(arguments, (str, Path)):
            path = Path(arguments)
            if path.suffix == ".pkl":
                arguments = Pickle.load(arguments)
            else:
                raise ValueError(f"Unsupported arguments file format: {path.suffix}, only .pkl, .yml, .yaml are supported")
        assert isinstance(arguments, RepairArguments), f"arguments could be None, a path to a file or an instance of RepairArguments, given: {type(arguments)}"
        if validation_config is None:
            assert arguments.subject_instance is not None, "Subject instance must be provided in arguments when validation_config is None"
            validation_config = ValidationConfig(arguments.subject_instance)

        self._arguments = arguments
        logger_config = LoggerConfig(root_dir=self._arguments.repair_results_dir, log_to_wandb=self.arguments.wandb is not None)
        self._validator = Validator(self._arguments.model_instance, validation_config, logger=logger_config, wandb_token=arguments.wandb)
        self.arguments.model_type = self._validator.model.engine.config.__class__.__name__.lower().replace("config", "")

        if self._arguments.feature_model:
            if self._arguments.forging_policy is not None:
                policy = arguments.forging_policy if isinstance(self._arguments.forging_policy, ForgingPolicy) else ForgingPolicy(arguments.forging_policy)
                self.logger.info(f"Setting feature forging policy to '{arguments.forging_policy}'")
                if isinstance(self._validator.model.engine.tokenizer, FeatureTokenizer):
                    self._validator.model.engine.tokenizer.feature_encoder.forging = policy
                if isinstance(self._validator.model.engine.output_tokenizer, FeatureTokenizer):
                    self._validator.model.engine.output_tokenizer.feature_encoder.forging = policy
        self.logger.info("Repairer initialized.")
        self.logger.info(str(self._arguments))

    @property
    def arguments(self) -> RepairArguments:
        return self._arguments

    @property
    def logger(self) -> Logger:
        return self._validator.logger

    @property
    def validator(self) -> Validator:
        return self._validator

    @staticmethod
    def __init_argument_parser__() -> argparse.ArgumentParser:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--subject", type=str, required=True, help="Optional subject name")
        arg_parser.add_argument("--root-dir", type=str, default="", help="Optional root directory for model, tokenizers, results")
        arg_parser.add_argument("--num-samples", type=int, default=0, help="Maximum number of samples from dataset to use for repair. 0 means no limit")
        arg_parser.add_argument("--trials", type=int, default=100, help="Maximum number of repair trials per input")
        arg_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for model training")
        arg_parser.add_argument("--max-length", type=int, default=0, help="Maximum length of the output sequence. 0 means no specified")
        arg_parser.add_argument('--move', action='store_true', help='Move model with results to the TEMP directory')
        arg_parser.add_argument('--debug', action='store_true', help='Flag to enable debug mode for repair process')
        arg_parser.add_argument('--mutate', action='store_true', help='Flag to enable mutation of inputs during repair process')
        arg_parser.add_argument("--task-id", type=str, default="", help="Optional slurm task identifier.")
        arg_parser.add_argument("--task-name", type=str, default="", help="Optional slurm task name.")
        arg_parser.add_argument('--feature-model', action='store_true', help='Flag to indicate feature-based model repair')
        arg_parser.add_argument('--permissive-evaluation', action='store_true', help='Flag to indicate non-strict evaluation during repair')
        arg_parser.add_argument('--force-repairing', action='store_true', help='Flag to indicate prevention of early stopping during bulk repair')
        arg_parser.add_argument('--wandb', type=str, default="", help="Weight and Biases API Token")
        arg_parser.add_argument("--forging-policy", type=str, default="", help="Optional forging policy for tokenizer during repair")
        arg_parser.add_argument("--test-split-name", type=str, default="", help="Dataset split name to use for repair (default: auto)")
        return arg_parser

    @staticmethod
    def print_help():
        arg_parser = Repairer.__init_argument_parser__()
        arg_parser.print_help()

    @staticmethod
    def parse_arguments() -> RepairArguments:
        arg_parser = Repairer.__init_argument_parser__()
        arguments = arg_parser.parse_args()

        assert isinstance(arguments.num_samples, int) and arguments.num_samples >= 0, f"Number of samples must be a non-negative integer, given: {arguments.num_samples}"
        assert isinstance(arguments.trials, int) and arguments.trials > 0, f"Trials must be a positive integer, given: {arguments.trials}"
        assert isinstance(arguments.epochs, int) and arguments.epochs > 0, f"Epochs must be a positive integer, given: {arguments.epochs}"

        return RepairArguments(
            subject=arguments.subject,
            root_dir=arguments.root_dir if len(arguments.root_dir) > 0 else None,
            num_samples=arguments.num_samples,
            trials=arguments.trials,
            epochs=arguments.epochs,
            max_length=arguments.max_length if arguments.max_length > 0 else None,
            move=arguments.move,
            debug=arguments.debug,
            mutate=arguments.mutate,
            force_repairing=arguments.force_repairing,
            test_split_name=arguments.test_split_name if len(arguments.test_split_name) > 0 else None,
            task_id=arguments.task_id if len(arguments.task_id) > 0 else os_getenv('SLURM_JOB_ID', None),
            task_name=arguments.task_name if len(arguments.task_name) > 0 else os_getenv('SLURM_JOB_NAME', None),
            wandb=arguments.wandb if len(arguments.wandb) > 0 else os_getenv('WANDB_API_KEY', None),
            feature_model=arguments.feature_model,
            forging_policy=arguments.forging_policy if len(arguments.forging_policy) > 0 else None,
            strict_evaluation=not arguments.permissive_evaluation,
        )

    def run_repair(self, inputs: list[ValidationResults] | list[FeatureResults]):
        passing, failing, repair_rate = self._validator.repair_set(inputs,
                                                                   trials=self._arguments.trials,
                                                                   train_epochs=self._arguments.epochs,
                                                                   debug=self._arguments.debug,
                                                                   mutate=self._arguments.mutate,
                                                                   early_stopping=not self._arguments.force_repairing)
        trials_distribution = Counter([v.used_trials for v in passing])
        return repair_rate, trials_distribution, passing, failing

    def forge_name(self) -> str:
        subject_name = self._arguments.subject if len(self._arguments.subject) > 0 else "modelizer"
        subject_name = f"{subject_name}_{self._arguments.model_type}"
        return subject_name

    def __preprocess_data__(self, data: Union[Sequence[Tuple[str, str, str]], Iterable[Tuple[str, str, str]], list[FeatureResults]]) -> list[ValidationResults] | list[FeatureResults]:
        processed_data = [d for d in data if not d.is_equal(strict=self._arguments.strict_evaluation)]
        if 0 < self.arguments.num_samples < len(processed_data):
            random_seed(self._arguments.seed)
            processed_data = random_sample(processed_data, self.arguments.num_samples)
        self.logger.info(f"Dataset processed. Collected {len(processed_data)} records.")
        return processed_data

    def preprocess_data(self,  data: Union[Sequence[Tuple[str, str, str]], Iterable[Tuple[str, str, str]]]) -> list[ValidationResults]:
        """
        Prepare data for repair process.
        :param data: Sequence or Iterable of tuples (input_text, expected_text, predicted_text)
        :return: List of ValidationResults objects ready for repair
        """
        backward_model = self._validator.model.engine.config.backward
        input_tokenizer = self._validator.model.engine.tokenizer
        output_tokenizer = self._validator.model.engine.output_tokenizer
        self.logger.info(f"Processing samples...")
        processed_data = []
        for input_text, expected_text, predicted_text in data:
            tokenized_input = input_tokenizer(input_text, return_tensors=False)["input_ids"]
            tokenized_expected = output_tokenizer(expected_text, return_tensors=False)["input_ids"]
            tokenized_predicted = output_tokenizer(predicted_text, return_tensors=False)["input_ids"]
            processed_data.append(ValidationResults(
                backward_mode=backward_model,
                model_input=input_text,
                model_input_tokens=tokenized_input,
                model_output=predicted_text,
                model_output_tokens=tokenized_predicted,
                ground_truth=expected_text,
                ground_truth_tokens=tokenized_expected,
            ))
        return self.__preprocess_data__(processed_data)

    def preprocess_feature_data(self, data: list[FeatureResults]) -> list[FeatureResults]:
        self.logger.info(f"Processing feature samples...")
        return self.__preprocess_data__(data)

    def repair(self, inputs: list[ValidationResults] | list[FeatureResults]):
        if len(inputs) > 0:
            self.logger.info("Starting repair process...")
            start_time = datetime.now()
            output_dir = self._arguments.repair_results_dir / "model" if self._arguments.move and self._arguments.repair_results_dir is not None else None
            passing, failing, repair_rate = self._validator.repair_set(inputs,
                                                                       trials=self._arguments.trials,
                                                                       train_epochs=self._arguments.epochs,
                                                                       debug=self._arguments.debug,
                                                                       mutate=self._arguments.mutate,
                                                                       strict=self._arguments.strict_evaluation,
                                                                       output_dir=output_dir,
                                                                       early_stopping=not self._arguments.force_repairing)
            total = len(passing) + len(failing)
            trials_distribution = Counter([v.used_trials for v in passing])
            self.logger.info(f"Repair Completed in {get_time_diff(start_time)}.\nRepaired {len(passing)} out of {total}."
                             f"\nRepair Rate: {repair_rate:.2f}\nTrials Distribution: {trials_distribution}")
            scores = repair_rate, trials_distribution
            results = passing, failing
            Pickle.dump(scores, self.arguments.repair_results_dir / f"{self.arguments.subject}_model_repair_scores.pkl")
            Pickle.dump(results, self.arguments.repair_results_dir / f"{self.arguments.subject}_model_repair_results.pkl")

            if self.arguments.move:
                DataHandlers.move_to_temp_folder(self.arguments.repair_results_dir, subject_name=self.forge_name())
        else:
            self.logger.info(f"Aborting. Nothing to repair.")

    def validate(self, inputs: Iterable[Any] | Sequence[Any], max_length: Optional[int] = None) -> Iterable[Any]:
        return self._validator.validate_set(inputs, max_length=max_length, stop_on_exception=False)
