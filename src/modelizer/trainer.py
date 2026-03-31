import random
import inspect
import argparse

from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass
from os import getenv as os_getenv
from importlib import import_module
from typing import Optional, Any, Callable, Union, List, Tuple, Type, Dict, Set

from tqdm.auto import tqdm
from shutil import rmtree as shutil_rmtree
from pandas import DataFrame, Series, concat, read_csv

from modelizer.utils import (
    Pickle,
    Logger,
    LoggerConfig,
    DataHandlers,
    SingletonMeta,
    get_time_diff,
    retrieve_init_arguments,
)

from modelizer.configs import SEED
from modelizer.learner import Modelizer
from modelizer.models import BaseConfig
from modelizer.generators import BaseSubject
from modelizer.tokenizers import DummyTokenizer
from modelizer.metrics import FeatureResults, FeatureMetrics
from modelizer.tokenizers import BaseTokenizer, SentencePieceTokenizer


########################################################################################################################
#                                    Class that utility functions for model training                                   #
########################################################################################################################
@dataclass
class TrainArguments:
    """
    Data class that encapsulates training configuration parameters.

    Attributes:
        dataset (str): Path to a file containing training and test data.
        subject (str | None): Optional subject name for the dataset.
        subject_instance (BaseSubject | None): Optional subject instance for specialized processing.
        root_dir (str | None): Optional root directory for the model, tokenizers, results.
        subset_size (int): Subset size for the dataset to use. Ignored if less or equal to 0.
        test_size (int): Size of the test subset.
        source (str): Identifier for the source datatype.
        target (str): Identifier for the target datatype.
        trials (int): Number of trials for hyperparameter optimization.
        test_epochs (int): Number of epochs for hyperparameter evaluation.
        train_epochs (int): Number of epochs for model training.
        batch_size (int): Batch size used during training.
        seed (int): Seed value for reproducibility.
        backward (bool): Flag indicating if backward model training is enabled.
        move (bool): Flag to move model and results to a temporary directory.
        cleanup (bool): Flag to force the cleanup of the temporary directory.
        no_checkpointing (bool): Flag to indicate that intermediate checkpoints should not be saved.
        no_optimization (bool): Flag to indicate that no hyperparameter optimization should be performed.
        fast (bool): Flag to indicate if trainer should reduce number of samples for fast hyperparameter optimization.
        test_early (bool): Flag to indicate testing after every training epoch.
        use_legacy (bool): Flag to indicate the usage of the legacy Modelizer v1 engine.
        use_flash (bool): Flag to indicate usage of flash attention.
        use_cpu (bool): Flag to force loading model to CPU.
        compile_model (bool): Flag to enable model compilation.
        split_input (bool): Flag to enable input splitting on character base.
        reduce_memory_usage (bool): Flag to reduce memory consumption by enabling model quantization in LLMs.
        reduce_spaces (bool): Flag to reduce number of spaces in the source and target columns.
        wandb (str | None): Weight and Biases API token.
        model_type (str): Type of the model to be used.
        task_id (str | None): Optional task identifier.
        task_name (str | None): Optional task name.
        kwargs (dict | None): Additional keyword arguments for training configuration.
        output_dir (Path | None): Directory to save the trained model.
        results_dir (Path | None): Directory to save the training results.
        split_dataset_path (Path | None): Optional path to save the prepared train/test datasets.
        comparator_func: Optional function to compare expected and predicted values
        post_formating: Function to post-process program output for processing with Modelizer
        max_usable_memory: Optional maximum usable memory in megabytes for model training
    """

    dataset: str
    subset_size: int
    test_size: int
    source: str
    target: str
    trials: int
    test_epochs: int
    train_epochs: int
    batch_size: int
    seed: int
    backward: bool
    wandb: Optional[str] = None                                                 # sensitive information
    use_legacy: bool = False
    use_flash: bool = True
    use_cpu: bool = False
    compile_model: bool = False
    split_input: bool = False
    move: bool = False
    fast: bool = False
    test_early: bool = False
    cleanup: bool = False
    no_checkpointing: bool = False
    no_optimization: bool = False
    reduce_memory_usage: bool = False
    update_dataset: bool = False
    reduce_spaces: bool = False
    subject: Optional[str] = None
    subject_instance: Optional[BaseSubject] = None
    model_type: str = "generic"
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    kwargs: Optional[dict[str, Any]] = None
    root_dir: Optional[str | Path] = None
    output_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    split_dataset_path: Optional[Path | str] = None
    comparator_func: Optional[Callable[[Any, Any], Any]] = None
    post_formating: Callable[[Any], Any] = lambda x: x
    legacy_padding_mode: bool = False
    max_usable_memory: int = 0  # in megabytes, 0 means no limit
    startup_arguments: Optional[str] = retrieve_init_arguments()
    source_tokenizer_class: Optional[Type[BaseTokenizer]] = SentencePieceTokenizer
    target_tokenizer_class: Optional[Type[BaseTokenizer]] = SentencePieceTokenizer

    def __post_init__(self):
        self.root_dir = DataHandlers.locate_temp_dir() if self.move else Path.cwd() if self.root_dir is None else Path(self.root_dir).resolve()
        self.output_dir = self.root_dir.joinpath("models")
        self.results_dir = self.root_dir.joinpath("results")

        if self.subject is not None and len(self.subject) > 0:
            self.output_dir = self.output_dir.joinpath(self.subject)
            self.results_dir = self.results_dir.joinpath(self.subject)

        if self.split_dataset_path is not None:
            self.split_dataset_path = Path(self.split_dataset_path).resolve()

        if self.post_formating is not None:
            assert callable(self.post_formating), "post_formating must be a callable function."
            signature = inspect.signature(self.post_formating)
            positional_params = [
                p for p in signature.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            required = [p for p in positional_params if p.default is inspect.Parameter.empty]
            assert len(required) == 1, "post_formating function must accept exactly one required positional argument."
        else:
            self.post_formating = lambda x: x

        if self.use_legacy:
            self.model_type = "legacy"

        if self.kwargs is None:
            self.kwargs = {
                "synced": False,
                "feature_encoding": "positive",
                "feature_forging": "sparse",
                "feature_forging_second_policy": None,
                "train_feature_model": True,
                "test_forging_policies": False,
                "max_features_samples": 5,
                "max_features_attempts": 5,
                "max_features_mutations": 5,
            }
        if self.cleanup:
            if self.output_dir.exists():
                shutil_rmtree(self.output_dir)
            if self.results_dir.exists():
                shutil_rmtree(self.results_dir)

    def __str__(self) -> str:
        sensitive_fields = {"huggingface_key", "openai_key", "claude_key", "wandb", "source_tokenizer_class", "target_tokenizer_class"}
        return "TrainArguments:\n" + "\n".join([f"{key}: {value}" for key, value in vars(self).items() if key not in sensitive_fields])

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.subject_instance is not None:
            state['subject_instance'] = Pickle.to_bytes(self.subject_instance)
        if self.comparator_func is not None:
            state['comparator_func'] = Pickle.to_bytes(self.comparator_func)
        state['post_formating'] = Pickle.to_bytes(self.post_formating)
        if state['source_tokenizer_class'] is not None:
            state['source_tokenizer_class'] = (self.source_tokenizer_class.__module__, self.source_tokenizer_class.__name__)
        if state['target_tokenizer_class'] is not None:
            state['target_tokenizer_class'] = (self.target_tokenizer_class.__module__, self.target_tokenizer_class.__name__)
        return state

    def __setstate__(self, state):
        if state['subject_instance'] is not None and isinstance(state['subject_instance'], bytes):
            state['subject_instance'] = Pickle.from_bytes(state['subject_instance'])
        if state['comparator_func'] is not None and isinstance(state['comparator_func'], bytes):
            state['comparator_func'] = Pickle.from_bytes(state['comparator_func'])
        if state['post_formating'] is not None and isinstance(state['post_formating'], bytes):
            state['post_formating'] = Pickle.from_bytes(state['post_formating'])
        if state.get('source_tokenizer_class') is not None:
            module_name, class_name = state['source_tokenizer_class']
            state['source_tokenizer_class'] = getattr(import_module(module_name), class_name)
        if state.get('target_tokenizer_class') is not None:
            module_name, class_name = state['target_tokenizer_class']
            state['target_tokenizer_class'] = getattr(import_module(module_name), class_name)
        self.__dict__.update(state)

    @staticmethod
    def forge(state: dict) -> 'TrainArguments':
        arguments = TrainArguments(**state)
        if arguments.subject_instance is not None and isinstance(arguments.subject_instance, bytes):
            arguments.subject_instance = Pickle.to_bytes(arguments.subject_instance)
        if arguments.comparator_func is not None and isinstance(arguments.comparator_func, bytes):
            arguments.comparator_func = Pickle.to_bytes(arguments.comparator_func)
        if arguments.post_formating is not None and isinstance(arguments.post_formating, bytes):
            arguments.post_formating = Pickle.to_bytes(arguments.post_formating)
        return arguments

    # python
    @staticmethod
    def get_sample_arguments() -> "TrainArguments":
        """
        Create a reference TrainArguments instance with safe, minimal defaults.
        Useful for generating example YAML or bootstrapping configs.
        """
        return TrainArguments(
            dataset="./data/sample.csv",
            subject="demo",
            root_dir="./.modelizer",
            subset_size=1000,
            test_size=200,
            source="Input",
            target="Expected",
            trials=10,
            test_epochs=2,
            train_epochs=3,
            batch_size=8,
            seed=SEED,
            backward=False,
            # runtime flags
            move=False,
            cleanup=False,
            fast=True,
            test_early=False,
            no_checkpointing=True,
            no_optimization=False,
            reduce_memory_usage=False,
            reduce_spaces=False,
            split_input=False,
            update_dataset=False,
            # hardware/model options
            use_flash=True,
            use_cpu=False,
            compile_model=False,
            model_type="encoder-decoder",
            # integrations and checkpoints (no secrets)
            wandb=None,
            # optional/advanced
            task_id=None,
            task_name=None,
            subject_instance=None,
            comparator_func=None,
            max_usable_memory=0,
            split_dataset_path=None,
            legacy_padding_mode=False,
            # feature-model defaults
            kwargs={
                "synced": False,
                "feature_encoding": "positive",
                "feature_forging": "sparse",
                "feature_forging_second_policy": None,
                "train_feature_model": False,
                "test_forging_policies": False,
                "max_features_samples": 5,
                "max_features_attempts": 5,
                "max_features_mutations": 5,
            },
        )

    def save_to_pickle(self, filepath: str | Path):
        Pickle.dump(self, filepath)


class Trainer:
    def __init__(self, arguments: TrainArguments | Path | str | None = None):
        if arguments is None:
            self._arguments = self.parse_arguments()
        elif isinstance(arguments, TrainArguments):
            self._arguments = arguments
        else:
            self._arguments = self.load_arguments(arguments)

        self._arguments.output_dir.mkdir(parents=True, exist_ok=True)
        self._arguments.results_dir.mkdir(parents=True, exist_ok=True)
            
        self._logger = Logger(LoggerConfig(root_dir=self._arguments.output_dir, log_to_wandb=self._arguments.wandb is not None))
        self._init_time = datetime.now()
        self.get_time_diff = get_time_diff
        self._model: Optional[Modelizer] = None
        self._logger.info("Trainer initialized.")
        self._logger.info(str(self._arguments))

    @property
    def arguments(self) -> TrainArguments:
        return self._arguments
    
    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def model(self) -> Modelizer | None:
        return self._model

    @model.setter
    def model(self, model: Modelizer):
        assert isinstance(model, Modelizer), f"model must be a Modelizer instance"
        self._model = model

    @staticmethod
    def __init_argument_parser__() -> argparse.ArgumentParser:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--dataset', type=str, required=True, help="Path to a file with train and test data")
        arg_parser.add_argument('--source', type=str, required=True, help="Name of the source datatype")
        arg_parser.add_argument('--target', type=str, required=True, help="Name of the target datatype")
        arg_parser.add_argument('--subject', type=str, default="", help="Optional subject name")
        arg_parser.add_argument('--root-dir', type=str, default="", help="Optional root directory for model, tokenizers, results")
        arg_parser.add_argument('--subset-size', type=int, default=0, help="Subset size for the training dataset to use. 0 means no subset")
        arg_parser.add_argument('--test-size', type=int, default=10000, help="Size of the test subset. 0 means no test")
        arg_parser.add_argument('--trials', type=int, default=100, help="Number of trials for hyperparameter optimization")
        arg_parser.add_argument('--test-epochs', type=int, default=5, help="Number of epochs for hyperparameter optimization")
        arg_parser.add_argument('--train-epochs', type=int, default=10, help="Number of epochs for model training")
        arg_parser.add_argument('--batch-size', type=int, default=1, help="Batch size for training")
        arg_parser.add_argument('--seed', type=int, default=SEED, help="Seed for reproducibility")
        arg_parser.add_argument('--move', action='store_true', help='Move model with results to the TEMP directory')
        arg_parser.add_argument('--cleanup', action='store_true', help='Clean up of the TEMP directory')
        arg_parser.add_argument('--fast', action='store_true', help='Flag to indicate if trainer should reduce number of samples for fast hyperparameter optimization.')
        arg_parser.add_argument('--test-early', action='store_true', help='Flag to indicate testing after every training epoch.')
        arg_parser.add_argument('--update-dataset-with-executions', action='store_true', help='Flag to enable dataset update with program executions at dataset loading time.')
        arg_parser.add_argument('--backward', action='store_true', help='Flag to indicate training of backward model')
        arg_parser.add_argument('--reduce-memory-usage', action='store_true', help='Flag to reduce memory consumption by enabling model quantization in LLMs')
        arg_parser.add_argument('--reduce-spaces', action='store_true', help='Flag to reduce number of spaces in the source and target columns. ')
        arg_parser.add_argument('--model-type', type=str, default="encoder-decoder", help="Type of the model to be used")
        arg_parser.add_argument('--use-vanilla', action='store_true', help='Flag to indicate the usage of vanilla models')
        arg_parser.add_argument('--use-legacy', action='store_true', help='Flag to indicate the usage of the legacy Modelizer v1 engine')
        arg_parser.add_argument('--use-flash', action='store_true', help='Flag to indicate the usage of flash attention')
        arg_parser.add_argument('--no-checkpointing', action='store_true', help='Flag to indicate that intermediate checkpoints should not be saved')
        arg_parser.add_argument('--no-optimization', action='store_true', help='Flag to indicate that no hyperparameter optimization should be performed')
        arg_parser.add_argument('--use-cpu', action='store_true', help='Flag to force loading model to CPU')
        arg_parser.add_argument('--compile-model', action='store_true', help='Flag to enable model compilation')
        arg_parser.add_argument('--split-input', action='store_true', help='Flag to enable input splitting on character base.')
        arg_parser.add_argument('--synced', action='store_true', help="Optional flag to indicate that the model is running in synced mode")
        arg_parser.add_argument('--distributed', type=str, default="none", help="Name of the target datatype. Could be 'dp', 'ddp', 'fsdp' or 'none'")
        arg_parser.add_argument('--wandb', type=str, default="", help="Weight and Biases API Token")
        arg_parser.add_argument('--task-id', type=str, default="", help="Optional task identifier")
        arg_parser.add_argument('--task-name', type=str, default="", help="Optional task identifier")
        arg_parser.add_argument('--max-usable-memory', type=int, default=0, help="Maximum usable memory in megabytes for model training. 0 means no limit. Setting this parameter impacts hyperparameter optimization")
        arg_parser.add_argument('--use-legacy-padding-mode', action='store_true', help='Flag to indicate replacement of padding token with eos token')
        # Feature models training arguments
        arg_parser.add_argument("--train-feature-model", action="store_true", help="Optional flag to indicate that the model should be trained as a feature-based model")
        arg_parser.add_argument("--feature-encoding", type=str, default="positive", help="Optional feature encoding mode. Could be 'positive', 'non-negative', or 'full'")
        arg_parser.add_argument("--feature-forging", type=str, default="sparse", help="Optional feature forging mode. Could be 'sparse', 'unset', 'random', 'reference', 'mutations")
        arg_parser.add_argument("--feature-forging-second-policy", type=str, default="none", help="Optional second feature forging mode. Could be 'none', 'sparse', 'unset', 'random', 'reference', 'mutations")
        arg_parser.add_argument("--test-forging-policies", action="store_true", help="Optional flag to indicate that the model should test different feature forging policies during training")
        arg_parser.add_argument("--max-features-samples", type=int, default=5, help="Maximum number of features to sample for feature-based training. Used only for feature-based training")
        arg_parser.add_argument("--max-features-attempts", type=int, default=5, help="Number of attempts to sample unique features. Used only for feature-based training")
        arg_parser.add_argument("--max-features-mutations", type=int, default=5, help="Maximum number of mutations to apply to the feature vectors. Used only for feature-based training")
        arg_parser.add_argument("--split-dataset-path", type=str, default="", help="Optional path to save the prepared train / test datasets")
        return arg_parser

    @staticmethod
    def print_help():
        arg_parser = Trainer.__init_argument_parser__()
        arg_parser.print_help()

    @staticmethod
    def parse_arguments() -> TrainArguments:
        arg_parser = Trainer.__init_argument_parser__()
        arguments = arg_parser.parse_args()

        assert isinstance(arguments.source, str) and len(arguments.source) > 0, "Source must be a non-empty string"
        assert isinstance(arguments.target, str) and len(arguments.target) > 0, "Target must be a non-empty string"
        assert isinstance(arguments.dataset, str) and len(arguments.dataset) > 0, "Dataset must be a non-empty string"
        assert isinstance(arguments.trials, int) and arguments.trials > 0, "Hyperparameter test trials must be a positive integer"
        assert isinstance(arguments.test_epochs, int) and arguments.test_epochs > 0, "Test epochs must be a positive integer"
        assert isinstance(arguments.train_epochs, int) and arguments.train_epochs > 0, "Train epochs must be a positive integer"
        assert isinstance(arguments.batch_size, int) and arguments.batch_size > 0, "Batch size must be a positive integer"
        assert arguments.feature_encoding in ("positive", "non-negative", "full"), "Feature encoding must be 'positive', 'non-negative', or 'full'"
        assert arguments.feature_forging in ("sparse", "unset", "random", "reference"), "Feature forging must be 'sparse', 'unset', 'random', or 'reference'"

        return TrainArguments(
            dataset=arguments.dataset,
            subject=arguments.subject if len(arguments.subject) > 0 else None,
            root_dir=arguments.root_dir if len(arguments.root_dir) > 0 else None,
            subset_size=arguments.subset_size,
            test_size=arguments.test_size,
            source=arguments.source,
            target=arguments.target,
            trials=arguments.trials,
            test_epochs=arguments.test_epochs,
            train_epochs=arguments.train_epochs,
            use_flash=arguments.use_flash,
            use_cpu=arguments.use_cpu,
            compile_model=arguments.compile_model,
            reduce_memory_usage=arguments.reduce_memory_usage,
            reduce_spaces=arguments.reduce_spaces,
            split_input=arguments.split_input,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
            move=arguments.move,
            cleanup=arguments.cleanup,
            fast=arguments.fast,
            test_early=arguments.test_early,
            backward=arguments.backward,
            no_checkpointing=arguments.no_checkpointing,
            no_optimization=arguments.no_optimization,
            legacy_padding_mode=arguments.use_legacy_padding_mode,
            update_dataset=arguments.update_dataset_with_executions,
            model_type=arguments.model_type.lower().replace("-", "").replace("_", ""),
            wandb=arguments.wandb if len(arguments.wandb) > 0 else os_getenv('WANDB_API_KEY', None),
            task_id=arguments.task_id if len(arguments.task_id) > 0 else os_getenv('SLURM_JOB_ID', None),
            task_name=arguments.task_name if len(arguments.task_name) > 0 else os_getenv('SLURM_JOB_NAME', None),
            split_dataset_path=arguments.split_dataset_path if len(arguments.split_dataset_path) > 0 else None,
            kwargs={
                "synced": arguments.synced,
                "feature_encoding": arguments.feature_encoding,
                "feature_forging": arguments.feature_forging,
                "feature_forging_second_policy": None if arguments.feature_forging_second_policy == "none" else arguments.feature_forging_second_policy,
                "train_feature_model": arguments.train_feature_model,
                "test_forging_policies": arguments.test_forging_policies,
                "max_features_samples": arguments.max_features_samples,
                "max_features_attempts": arguments.max_features_attempts,
                "max_features_mutations": arguments.max_features_mutations,
            }
        )

    @staticmethod
    def load_arguments(filepath: str | Path) -> TrainArguments:
        """
        Load arguments from a file or string.
        :param filepath: Path to the file or string containing the arguments
        :return: Parsed TrainArguments object
        """
        filepath = Path(filepath)
        assert filepath.is_file(), "args must be a file"
        if filepath.suffix in (".pkl", ".pickle"):
            args = Pickle.load(filepath)
        else:
            raise ValueError("Unsupported file format for arguments. Supported formats are .pkl, .pickle, .yml, .yaml")
        assert isinstance(args, TrainArguments), "args must be a TrainArguments object"
        return args

    def save_arguments(self, filepath: str | Path):
        """
        Save arguments to a file.
        :param filepath: Path to the file where the arguments will be saved
        """
        assert isinstance(filepath, str | Path), "filepath must be a string or Path object"
        Pickle.dump(self._arguments, filepath)

    def reload_model(self, directory: str | Path):
        assert isinstance(directory, str | Path), "directory must be a string or Path object"
        directory = Path(directory)
        assert directory.exists(), "directory does not exist"
        assert directory.is_dir(), "directory is not a directory"
        self._model = Modelizer(directory, wandb_token=self._arguments.wandb)

    def __run_subject_processing__(self, name, df, policy, feature_tokenizer, program_input_tokenizer):
        if hasattr(feature_tokenizer, policy):
            feature_tokenizer.forging = policy
        if hasattr(program_input_tokenizer, policy):
            program_input_tokenizer.forging = policy
        test_name = f"{name}_{policy}"
        test_results, _ = self._model.test(df, test_name, max_length=None, save_results=False, get_metrics=False)
        self.process_results_with_subject(test_results, feature_tokenizer, program_input_tokenizer, test_name)

    def evaluate_feature_dataset(self, name: str, df: DataFrame):
        """
        Evaluate the feature model on a given dataset and log the results.
        :param name: name of the dataset
        :param df: DataFrame containing the dataset
        """
        assert self._model is not None, ("model must be instantiated before running evaluation. "
                                         "Either train model or use reload_model method to reinitialize the model.")

        self._logger.info(f"Test dataset '{name}' shape: {df.shape}")
        start_time = datetime.now()

        if self._arguments.backward:
            feature_tokenizer = self._model.engine.tokenizer
            program_input_tokenizer = self._model.engine.output_tokenizer
        else:
            feature_tokenizer = self._model.engine.output_tokenizer
            program_input_tokenizer = self._model.engine.tokenizer

        if self._arguments.kwargs["test_forging_policies"]:
            backup_feature_policy = feature_tokenizer.forging if hasattr(feature_tokenizer, "forging") else None
            backup_program_policy = program_input_tokenizer.forging if hasattr(program_input_tokenizer, "forging") else None
            for policy in ("sparse", "unset", "random", "reference", "mutations"):
                self.__run_subject_processing__(name, df, policy, feature_tokenizer, program_input_tokenizer)

            if backup_feature_policy is not None:
                feature_tokenizer.forging = backup_feature_policy
            if backup_program_policy is not None:
                program_input_tokenizer.forging = backup_program_policy
        elif self._arguments.kwargs["feature_forging_second_policy"] is not None:
            backup_feature_policy = feature_tokenizer.forging if hasattr(feature_tokenizer, "forging") else None
            backup_program_policy = program_input_tokenizer.forging if hasattr(program_input_tokenizer, "forging") else None
            for policy in (self._arguments.kwargs["feature_forging"], self._arguments.kwargs["feature_forging_second_policy"]):
                self.__run_subject_processing__(name, df, policy, feature_tokenizer, program_input_tokenizer)

            if backup_feature_policy is not None:
                feature_tokenizer.forging = backup_feature_policy
            if backup_program_policy is not None:
                program_input_tokenizer.forging = backup_program_policy
        else:
            test_results, _ = self._model.test(df, name, max_length=None, save_results=False, get_metrics=False)
            self.process_results_with_subject(test_results, feature_tokenizer, program_input_tokenizer, name)

        self._logger.info(f"Evaluation of '{name}' completed in {get_time_diff(start_time)}")

    def process_results_with_subject(self, predictions_df: DataFrame, feature_tokenizer: BaseTokenizer, program_input_tokenizer: BaseTokenizer, name: str = "auto",) -> DataFrame:
        """
        Process the results DataFrame using the subject instance and log the evaluation results.
        :param predictions_df: DataFrame containing the predictions to be processed
        :param feature_tokenizer: Tokenizer for the program execution features
        :param program_input_tokenizer: Tokenizer for the program inputs
        :param name: Name of the dataset for logging purposes
        :return:
        """
        subject: BaseSubject = self._arguments.subject_instance
        assert subject is not None and isinstance(subject, BaseSubject), f"subject must an instance of BaseSubject, got {type(subject)} => {subject}"

        # DF Columns Input / Expected / Predicted
        results = []
        if self._arguments.backward:
            # This mode requires subject execution for results comparison. Compare model input to monitored behavior
            for _, row in predictions_df.iterrows():
                inp = row["Input"]
                predicted = row["Predicted"]
                monitored = subject.execute(predicted)
                monitored = self.arguments.post_formating(monitored)
                prog_inp_tokens = program_input_tokenizer.tokenize_no_specials(predicted, to_string_tokens=True)
                monitored_tokens = feature_tokenizer.tokenize_no_specials(monitored, to_string_tokens=True)
                feature_tokens = feature_tokenizer.tokenize_no_specials(inp, to_string_tokens=True)
                results.append(FeatureResults(program_input=predicted, features=inp,
                                              monitored=monitored, is_backward=True,
                                              input_tokens=prog_inp_tokens, feature_tokens=feature_tokens, monitored_tokens=monitored_tokens))
        else:
            # This mode could benefit from direct comparison of Input / Expected and Predicted if Expected is available
            for _, row in predictions_df.iterrows():
                inp = row["Input"]
                predicted = row["Predicted"]
                monitored = row["Expected"]
                if not len(monitored):
                    monitored = subject.execute(inp)
                    monitored = self.arguments.post_formating(monitored)
                prog_inp_tokens = program_input_tokenizer.tokenize_no_specials(inp, to_string_tokens=True)
                monitored_tokens = feature_tokenizer.tokenize_no_specials(monitored, to_string_tokens=True)
                feature_tokens = feature_tokenizer.tokenize_no_specials(predicted, to_string_tokens=True)
                results.append(FeatureResults(program_input=inp, features=predicted,
                                              monitored=monitored, is_backward=False,
                                              input_tokens=prog_inp_tokens, feature_tokens=feature_tokens, monitored_tokens=monitored_tokens))

        results_df = FeatureMetrics.to_computed_results(results, subject.comparator if subject.comparator is not None else self._arguments.comparator_func)

        # save results as .pkl file
        Pickle.dump(results, self._arguments.results_dir / f"evaluation_results_{name}.pkl")

        # save processed results as a .csv file
        results_df.to_csv(self._arguments.results_dir / f"evaluation_processed_{name}.csv", index=False)

        # compute metrics and save as .csv file
        metrics_df = FeatureMetrics.compute_metrics(results_df)
        metrics_df.to_csv(self._arguments.results_dir / f"evaluation_metrics_{name}.csv", index=False)

        return results_df

    def execute(self,
                config: Optional[BaseConfig | str],
                train_data: DataFrame,
                test_data: Union[DataFrame, List[Tuple[str, DataFrame]], Dict[str, DataFrame]],
                tokenizer: Optional[BaseTokenizer] = None,
                output_tokenizer: Optional[BaseTokenizer] = None):
        """
        Execute the training and evaluation of the model.
        :param config: Configuration for the model.
          If None, it will be forged based on the arguments.
          If a string is provided, it will be used to determine the model type.
          If an instance of BaseConfig is provided, it will be used directly.
        :param train_data: DataFrame containing the training data
        :param test_data: DataFrame or list of tuples containing the test data
        :param tokenizer: Optional tokenizer for the input data
        :param output_tokenizer: Optional tokenizer for the output data
        """
        assert not train_data.empty, "Training DataFrame cannot be empty"
        start_time = datetime.now()
        train_data = train_data.drop_duplicates(keep="first")

        # Test datasets preparation
        if isinstance(test_data, DataFrame):
            test_datasets: List[Tuple[str, DataFrame]] = [("auto", test_data.drop_duplicates(keep="first"))]
        elif isinstance(test_data, dict) and len(test_data) > 0:
            test_datasets: List[Tuple[str, DataFrame]] = [(name, df.drop_duplicates(keep="first")) for name, df in test_data.items()]
        elif isinstance(test_data, list) and len(test_data) > 0:
            test_datasets: List[Tuple[str, DataFrame]] = [(name, df.drop_duplicates(keep="first")) for name, df in test_data]
        else:
            test_datasets: List[Tuple[str, DataFrame]] = []

        if config is None or isinstance(config, str):
            match_target = config if isinstance(config, str) else self._arguments.model_type.lower()
            match_target = match_target.lower().replace("_", "").replace("-", "")

            match match_target:
                case "legacy" | "v1" | "legacymodel":
                    config = ConfigForger.forge_legacy_config(self._arguments)
                case "encoderdecoder" | "encdec":
                    config = ConfigForger.forge_encoder_decoder_config(self._arguments)
                case _:
                    raise ValueError(f"Invalid model type: {self._arguments.model_type}")

        if self._arguments.no_checkpointing:
            config.total_save_limit = 1

        config.make_cross_platform_compatible()

        self._model = Modelizer(config, wandb_token=self._arguments.wandb,
                                tokenizer=tokenizer, output_tokenizer=output_tokenizer,
                                logger=self._logger, late_init=False, track_memory_usage=True)

        # --- Training ---
        if not self._arguments.no_optimization:  # with hyperparameter optimization
            if self._arguments.fast:
                if self._arguments.kwargs.get("train_feature_model", False):
                    optimize_df = train_data.sample(n=min(2500, len(train_data), sum([len(df) for _, df in test_datasets])), random_state=self._arguments.seed)
                else:
                    optimize_df = concat([train_data] + [df for _, df in test_datasets])
                    optimize_df = optimize_df.sample(n=min(2500, len(optimize_df)), random_state=self._arguments.seed)
            else:
                if self._arguments.kwargs.get("train_feature_model", False):
                    optimize_df = train_data.sample(n=min(len(train_data), sum([len(df) for _, df in test_datasets])), random_state=self._arguments.seed)
                else:
                    optimize_df = concat([df for _, df in test_datasets])

            optimize_df = optimize_df.drop_duplicates(keep="first")
            optimize_df.reset_index(drop=True, inplace=True)

            self._logger.info(f"Dataset for hyperparameter optimization: {optimize_df.shape}")
            config = self._model.optimize(self._arguments.trials, optimize_df, None, self._arguments.test_epochs, self._arguments.batch_size, reset=False)
            self._logger.info(f"Modelizer optimized model parameters in {get_time_diff(start_time)}")
            self._logger.info(f"Train dataset shape: {train_data.shape}")

            train_incomplete = True
            while train_incomplete and config.have_more_trials:
                try:
                    self._logger.info("Initializing modelizer with optimized parameters")
                    self._model.__reset__(config)
                    self._logger.info(f"\n{str(self._model)}")
                    if self.arguments.test_early:
                        self.model.engine.enable_epochwise_checkpointing()
                        for epoch_id in range(1, self._arguments.train_epochs + 1):
                            self._model.train(train_data, 1, self._arguments.batch_size)
                            self.run_testing([(f"{name}_epoch{epoch_id}", ds) for name, ds in test_datasets])
                        self.model.engine.disable_epochwise_checkpointing()
                    else:
                        self._model.train(train_data, self._arguments.train_epochs, self._arguments.batch_size)
                except RuntimeError:
                    self._logger.error(f"Failed to train the model with optimized parameters:\n{str(config)}")
                    self._logger.info(f"Retrying with the next best hyperparameters...")
                    train_incomplete = True
                else:
                    train_incomplete = False
        else:                                   # without hyperparameter optimization
            self._logger.info("Skipping hyperparameter optimization")
            self._logger.info(f"\n{str(self._model)}")
            self._model.train(train_data, self._arguments.train_epochs, self._arguments.batch_size)

        self._logger.info(f"Peak training memory usage: {self._model.engine.config.memory_requirements}")

        # --- Evaluation ---
        if not self.arguments.test_early and len(test_datasets) > 0:
            self.run_testing(test_datasets)

        # --- Finalization ---
        self._model.finalize()
        self._logger.info(f"Peak memory usage: {self._model.engine.config.memory_requirements}")
        self._logger.info(f"Total runtime: {get_time_diff(self._init_time)}")

        if self._arguments.move:
            self._logger.info(f"Moving model and results to TEMP folder")
            self._arguments.model_type = config.__class__.__name__.lower().replace("config", "")
            subject_name = self._arguments.subject if len(self._arguments.subject) > 0 else "modelizer"
            subject_name = f"{subject_name}_{self._arguments.model_type}_{self._arguments.source}_{self._arguments.target}_{self._arguments.backward}"
            DataHandlers.move_to_temp_folder(self._arguments.output_dir, subject_name=subject_name)
            DataHandlers.move_to_temp_folder(self._arguments.results_dir, subject_name=subject_name)

    def run_testing(self, test_datasets: List[Tuple[str, DataFrame]]):
        if self._arguments.kwargs.get("train_feature_model", False) and self._arguments.subject_instance is not None:
            for name, df in test_datasets:
                if df.empty:
                    self._logger.warning(f"Test dataset '{name}' is empty. Skipping evaluation.")
                else:
                    self.evaluate_feature_dataset(name, df)
        else:
            for name, df in test_datasets:
                if df.empty:
                    self._logger.warning(f"Test dataset '{name}' is empty. Skipping evaluation.")
                else:
                    self._logger.info(f"Test dataset '{name}' shape: {df.shape}")
                    start_time = datetime.now()
                    self._model.test(df, name, max_length=None, save_results=True, output_dir=self._arguments.results_dir)
                    self._logger.info(f"Evaluation of '{name}' completed in {get_time_diff(start_time)}")

    def update_dataset_with_subject_executions(self, data: DataFrame | dict[str, list[Any]]) -> DataFrame:
        assert self._arguments.subject_instance is not None, "subject_instance must be provided to update dataset with executions"
        assert isinstance(self._arguments.subject_instance, BaseSubject), "subject_instance must be an instance of BaseSubject"
        subject: BaseSubject = self._arguments.subject_instance
        prog_input_column = self._arguments.target if self._arguments.backward else self._arguments.source
        prog_output_column = self._arguments.source if self._arguments.backward else self._arguments.target
        inputs = data[prog_input_column].tolist() if isinstance(data, DataFrame) else data[prog_input_column]
        executions = [
            self._arguments.post_formating(subject.execute(prog_input)) for prog_input in
            tqdm(inputs, desc="Updating dataset...", unit=" inputs", total=len(inputs))
        ]
        data = {prog_input_column: inputs, prog_output_column: executions}
        return DataFrame(data)

    @staticmethod
    def __load_file_to_df__(filepath: Path, use_tmp_directory: bool) -> Tuple[Optional[DataFrame], Optional[Path]]:
        # Unpacking
        if filepath is not None and filepath.is_file() and filepath.suffix not in (".pkl", ".csv"):
            match filepath.suffix:
                case ".zip":
                    try:
                        filepath = DataHandlers.unzip(filepath, use_tmp_directory=use_tmp_directory)
                    except (FileNotFoundError, FileExistsError, ValueError):
                        filepath = None
                case _:
                    raise TypeError(f"Unsupported file extension: {filepath.suffix}")

        # Loading data into Dataframe
        dataframe = None
        if filepath is not None and filepath.is_file():
            match filepath.suffix:
                case ".pkl":
                    dataframe = DataFrame(Pickle.load(filepath))
                case ".csv":
                    # noinspection PyArgumentList
                    dataframe = read_csv(filepath)
                case _:
                    raise TypeError(f"Unsupported file extension: {filepath.suffix}")
        return dataframe, filepath

    def load_dataset(self, simple: bool = False) -> tuple[DataFrame, DataFrame, Union[DataFrame, Dict[str, DataFrame]]]:
        """
        Prepare the dataset for training with optional simplified processing.
        :param simple: If True, use simplified processing (skip reduce_spaces)
        :return: Tuple with shuffled dataset, training and test splits
        """
        dataframe, filepath = self.__load_file_to_df__(Path(self._arguments.dataset).resolve(), not self._arguments.move)
        if dataframe is None:
            error_msg = f"Failed to load dataset from {self._arguments.dataset}"
            self._logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self._arguments.dataset = filepath.as_posix()
        if self._arguments.update_dataset:
            dataframe = self.update_dataset_with_subject_executions(dataframe)
            self._logger.info("Dataset updated with subject executions.")

        dataframe = self.prepare_shuffled_dataset(dataframe, simple)
        train_data, test_data = self.prepare_train_test_splits(dataframe)
        return dataframe, train_data, test_data

    def load_feature_dataset(self, simple: bool = False) -> tuple[DataFrame, DataFrame, dict[str, DataFrame]]:
        """
        Prepare the dataset for feature-based training with optional simplified processing.
        :param simple: If True, use simplified processing (skip reduce_spaces)
        :return: Tuple with shuffled dataset, training and test splits
        """
        dataframe, filepath = self.__load_file_to_df__(Path(self._arguments.dataset).resolve(), not self._arguments.move)
        if dataframe is None:
            error_msg = f"Failed to load dataset from {self._arguments.dataset}"
            self._logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self._arguments.dataset = filepath.as_posix()
        if self._arguments.update_dataset:
            dataframe = self.update_dataset_with_subject_executions(dataframe)
            self._logger.info("Dataset updated with subject executions.")
            dataframe = self.prepare_shuffled_dataset(dataframe, simple)
            train_df, test_data = self.prepare_feature_dataset(
                data=dataframe,
                source=self._arguments.source,
                target=self._arguments.target,
                test_size=self._arguments.test_size,
                max_features=self._arguments.kwargs["max_features_samples"],
                max_attempts=self._arguments.kwargs["max_features_attempts"],
                seed=self._arguments.kwargs["seed"],
                output_path=Path(self._arguments.dataset).resolve(),
                report_progress=True,
            )
            self._logger.info(
                f"Dataset prepared in {get_time_diff(self._init_time)}. Train size: {len(train_df)}, Test size: {sum(len(df) for df in test_data.values())}")
        else:
            assert self._arguments.split_dataset_path is not None, "split_dataset_path must be provided for feature-based training from preprocessed data"
            assert self._arguments.split_dataset_path.exists() and self._arguments.split_dataset_path.is_dir(), "split_dataset_path must be an existing directory for feature-based training from preprocessed data"
            ensure_string = not self._arguments.split_input
            dataframe = self.prepare_shuffled_dataset(dataframe, simple)
            train_df = self.load_preprocessed_data(self._arguments.split_dataset_path, "train", ensure_string, not self._arguments.move)
            assert train_df is not None, "Failed to load preprocessed training data from split_dataset_path"
            test_data = {}
            test_df_random = self.load_preprocessed_data(self._arguments.split_dataset_path, "test", ensure_string, not self._arguments.move)
            if test_df_random is not None:
                test_data["random"] = test_df_random
            test_df_rare = self.load_preprocessed_data(self._arguments.split_dataset_path, "test2", ensure_string, not self._arguments.move)
            if test_df_rare is not None:
                test_data["rare"] = test_df_rare
            test_df_mixed = self.load_preprocessed_data(self._arguments.split_dataset_path, "test3", ensure_string, not self._arguments.move)
            if test_df_mixed is not None:
                test_data["mixed"] = test_df_mixed
            self._logger.info(f"Dataset loaded in {get_time_diff(self._init_time)}. Train size: {len(train_df)}, Test size: {sum(len(df) for df in test_data.values())}")
        return dataframe, train_df, test_data

    @staticmethod
    def load_preprocessed_data(file_directory: Path, partition_name: str, ensure_string: bool = True,
                               use_tmp_directory: bool = True) -> DataFrame | None:
        filepath = file_directory / partition_name
        dataframe, _ = Trainer.__load_file_to_df__(filepath, use_tmp_directory)
        if dataframe is None:
            for ext in (".zip", ".pkl", ".csv"):
                filepath = file_directory / f"{partition_name}{ext}"
                if filepath.exists():
                    dataframe, _ = Trainer.__load_file_to_df__(filepath, use_tmp_directory)
                    if dataframe is not None:
                        break
        if dataframe is not None and ensure_string:
            for col in dataframe.columns:
                dataframe[col] = dataframe[col].apply(DataHandlers.stringify)
        return dataframe

    @staticmethod
    def save_preprocessed_dataframe(dataframe: DataFrame, file_directory: Path, partition_name: str, cleanup: bool = False):
        file_directory.mkdir(parents=True, exist_ok=True)
        filepath = file_directory / f"{partition_name}.csv"
        dataframe.to_csv(filepath, index=False)
        DataHandlers.zip(filepath)
        if cleanup:
            filepath.unlink()

    def prepare_shuffled_dataset(self, dataframe: DataFrame, simple: bool = False) -> DataFrame:
        random.seed(self._arguments.seed)
        dataframe = dataframe.sample(frac=1, random_state=self._arguments.seed).reset_index(drop=True)

        # Apply subset filtering if specified and using simple mode
        if simple and self._arguments.subset_size > 0:
            dataframe = dataframe.iloc[:self._arguments.subset_size + self._arguments.test_size]

        # Determine primary and secondary columns based on backward flag
        primary_col = self._arguments.target if self._arguments.backward else self._arguments.source
        assert primary_col in dataframe.columns, f"Primary column '{primary_col}' not found in dataset columns: {dataframe.columns.tolist()}"
        secondary_col = self._arguments.source if self._arguments.backward else self._arguments.target
        assert secondary_col in dataframe.columns, f"Secondary column '{secondary_col}' not found in dataset columns: {dataframe.columns.tolist()}"
        dataframe = dataframe[[primary_col, secondary_col]]

        # Apply transformations to primary column
        if self._arguments.split_input:
            dataframe[primary_col] = dataframe[primary_col].apply(lambda val: " ".join(val))
        elif not simple and self._arguments.reduce_spaces:
            dataframe[primary_col] = dataframe[primary_col].apply(DataHandlers.replace_spaces_except_after_comma)

        # Apply reduce_spaces to secondary column if needed (only in full processing mode)
        if not simple and self._arguments.reduce_spaces:
            dataframe[secondary_col] = dataframe[secondary_col].apply(
                DataHandlers.replace_spaces_except_after_comma)

        # Ensure string type for both columns
        if not self._arguments.split_input:
            dataframe[primary_col] = dataframe[primary_col].apply(DataHandlers.stringify)
        dataframe[secondary_col] = dataframe[secondary_col].apply(DataHandlers.stringify)
        return dataframe

    def prepare_train_test_splits(self, dataframe: DataFrame) -> tuple[DataFrame, Union[DataFrame, Dict[str, DataFrame]]]:
        relevant_columns = [self._arguments.source, self._arguments.target]
        if self._arguments.split_dataset_path is not None and self._arguments.split_dataset_path.exists():
            if self._arguments.split_dataset_path.is_dir():
                train_datasets = [filepath.name for filepath in self._arguments.split_dataset_path.glob("train*")]
                if len(train_datasets) > 1:
                    train_data = []
                    for name in train_datasets:
                        df = self.load_preprocessed_data(self._arguments.split_dataset_path, name, not self._arguments.move)
                        if df is not None and all(col in df.columns for col in relevant_columns):
                            df = df[relevant_columns]
                            train_data.append(df)
                    train_df = concat(train_data).reset_index(drop=True)
                    assert not train_df.empty, "Failed to load any valid preprocessed training data from split_dataset_path"
                else:
                    train_df = self.load_preprocessed_data(self._arguments.split_dataset_path, "train", not self._arguments.move)
                test_datasets = [filepath.name for filepath in self._arguments.split_dataset_path.glob("test*")]
                if len(test_datasets) > 1:
                    test_df = {}
                    for name in test_datasets:
                        df = self.load_preprocessed_data(self._arguments.split_dataset_path, name, not self._arguments.move)
                        if df is not None and all(col in df.columns for col in relevant_columns):
                            test_df[name] = df[relevant_columns]
                else:
                    test_df = self.load_preprocessed_data(self._arguments.split_dataset_path, "test", not self._arguments.move)
            else:
                raise ValueError("split_dataset_path must be a directory containing preprocessed train/test data")
        else:
            # If subset_size is specified, limit the size of the shuffled DataFrame
            test_df = dataframe.iloc[:self._arguments.test_size]
            train_df = dataframe.iloc[self._arguments.test_size:]

            if self._arguments.subset_size > 0:
                train_df = train_df.sample(frac=1, random_state=self._arguments.seed).reset_index(drop=True)
                train_df = train_df.iloc[:self._arguments.subset_size]

            if self._arguments.split_dataset_path is not None:
                self.save_preprocessed_dataframe(train_df, self._arguments.split_dataset_path, "train", cleanup=True)
                self.save_preprocessed_dataframe(test_df, self._arguments.split_dataset_path, "test", cleanup=True)

        return train_df, test_df

    @staticmethod
    def is_valid_feature_testing_data(row: Series, column_name: str, min_feature_count: int) -> bool:
        if isinstance(row[column_name], list):
            source_data = row[column_name]
        else:
            source_data = row[column_name].split()
        return len(source_data) >= min_feature_count

    @staticmethod
    def find_traces_with_rare_features(
            df: DataFrame,
            input_column: str,
            features_column: str,
            rare_features: List[str],
            min_feature_count: int,
            max_feature_count: int | None = None) -> List[Tuple[str, str, List[str]]]:
        """
        Prepares a list of traces whose coverage contains at least `min_count` rare features.
        :param df: DataFrame containing the data
        :param input_column: Name of the column containing the input data
        :param features_column: Name of the column containing the features/coverage data
        :param rare_features: List of rare features to look for
        :param min_feature_count: Minimum number of rare features required in a row's coverage
        :param max_feature_count: Maximum number of rare features to keep per row (if is not None, at most that many rare features are kept per row (sampled without replacement)).
        :return: Returns a list of (input, coverage, [selected_rare_features]) for rows whose coverage contains at least `min_count` rare features.
        """
        rare_set = set(rare_features)
        results: List[Tuple[str, str, List[str]]] = []

        for _, row in df.iterrows():
            bc = str(row[input_column])
            coverage = str(row[features_column])
            features = coverage.split()
            rare_in_trace = sorted({f for f in features if f in rare_set})

            if len(rare_in_trace) >= min_feature_count:
                if max_feature_count is not None and len(rare_in_trace) > max_feature_count:
                    selected_rare = random.sample(rare_in_trace, max_feature_count)
                else:
                    selected_rare = rare_in_trace
                results.append((bc, coverage, selected_rare))

        return results

    @staticmethod
    def forge_mixed_features(
            df: DataFrame,
            input_column: str,
            features_column: str,
            rare_features: List[str] | Set[str],
            num_non_rare: int,
            num_rare: int,
    ) -> List[Tuple[str, str, List[str]]]:
        """
        Prepares a list of traces with mixed features: some non-rare and some negated rare features.
        :param df: DataFrame containing the data
        :param input_column: Name of the column containing the input data
        :param features_column: Name of the column containing the features/coverage data
        :param rare_features: List or set of rare features
        :param num_non_rare: Number of non-rare features to select per row
        :param num_rare: Number of rare features to select and negate per row
        :return: List of tuples (input, original_trace, new_trace_as_list)
        For each row in `df`:
          - select `num_non_rare` features from coverage that are not in `rare_features`
          - select `num_rare` features from coverage that are in `rare_features`, negate them (`!feature`)
          - combine into a single list (optionally shuffled)

        """
        rare_set: Set[str] = set(rare_features)
        results: List[Tuple[str, str, List[str]]] = []

        for _, row in df.iterrows():
            inp = str(row[input_column])
            coverage = str(row[features_column])
            features = coverage.split()
            non_rare = list({f for f in features if f not in rare_set})
            rare_in_row = list({f for f in features if f in rare_set})
            selected_non_rare = non_rare if len(non_rare) < num_non_rare else random.sample(non_rare, num_non_rare)
            if len(rare_in_row) < num_rare:
                filtered_rare = [f for f in rare_features if f not in selected_non_rare]
                selected_rare = filtered_rare if len(filtered_rare) < num_rare else random.sample(filtered_rare, num_rare)
            else:
                selected_rare = random.sample(rare_in_row, num_rare)
            negated_rare = [f"!{f}" for f in selected_rare]
            results.append((inp, coverage, selected_non_rare + negated_rare))

        return results

    @staticmethod
    def prepare_feature_dataset(data: Union[DataFrame, dict[str, list[Any]]],
                                source: str,
                                target: str,
                                test_size: int,
                                min_features: int = 3,
                                max_features: int = 5,
                                max_attempts: int = 1,
                                seed: int = SEED,
                                output_path: Optional[str | Path] = None,
                                report_progress: bool = False,
                                rare_feature_threshold: int = 25) -> tuple[DataFrame, dict[str, DataFrame]]:
        """
        Prepare a test set for feature-based training by sampling combinations of features.
        :param data: Input data containing initial test samples. Could be a DataFrame or a dictionary with lists.
        :param source: Source column name
        :param target: Target column name
        :param test_size: Desired size of the test set
        :param min_features: Minimum number of features per sample. By default, at least 3 features are used.
        :param max_features: Maximum number of features per sample. By default, up-to 5 features are used.
        :param max_attempts: Number of attempts to sample unique feature combinations. By default, 1 attempts are made.
        :param seed: Random seed for reproducibility
        :param output_path: Optional path to save the prepared train / test datasets
        :param report_progress: Whether to report progress during dataset preparation
        :param rare_feature_threshold: Threshold to consider a feature as rare
        :return: Tuple containing the training and test DataFrames
        """
        assert min_features >= 1, "min_features should be at least 1."
        assert min_features <= max_features, "min_features should be less than or equal to max_features."
        dataframe = data if isinstance(data, DataFrame) else DataFrame(data)
        src = source if any(x in source for x in ("trace", "coverage")) else target
        trg = target if all(x not in target for x in ("trace", "coverage")) else source

        dataframe[src] = dataframe[src].apply(DataHandlers.stringify)
        dataframe[trg] = dataframe[trg].apply(DataHandlers.stringify)

        random.seed(seed)
        shuffled_df = dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)

        test_count = 0
        test_rows = []
        train_rows = []
        test_data_random = {src: [], trg: []}
        test_data_rare = {src: [], trg: []}
        test_data_mixed = {src: [], trg: []}

        for idx, row in shuffled_df.iterrows():
            if test_count < test_size and Trainer.is_valid_feature_testing_data(row, src, max_features):
                test_rows.append(row)
                test_count += 1
            else:
                train_rows.append(row)

        test_df = DataFrame(test_rows)
        train_df = DataFrame(train_rows)
        test_df.reset_index(inplace=True)
        train_df.reset_index(inplace=True)

        for i in range(min_features, max_features + 1):
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Preparing features {i}") if report_progress else test_df.iterrows():
                if isinstance(row[src], list):
                    source_data = row[src]
                else:
                    source_data = row[src].split()
                inputs_set = set()
                while len(inputs_set) < max_attempts:
                    inputs_set.add(" ".join(random.sample(source_data, i)))
                for inputs in inputs_set:
                    test_data_random[src].append(inputs)
                    test_data_random[trg].append(row[trg])

        all_coverage = test_df[src].tolist()
        all_features = [feature for cov in all_coverage for feature in cov.split(" ")]
        feature_distribution = sorted(Counter(all_features).items(), key=lambda x: (x[1], x[0]))
        rare_features = [feature for feature, count in feature_distribution if count < rare_feature_threshold]
        if len(rare_features) == 0:
            rare_features = [feature for feature, count in feature_distribution[:max(1, len(feature_distribution) // 10)]]
        assert len(rare_features) > 0, "No rare features found in the dataset."

        for i in range(min_features, max_features + 1):
            rares = Trainer.find_traces_with_rare_features(test_df, trg, src, rare_features, i, i)
            for input_data, _, selected_rares in rares:
                test_data_rare[trg].append(input_data)
                test_data_rare[src].append(" ".join(selected_rares))

        negative_feature_counter = 0
        for i in range(min_features, max_features + 1):
            negative_feature_counter += 1
            mixed = Trainer.forge_mixed_features(test_df, trg, src, rare_features, i, negative_feature_counter)
            for input_data, _, new_trace in mixed:
                test_data_mixed[trg].append(input_data)
                test_data_mixed[src].append(" ".join(new_trace))

        test_df_random = DataFrame(test_data_random)
        test_df_rare = DataFrame(test_data_rare)
        test_df_mixed = DataFrame(test_data_mixed)

        if output_path is not None:
            output_path = Path(output_path).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(output_path.joinpath("train.csv"), index=False)
            test_df_random.to_csv(output_path.joinpath("test.csv"), index=False)
            test_df_rare.to_csv(output_path.joinpath("test2.csv"), index=False)
            test_df_mixed.to_csv(output_path.joinpath("test3.csv"), index=False)
        return train_df, {"random": test_df_random, "rare": test_df_rare, "mixed": test_df_mixed}

    def train_encoder_decoder_tokenizers(self,
                                         dataframe: DataFrame,
                                         source_tokenizer_factory: Optional[Callable[..., BaseTokenizer]] = None,
                                         target_tokenizer_factory: Optional[Callable[..., BaseTokenizer]] = None) -> tuple[Optional[BaseTokenizer], Optional[BaseTokenizer]]:
        start_time = datetime.now()
        if source_tokenizer_factory is None:
            source_tokenizer_factory = self._arguments.source_tokenizer_class
        if target_tokenizer_factory is None:
            target_tokenizer_factory = self._arguments.target_tokenizer_class

        if source_tokenizer_factory is not None:
            tokenizer = source_tokenizer_factory(None)
            tokenizer.train(dataframe[self._arguments.source].tolist(), legacy_padding_mode=self._arguments.legacy_padding_mode)
        else:
            tokenizer = None
        if target_tokenizer_factory is not None:
            output_tokenizer = target_tokenizer_factory(None)
            output_tokenizer.train(dataframe[self._arguments.target].tolist(), legacy_padding_mode=self._arguments.legacy_padding_mode)
        else:
            output_tokenizer = None

        self._arguments.kwargs["tokenizer"] = tokenizer
        self._arguments.kwargs["output_tokenizer"] = output_tokenizer
        self._logger.info(f"Tokenizers trained in {get_time_diff(start_time)} | {tokenizer.__class__.__name__} | {output_tokenizer.__class__.__name__}")
        return tokenizer, output_tokenizer

    def train_decoder_tokenizer(self, dataframe: DataFrame, tokenizer_factory: Callable[..., BaseTokenizer]):
        start_time = datetime.now()
        dataframe["entries"] = dataframe.apply(lambda row: f"{row[self._arguments.source]} <|cls|> {row[self._arguments.target]}", axis=1)
        tokenizer = tokenizer_factory(None)
        tokenizer.train(dataframe["entries"].tolist())
        self._arguments.kwargs["tokenizer"] = tokenizer
        self._logger.info(f"Tokenizer trained in {get_time_diff(start_time)}")
        return tokenizer

    def find_dummy_length(self, dataframe: DataFrame):
        start_time = datetime.now()
        dataframe["entries"] = dataframe.apply(lambda row: f"{row[self._arguments.source]} <|cls|> {row[self._arguments.target]}", axis=1)
        tokenizer = DummyTokenizer("microsoft/codebert-base")
        max_length = tokenizer.estimate_max_length(dataframe["entries"].tolist())
        self._arguments.kwargs["max_length"] = max_length
        self._logger.info(f"Preprocessing completed in {get_time_diff(start_time)}.\nMax length: {max_length}")


class ConfigForger(metaclass=SingletonMeta):
    @staticmethod
    def forge_encoder_decoder_config(arguments: TrainArguments):  # -> EncoderDecoderConfig
        from modelizer.models.custom import EncoderDecoderConfig
        return EncoderDecoderConfig(
            output_dir=arguments.output_dir,
            source_vocab_size=arguments.kwargs["tokenizer"].vocab_size,
            target_vocab_size=arguments.kwargs["output_tokenizer"].vocab_size,
            source_max_len=arguments.kwargs["tokenizer"].max_sequence_length,
            target_max_len=arguments.kwargs["output_tokenizer"].max_sequence_length,
            source=arguments.source,
            target=arguments.target,
            backward=arguments.backward,
            use_flash=arguments.use_flash,
            compile_model=False,
            reduce_memory_usage=False,
            optimizer="adamw",
            scheduler=None,
            positional_encoding_type="none",
            embedding_size=256,
            hidden_size=256,
            feedforward_size=512,
            num_heads=8,
            enc_layers=2,
            dec_layers=4,
            learning_rate=1e-05,
            validation_fraction=0.8,
            reduce_spaces=arguments.reduce_spaces,
            wandb_token=arguments.wandb if arguments.wandb is not None and len(arguments.wandb) > 0 else None,
            max_memory_usage=arguments.max_usable_memory,
        )

    @staticmethod
    def forge_legacy_config(arguments: TrainArguments):  # -> LegacyConfig
        from modelizer.models.legacy import LegacyConfig
        return LegacyConfig(
            output_dir=arguments.output_dir,
            source_vocab_size=arguments.kwargs["tokenizer"].vocab_size,
            target_vocab_size=arguments.kwargs["output_tokenizer"].vocab_size,
            source_max_len=arguments.kwargs["tokenizer"].max_sequence_length,
            target_max_len=arguments.kwargs["output_tokenizer"].max_sequence_length,
            source=arguments.source,
            target=arguments.target,
            backward=arguments.backward,
            embedding_size=256,
            feedforward_size=512,
            num_heads=8,
            enc_layers=2,
            dec_layers=4,
            dropout=0.,
            force_cpu=arguments.use_cpu,
            compile_model=False,
            optimizer="adamw",
            scheduler=None,
            learning_rate=1e-05,
            weight_decay=0.1,
            validation_fraction=0.8,
            shuffle_train_data=True,
            report_epoch_progress=False,
            reduce_spaces=arguments.reduce_spaces,
            wandb_token=arguments.wandb if arguments.wandb is not None and len(arguments.wandb) > 0 else None,
            max_memory_usage=arguments.max_usable_memory,
        )
