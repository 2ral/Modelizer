from pathlib import Path
from typing import Optional
from datetime import datetime
from abc import ABC, abstractmethod
from random import seed as random_seed

from pandas import DataFrame

from modelizer import configs
from modelizer.generators.subjects import BaseSubject
from modelizer.utils import Pickle, Logger, LoggerConfig, get_time_diff


class GeneratorInterface(ABC):
    """High-level interface for collecting input-output pairs."""
    def __init__(self, source: str, target: str, subject: BaseSubject, seed: int = configs.SEED, logger: Optional[Logger | LoggerConfig] = None, **_):
        """
        Constructor for the GeneratorInterface class.
        :param source: name of the source type
        :param target: name of the target type
        :param subject: instance of the BaseSubject subclass
        :param seed: seed to initialize random number generator
        :param logger: (Optional) instance of the Logger class for logging
        """
        assert isinstance(subject, BaseSubject), "subject must be an instance of SubjectInterface"
        assert isinstance(source, str), "source must be a string"
        assert isinstance(target, str), "target must be a string"
        assert isinstance(seed, int), "seed must be an integer"

        if logger is None:
            logger = Logger(None)
        elif isinstance(logger, LoggerConfig):
            logger = Logger(logger)

        random_seed(seed)
        self._subject = subject
        self._source = source
        self._target = target
        self._monitored_inputs = set()
        self._source_data = []
        self._target_data = []
        self._logger = logger

    @property
    def subject(self) -> BaseSubject:
        return self._subject

    @property
    def source(self) -> str:
        return self._source

    @property
    def target(self) -> str:
        return self._target

    @property
    def logger(self) -> Logger:
        return self._logger

    @abstractmethod
    def generate(self) -> str | list:
        """Executes the generator and outputs program input, which optionally can be already tokenized"""
        raise NotImplementedError("generate method not implemented in the subclass")

    def generate_samples(self, count: int, seed: int = configs.SEED, fresh_start: bool = True) -> tuple[list, list]:
        """
        Executes the generation pipeline for a given number of iterations.
        :param count: number of iterations
        :param seed: seed to initialize random number generator
        :param fresh_start: whether to start from scratch or continue from previous state
        """
        if fresh_start:
            self._source_data.clear()
            self._target_data.clear()
            self._monitored_inputs.clear()

        self._logger.info(f"Generating {count} unique input-output pairs.")
        start_time = datetime.now()
        random_seed(seed)
        fails = 0
        while len(self._source_data) < count and fails < count:
            data = self.generate()
            if isinstance(data, list):
                data = tuple(data)
            if data in self._monitored_inputs:
                fails += 1
                continue
            fails = 0
            self._source_data.append(data)
            self._monitored_inputs.add(data)
        self._target_data = [self._subject.execute(i) for i in self._source_data]
        self._logger.info(f"Dataset formation completed in {get_time_diff(start_time)}")
        print(f"generated pairs until failsafe: {len(self._source_data)}")
        return list(self._source_data), self._target_data

    def export(self, filepath: Optional[str | Path] = None, to_csv: bool = True) -> DataFrame:
        """
        Exports the generated data to a pandas DataFrame and optionally saves it to a CSV file or pickle object.
        :param filepath: path to save results as a file.
        :param to_csv: whether to save results as a CSV file.
        Could be str, pathlib.Path or None. By default, is None.
        If not None is passed, the results will be saved to the filepath as a CSV file.
        :return: pandas DataFrame containing the generated input-output pairs
        """
        results = {self.source: self._source_data, self.target: self._target_data}
        dataframe = DataFrame(results)
        if filepath is not None:
            if not isinstance(filepath, str | Path):
                error_msg = "filepath must be a string or pathlib.Path object"
                self._logger.error(error_msg)
                raise ValueError(error_msg)
            filepath = Path(filepath)
            if to_csv:
                if filepath.suffix != ".csv":
                    filepath = Path(filepath).with_suffix(".csv")
                dataframe.to_csv(filepath, index=False)
            else:
                if filepath.suffix != ".pkl":
                    filepath = Path(filepath).with_suffix(".pkl")
                Pickle.dump(results, filepath)
            self._logger.info(f"Data exported to {filepath}")
        return dataframe
