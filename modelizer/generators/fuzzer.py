import modelizer.utils as utils

from enum import Enum
from typing import Type
from pathlib import Path
from hashlib import sha384
from datetime import datetime
from warnings import simplefilter
from math import ceil as math_ceil
from itertools import repeat, chain
from abc import ABC, abstractmethod
from random import seed as random_seed

from fuzzingbook.Timeout import Timeout
from fuzzingbook.Grammars import Grammar
from fuzzingbook.GrammarCoverageFuzzer import GrammarCoverageFuzzer
from fuzzingbook.ProbabilisticGrammarFuzzer import ProbabilisticGrammarFuzzer
from pandas import DataFrame, read_csv, concat as df_concat

from modelizer.generators.grammars import remove_probabilities
from modelizer.generators.utils import PlaceholderProcessor, placeholder_updator


def __fuzz__(fuzzing_params: tuple[Grammar, int, int, int, int]):
    random_seed(datetime.now().timestamp())
    fuzzing_results = set()
    grammar, min_nonterminals, max_nonterminals, fuzzing_rounds, fuzzing_timeout = fuzzing_params
    if min_nonterminals < 0 or max_nonterminals < 0:
        grammar_fuzzer = GrammarCoverageFuzzer(grammar=remove_probabilities(grammar))
    else:
        grammar_fuzzer = ProbabilisticGrammarFuzzer(grammar=grammar,
                                                    min_nonterminals=min_nonterminals,
                                                    max_nonterminals=max_nonterminals)
    for _ in range(fuzzing_rounds):
        try:
            with Timeout(fuzzing_timeout):
                candidate = grammar_fuzzer.fuzz()
        except TimeoutError:
            continue
        else:
            fuzzing_results.add(candidate)
    return fuzzing_results


def fuzzer_generator(fuzzing_rounds: int,
                     fuzzing_params: list,
                     hashes_filepath: [str, Path],
                     logger,
                     partition: bool = False,
                     fuzzing_attempts: int = 100000,
                     fuzzing_rounds_per_thread: int = 256) -> tuple[list[str], list]:
    loaded = utils.pickle_load(hashes_filepath)
    hashes_dataset = set() if loaded is None else loaded
    generated = []
    partition_files = 0
    attempts = fuzzing_attempts
    fuzzing_with_probabilities = fuzzing_params[1] < 0 and fuzzing_params[2] < 0
    if partition and fuzzing_with_probabilities:
        d = 4 if fuzzing_rounds <= 10000000 else fuzzing_rounds // 1000000
        partition_size = math_ceil(fuzzing_rounds / d)
        logger.info(f"Maximum partition size: {partition_size}")
    else:
        partition_size = -1

    while len(generated) < fuzzing_rounds:
        total_remaining_files = fuzzing_rounds - len(generated)
        logger.info(f"{total_remaining_files} strings left to generate")
        if fuzzing_with_probabilities:
            logger.info(f"Max-Nonterminals: {fuzzing_params[2]} | Attempts left: {attempts}")

        if partition:
            partition_remaining_files = partition_size - partition_files

            if partition_remaining_files < total_remaining_files:
                files_per_thread = math_ceil(partition_remaining_files / utils.Multiprocessing.CORE_COUNT)
                fuzzing_params[3] = files_per_thread if files_per_thread < fuzzing_rounds_per_thread else fuzzing_rounds_per_thread
            else:
                files_per_thread = math_ceil(total_remaining_files / utils.Multiprocessing.CORE_COUNT)
                fuzzing_params[3] = files_per_thread if files_per_thread < fuzzing_rounds_per_thread else fuzzing_rounds_per_thread
        else:
            fuzzing_params[3] = fuzzing_rounds_per_thread

        records_before = len(generated)
        results = set.union(*utils.Multiprocessing.parallel_run(__fuzz__, list(repeat(fuzzing_params, utils.Multiprocessing.CORE_COUNT))))
        for formula in results:
            hash_v = sha384(formula.encode("UTF-8")).hexdigest()
            if hash_v in hashes_dataset:
                attempts -= 1
            else:
                hashes_dataset.add(hash_v)
                generated.append(formula)
                partition_files += 1
                if len(generated) == fuzzing_rounds or partition_files == partition_size:
                    break

        if fuzzing_with_probabilities and attempts <= 0 or partition_files == partition_size or records_before == len(generated):
            attempts = fuzzing_attempts
            fuzzing_params[1] = fuzzing_params[2] - 5
            fuzzing_params[2] += 5
            partition_files = 0

    if utils.pickle_dump(hashes_dataset, hashes_filepath):
        logger.error(f"Hashes dataset cannot be written to {str(hashes_filepath)}")
    return generated, fuzzing_params


def __token_checker__(params: tuple[list[tuple], list[str], Type[PlaceholderProcessor]]):
    subjects, markers, factory = params
    processor = factory(markers)
    results = []
    for tokens_group in subjects:
        row_results = []
        for token_seq in tokens_group:
            tokens = processor.split_tokens_with_placeholders(token_seq)
            row_results.append(tokens)
        results.append(tuple(row_results))
    return results


def token_checker(subjects: list[tuple], markers: list[str], factory: Type[PlaceholderProcessor]) -> list[tuple]:
    subjects = list(utils.Multiprocessing.chunk_generator(subjects))
    markers = list(repeat(markers, len(subjects)))
    factories = list(repeat(factory, len(subjects)))
    params = list(zip(subjects, markers, factories))
    return list(chain(*utils.Multiprocessing.parallel_run(__token_checker__, params, text="Checking tokens")))


class AbstractFuzzer(ABC):
    def __init__(self, task: str,
                 root_dir: Path,
                 grammar: Grammar,
                 markers: list | None,
                 columns: list[str],
                 n_jobs: int = utils.Multiprocessing.CORE_COUNT,
                 fuzzing_timout: int = 300,
                 placeholder_factory: Type[PlaceholderProcessor] = PlaceholderProcessor):

        simplefilter("ignore")
        self.grammar = grammar
        self.markers = markers
        self.columns = columns
        self.task = task
        self.n_jobs = n_jobs
        root_dir.mkdir(parents=True, exist_ok=True)
        self.logger = utils.Logger(utils.LoggingLevel.INFO, root_dir)
        self.__dataframe_fh__ = root_dir.joinpath("data.csv")
        self.__hashes_fh__ = root_dir.joinpath("hashes.pickle")
        self.__dataset_fh__ = root_dir.joinpath("dataset.pickle")
        self.dataframe = read_csv(self.__dataframe_fh__.as_posix(), index_col=0) \
            if self.__dataframe_fh__.exists() else DataFrame(columns=self.columns, dtype="str")
        self.fuzzing_timout = fuzzing_timout
        self.__placeholder_factory__ = placeholder_factory

    def fuzz(self, fuzzing_rounds: int = 1, max_nonterminals: int = 20, partition: bool = False) -> str:
        # Generating
        start_time = datetime.now()
        self.logger.info(f"Task: {self.task} Generation | Records to generate: {fuzzing_rounds}")
        if max_nonterminals < 0:
            min_nonterminals = max_nonterminals
            self.logger.info("Gathering formulas with GrammarCoverageFuzzer")
        else:
            self.logger.info("Gathering formulas with ProbabilisticGrammarFuzzer")
            min_nonterminals = 10 if max_nonterminals - 10 <= 0 else max_nonterminals - 10
            max_nonterminals = min_nonterminals + 10 if max_nonterminals < min_nonterminals else max_nonterminals
        fuzzing_params = [self.grammar, min_nonterminals, max_nonterminals, 0, self.fuzzing_timout]
        results, fuzzing_params = fuzzer_generator(fuzzing_rounds, fuzzing_params, self.__hashes_fh__, self.logger, partition)
        results = self.update_placeholders(results)
        self.logger.info(f"Generated {len(results)} formulas in {datetime.now() - start_time}")
        # Parsing
        job_start_time = datetime.now()
        self.logger.info(f"Task: Conversion | Records to process: {len(results)}")
        results = list(chain(*utils.Multiprocessing.parallel_run(self.__convert__, results, text="Converting...", chunkify=True, n_jobs=self.n_jobs)))
        if len(results) >= fuzzing_rounds:
            self.logger.info(f"All elements converted in {datetime.now() - job_start_time}")
        else:
            self.logger.info(f"Conversion succeeded only for {len(results)} elements")
            remaining = fuzzing_rounds - len(results)
            while remaining > 0:
                planned = remaining * 4 if remaining < 1000 else remaining
                self.logger.info(f"Task: Generating additional formulas | Remaining: {remaining} | Planned: {planned}")
                fuzzing_params = [self.grammar, min_nonterminals, max_nonterminals, 0, self.fuzzing_timout] if partition else fuzzing_params
                additional_results, fuzzing_params = fuzzer_generator(planned, fuzzing_params, self.__hashes_fh__,
                                                                      self.logger, partition if planned > 25000 else False)
                additional_results = self.update_placeholders(additional_results)
                self.logger.info(f"Task: Converting additional formulas | Records to process: {len(additional_results)}")
                additional_results = list(chain(*utils.Multiprocessing.parallel_run(self.__convert__, additional_results, text="Converting...", chunkify=True, n_jobs=self.n_jobs)))
                num_additionally_converted = len(additional_results)
                if num_additionally_converted >= planned:
                    self.logger.info(f"All remaining elements got converted")
                else:
                    self.logger.info(f"Conversion succeeded only for {num_additionally_converted} elements")
                results.extend(additional_results)
                remaining -= num_additionally_converted
        if len(results) > fuzzing_rounds:
            leftover = results[fuzzing_rounds:]
            hashes = utils.pickle_load(self.__hashes_fh__)
            assert hashes is not None, "Hashes dataset cannot be loaded"
            assert isinstance(hashes, set), "Hashes dataset is not a set"
            hashes = hashes.difference({sha384(elems[0].encode("UTF-8")).hexdigest() for elems in leftover})
            assert utils.pickle_dump(hashes, self.__hashes_fh__) == 0, "Hashes dataset cannot be saved"
            self.logger.info("Hashes dataset updated")
            results = results[:fuzzing_rounds]
        self.dataframe = df_concat([self.dataframe, DataFrame(results, columns=self.columns, dtype="str")], ignore_index=True)
        self.dataframe.to_csv(self.__dataframe_fh__.as_posix())
        total_records = len(results)
        # Generating Tokens and Saving the Dataset
        job_start_time = datetime.now()
        self.logger.info(f"Task: Tokenization | Records to process: {total_records}")
        results = list(chain(*utils.Multiprocessing.parallel_run(self.__tokenize__, results, text="Tokenizing...", chunkify=True)))
        if self.markers is not None:
            results = token_checker(results, self.markers, self.__placeholder_factory__)
        df = DataFrame(results, columns=self.columns)
        utils.pickle_dump(df.to_dict('list'), self.__dataset_fh__)
        if total_records == len(results):
            self.logger.info(f"All elements tokenized in {datetime.now() - job_start_time}")
        else:
            self.logger.info(f"Tokenization succeeded for {len(results)} elements")

        self.logger.info(f"Dataset saved at {self.__dataset_fh__.as_posix()}")
        self.logger.info(f"Overall Duration: {datetime.now() - start_time}")
        return self.__dataset_fh__.as_posix()

    def coverage_fuzz(self, fuzzing_rounds: int = 1) -> str:
        return self.fuzz(fuzzing_rounds, max_nonterminals=-1)

    def update_placeholders(self, subjects: list[str]) -> list[str]:
        return placeholder_updator(subjects, self.markers, self.__placeholder_factory__) if self.markers is not None else subjects

    @staticmethod
    @abstractmethod
    def __convert__(subjects: list[str]) -> list[tuple]:
        pass

    @staticmethod
    @abstractmethod
    def __tokenize__(subjects: list[tuple]) -> list[tuple]:
        pass
