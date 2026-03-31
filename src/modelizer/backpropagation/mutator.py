import random

from inspect import signature, Parameter
from typing import Callable, Optional, Any
from collections.abc import Sequence, Iterable

from modelizer.configs import SEED, TIMEOUT_SECONDS
from modelizer.tokenizers.abstract import BaseTokenizer, Tensor
from modelizer.generators.subjects import BaseSubject, ExecutionState


class SequenceMutator:
    """A class that provides configurable methods for mutating data sequences."""

    def __init__(
            self,
            max_mutations: int = 5,
            mutation_strategies: Optional[Sequence[Any] | Iterable[Any]] = None,
            *,
            seed: int = SEED,
            placeholders: Optional[Sequence[str]] = None,
    ):
        """
        Initializes the mutator.
        :param max_mutations: the maximum number of mutations to be performed on the input sequence. Default is 5.
        :param mutation_strategies: (optional) an iterable or a sequence of mutation functions. If None, a default set of strategies is used.
        :param placeholders: (optional) an iterable with placeholders. If None, no mutations that involve placeholders
        could be performed.
        :param seed: seed for the random module. Default is the global seed.
        """
        self._max_mutations = max_mutations
        random.seed(seed)

        if placeholders is not None:
            self._placeholders_list = list(placeholders)
            self._placeholders_set = set(placeholders)
        else:
            self._placeholders_set = set()
            self._placeholders_list = list()

        if mutation_strategies is not None:
            assert isinstance(mutation_strategies, Sequence | Iterable), "Strategies must be an iterable collection or a sequence."
            assert all(callable(func) for func in mutation_strategies), "All strategies must be callable."
            self._strategies: list[Callable] = list()
            for func in mutation_strategies:
                self.register_strategy(func)
        else:
            self._strategies: list[Callable] = [
                self._delete_mutation,
                self._truncate_mutation,
                self._insert_keyword,
            ]
            if placeholders is not None:
                self._strategies.append(self._insert_placeholder)

    @property
    def max_mutations(self) -> int:
        return self._max_mutations

    @max_mutations.setter
    def max_mutations(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"max_mutations must be a positive integer, got type={type(value)} value={value} instead.")
        self._max_mutations = value

    @property
    def placeholders(self):
        return self._placeholders_list

    def register_strategy(self, func):
        """
        Register (or override) a mutation strategy.
        :param func: A function that accepts a sequence (list) and returns the mutated sequence.
        """
        if not callable(func):
            raise ValueError("The strategy function must be callable.")
        elif len([param for param in signature(func).parameters.values() if param.default == Parameter.empty]) != 1:
            raise ValueError("The strategy function must accept exactly one argument.")
        self._strategies.append(func)

    def mutate(self, data: list) -> list:
        """
        Mutates the input data using a randomly selected mutation strategy from the registry.

        :param data: Input sequence to be mutated.
        :return: Mutated sequence.
        """
        if len(data) == 0:
            raise ValueError("Data must not be empty.")

        new_sequence = data.copy()
        num_mutations = random.randint(1, self.max_mutations)

        mutants = []

        for _ in range(num_mutations):
            new_sequence = random.choice(self._strategies)(new_sequence)
            mutants.append(new_sequence)

        return mutants

    # Built-in mutation strategy methods:
    @staticmethod
    def _delete_mutation(sequence: list) -> list:
        """Delete a random element from the sequence if possible."""
        if len(sequence) > 1:
            to_delete = random.choice(sequence)
            sequence.remove(to_delete)
        return sequence

    @staticmethod
    def _truncate_mutation(sequence: list) -> list:
        """Performs truncation by removing the last element if sequence length > 1."""
        if len(sequence) > 1:
            sequence = sequence[:-1]
        return sequence

    def _is_keyword(self, token: str) -> bool:
        """Checks if token qualifies as a keyword (i.e. does not contain any placeholder)."""
        return all(placeholder not in token for placeholder in self._placeholders_set)

    def _insert_keyword(self, sequence: list) -> list:
        """Inserts a keyword into the sequence. Chooses one keyword among tokens that do not pass is_keyword."""
        keywords = [k for k in sequence if self._is_keyword(k)]
        if keywords:
            index = sequence.index(random.choice(sequence))
            sequence.insert(index, random.choice(keywords))
        return sequence

    def _insert_placeholder(self, sequence: list) -> list:
        """Inserts a placeholder into the sequence. If the selected placeholder already exists, an enumerated version is inserted."""
        placeholder = random.choice(self._placeholders_list)
        target_placeholders = [token for token in sequence if placeholder in token]
        if all(token == placeholder for token in target_placeholders):
            insert_token = placeholder
        else:
            insert_token = f"{placeholder}_{len(target_placeholders) + 1}"
        index = sequence.index(random.choice(sequence))
        sequence.insert(index, insert_token)
        return sequence


class MutationTester:
    def __init__(self,
                 subject: BaseSubject,
                 tokenizer: BaseTokenizer,
                 *,
                 max_mutations: int = 5, timeout: int | float | None = TIMEOUT_SECONDS,
                 seed: int = SEED,
                 placeholders: Optional[Sequence[str]] = None,
                 mutation_strategies: Optional[Sequence[Any] | Iterable[Any]] = None,
                 collect_only_passing_tests: bool = True):
        """
        Initializes the MutationTester.
        :param subject: an object that implements the BaseSubject interface and can interact with program under test.
        :param tokenizer: an object that implements the BaseTokenizer interface and can process test data.
        :param max_mutations: the maximum number of mutations to be performed on the input sequence. Default is 5.
        :param timeout: the maximum time (in seconds) to wait for the test to complete. Default is 10.
        :param seed: seed for the random module. Default is the global seed.
        :param placeholders: (optional) an iterable with placeholders. If None, no mutations that involve placeholders
        could be performed.
        :param mutation_strategies: (optional) an iterable or a sequence of mutation functions. If None, a default set of strategies is used.
        :param collect_only_passing_tests: if True, only passing tests are collected. Default is True.
        """
        assert isinstance(subject, BaseSubject), "subject must be an instance of BaseSubject."
        assert isinstance(tokenizer, BaseTokenizer), "tokenizer must be an instance of BaseTokenizer."
        self._mutator = SequenceMutator(max_mutations, mutation_strategies, seed=seed, placeholders=placeholders)
        self._subject = subject
        self._tokenizer = tokenizer
        self._subject.timeout = timeout
        self._only_passing_tests = collect_only_passing_tests

    @property
    def timeout(self) -> int:
        return self._subject.timeout

    @timeout.setter
    def timeout(self, value: int | float | None):
        if value is not None and not isinstance(value, int | float):
            raise ValueError(f"timeout can be None or a number, got type={type(value)} instead.")
        if value is not None and value <= 0:
            raise ValueError(f"timeout must be greater than zero, got {value} instead.")
        self._subject.timeout = value

    def mutate(self, data: str | list[int] | tuple[int] | Tensor, is_backward_model: bool = False, trials: int = 1) -> list[tuple[str, str]]:
        """
        Mutates the input data and tests the mutated data using the subject.
        :param data: input data to be mutated.
        :param is_backward_model: indicates whether the order of input and output should be reversed.
        :param trials: the number of mutation trials to be performed. Default is 1.
        :return: a list of tuples containing the mutated input and the corresponding output.
        """
        assert len(data) > 0, "Data must not be empty."
        if isinstance(data, Tensor):
            if len(data.shape) > 1:
                assert data.shape[0] == 1, "MutationTester does not support batched inputs."
                data = data.squeeze()
            data = data.tolist()

        result = list()

        if isinstance(data, str):
            original_input_str = data
            data = self._tokenizer(data, return_tensors=False)["input_ids"]
        else:
            original_input_str = self._tokenizer.reconstruct(data)

        cache = {original_input_str}  # Use the original string for cache
        for _ in range(trials):
            mutants = self._mutator.mutate(data)
            input_candidates = {self._tokenizer.reconstruct(m) for m in mutants}
            unseen = input_candidates.difference(cache)
            cache.update(unseen)
            for candidate in unseen:
                output = self._subject.execute(candidate)
                output = self._subject.get_encoder()(output)
                if self._subject.state == ExecutionState.PASS or not self._only_passing_tests:
                    result.append((output, candidate) if is_backward_model else (candidate, output))

        return result
