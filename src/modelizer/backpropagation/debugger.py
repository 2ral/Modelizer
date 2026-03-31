from typing import Any
from abc import ABC, abstractmethod

from modelizer.configs import TIMEOUT_SECONDS
from modelizer.tokenizers.abstract import BaseTokenizer
from modelizer.generators.subjects import BaseSubject, ExecutionState


class DeltaDebugger(ABC):
    SUPPORTED_MODES = ('+', '-', '+-')

    def __init__(self, supress_assertions: bool):
        self.cache: dict[str, tuple[ExecutionState, Any]] = dict()
        self.supress_assertions = supress_assertions

    def reset(self):
        self.cache.clear()

    @abstractmethod
    def test(self, inp: list[Any] | tuple[Any]) -> ExecutionState:
        """
        Test the program under test using the provided input and return whether it passes or fails.
        It is recommended to store the input -> execution state mapping in the cache.
        :param inp: The test input sequence
        """
        raise NotImplementedError("test method not implemented in the subclass")

    def repair(self, fail_inp: list[Any] | tuple[Any], mode: str = '+') -> list[Any] | None:
        """
        Perform Delta Debugging on the input sequence.
        :param fail_inp: The failing input sequence
        :param mode: algorithm mode '+' for maximizing passing inputs, '-' for minimizing failing inputs, '+-' for both
        :return: a reduced input sequence or None if not a valid reduction found
        """
        assert mode in self.SUPPORTED_MODES, f"Invalid delta debugging mode: {mode}. Supported modes are {self.SUPPORTED_MODES}."

        def ret(set_passing: set, set_failing: set) -> list[Any] | None:
            if mode == '+':
                result = self.__from_set__(set_passing, fail_inp)
                if self.test(result) != ExecutionState.PASS:
                    result = None
            else:
                result = self.__from_set__(set_failing, fail_inp)
                if self.test(result) in [ExecutionState.FAIL, ExecutionState.TIMEOUT, ExecutionState.EXCEPTION]:
                    result = None
            return result

        self.reset()

        # Check that the subject really fails with the failing input.
        try:
            assert self.test(fail_inp) != ExecutionState.PASS, f"The input already passes the test. No debugging needed.\n{fail_inp}"
        except AssertionError:
            if self.supress_assertions:
                # print(f"Assertion suppressed. The input already passes the test.\n{fail_inp}")
                return fail_inp
            else:
                raise

        n = 1  # Initial granularity
        offset = 0

        c_pass = set()
        c_fail = set(range(len(fail_inp)))

        minimize_fail = '-' in mode
        maximize_pass = '+' in mode

        # Main loop
        while True:
            delta = c_fail - c_pass
            if len(delta) < n:
                return ret(c_pass, c_fail)

            deltas = self.__split__(delta, n)

            reduction_found = False

            j = 0
            while j < n:
                i = (j + offset) % n
                next_c_pass = c_pass | deltas[i]
                next_c_fail = c_fail - deltas[i]

                if minimize_fail and n == 2 and self.test(self.__from_set__(next_c_pass, fail_inp)) != ExecutionState.PASS:
                    c_fail = next_c_pass
                    offset = i  # was offset = 0 in original dd()
                    reduction_found = True
                    break

                elif maximize_pass and n == 2 and self.test(self.__from_set__(next_c_fail, fail_inp)) == ExecutionState.PASS:
                    c_pass = next_c_fail
                    offset = i  # was offset = 0 in original dd()
                    reduction_found = True
                    break

                elif minimize_fail and self.test(self.__from_set__(next_c_fail, fail_inp)) != ExecutionState.PASS:
                    c_fail = next_c_fail
                    n = max(n - 1, 2)
                    offset = i
                    reduction_found = True
                    break

                elif maximize_pass and self.test(self.__from_set__(next_c_pass, fail_inp)) == ExecutionState.PASS:
                    c_pass = next_c_pass
                    n = max(n - 1, 2)
                    offset = i
                    reduction_found = True
                    break

                else:
                    j += 1  # choose the next subset

            if not reduction_found:
                if n >= len(delta):
                    return ret(c_pass, c_fail)
                n = min(n * 2, len(delta))

    @staticmethod
    def __add_to__(collection: Any, elem: Any) -> Any:
        """Add the element to the collection; return a new collection."""
        if isinstance(collection, str):
            return collection + elem  # Strings

        try:  # Lists and other collections
            return collection + type(collection)([elem])
        except TypeError:
            pass

        try:  # Sets
            return collection | type(collection)([elem])
        except TypeError:
            pass

        raise ValueError("Cannot add element to collection")

    @staticmethod
    def __from_set__(the_set: Any, inp: list[Any] | tuple[Any]) -> Any:
        # Efficient reconstruction using type-specific methods.
        if isinstance(inp, str):
            return "".join(c for i, c in enumerate(inp) if i in the_set)
        elif isinstance(inp, (list, tuple)):
            return type(inp)([c for i, c in enumerate(inp) if i in the_set])
        elif isinstance(inp, set):
            return set(c for i, c in enumerate(inp) if i in the_set)
        else:
            # Fallback for other types
            ret = type(inp)()
            for i, c in enumerate(inp):
                if i in the_set:
                    ret = DeltaDebugger.__add_to__(ret, c)
            return ret

    @staticmethod
    def __split__(elems: Any, n: int) -> list:
        assert 1 <= n <= len(elems)

        k, m = divmod(len(elems), n)
        try:
            subsets = list(elems[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
        except TypeError:
            # Convert to list and back
            subsets = list(type(elems)(list(elems)[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n))

        assert len(subsets) == n
        assert sum(len(subset) for subset in subsets) == len(elems)
        assert all(len(subset) > 0 for subset in subsets)
        return subsets


class SequenceDebugger(DeltaDebugger):
    """A debugger for model predictions"""
    def __init__(self,
                 subject: BaseSubject,
                 tokenizer: BaseTokenizer,
                 supress_assertions: bool = False,
                 timeout: int | float | None = TIMEOUT_SECONDS,
                 max_repair_attempts: int = 10):
        """
        Initialize the SequenceDebugger

        :param subject: an instance of BaseSubject
        :param tokenizer: an instance of BaseTokenizer
        :param supress_assertions: if True it suppresses assertions while debugging
        :param timeout: The maximum time to wait for the subject to execute before timing out
        :param max_repair_attempts: The maximum number of attempts to repair the input sequence
        """
        assert isinstance(tokenizer, BaseTokenizer), "tokenizer must be an instance of BaseTokenizer."
        assert isinstance(subject, BaseSubject), "test_runner must be an instance of BaseSubject."
        super().__init__(supress_assertions)
        self._subject = subject
        self._subject.timeout = timeout
        self._tokenizer = tokenizer
        self._args_cache = None
        self._mapping = None
        self._max_repair_attempts = max_repair_attempts

    @property
    def subject(self) -> BaseSubject:
        return self._subject

    @property
    def max_repair_attempts(self) -> int:
        """
        Get the maximum number of attempts to repair the input sequence.
        :return: The maximum number of attempts
        """
        return self._max_repair_attempts

    @max_repair_attempts.setter
    def max_repair_attempts(self, value: int):
        """
        Set the maximum number of attempts to repair the input sequence.
        :param value: The maximum number of attempts
        """
        if not isinstance(value, int):
            raise ValueError(f"max_repair_attempts must be an integer, got type={type(value)} instead.")
        if value <= 0:
            raise ValueError(f"max_repair_attempts must be greater than 0, got {value} instead.")
        self._max_repair_attempts = value

    def reset(self):
        """Reset the cache"""
        super().reset()
        self._args_cache = None
        self._mapping = None

    def __sequence_to_input__(self, inp: list[Any] | tuple[Any], *args):
        """
        Convert the input sequence to a string that can be used as input to the subject.
        Do not call this method directly. Use the test method instead.

        :param inp: The input sequence
        :param args: additional arguments to be passed to the subject
        :return: The reconstructed input string
        """
        data = list(inp)
        if args != self._args_cache:
            mapping = None
            for arg in args:
                if isinstance(arg, dict) and all([isinstance(k, str) and "_" in k and k[-1].isdigit() for k in arg.keys()]):
                    mapping = arg
                    break
            self._args_cache = args
            self._mapping = mapping
        return self._tokenizer.reconstruct(data)

    def repair(self, fail_inp: list[Any] | tuple[Any], mode: str = '+') -> tuple[str | list, ExecutionState, Any] | None:
        """
        Perform Delta Debugging on the input sequence.
        :param fail_inp: The failing input sequence
        :param mode: algorithm mode '+' for maximizing passing inputs, '-' for minimizing failing inputs, '+-' for both
        :return: A reduced input sequence together with the execution state or None if no valid reduction is found
        """
        result = super().repair(fail_inp, mode=mode)

        if result is None:
            return None

        result = self.__sequence_to_input__(result)
        state, value = self.cache[result]
        return result, state, value

    def test(self, inp: list[Any] | tuple[Any]) -> ExecutionState:
        """
        Test the program under test using the provided input and return whether it passes or fails.
        :param inp: The test input sequence
        :return: ExecutionState
        """
        reconstructed_input = self.__sequence_to_input__(inp)
        if reconstructed_input not in self.cache:
            result = self._subject.execute(reconstructed_input)
            result = self._subject.get_encoder()(result)
            self.cache[reconstructed_input] = (self._subject.state, result)
        return self.cache[reconstructed_input][0]
