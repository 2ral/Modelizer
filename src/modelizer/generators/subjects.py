import inspect
import requests
import subprocess

from enum import Enum
from time import sleep
from abc import ABC, abstractmethod
from random import uniform as random_uniform
from typing import Callable, Optional, Any, Sequence

from func_timeout import func_timeout, FunctionTimedOut

from modelizer.utils import Pickle
from modelizer.configs import SPACE_TOKEN


class ExecutionState(Enum):
    UNDEFINED = 0
    PASS = 1
    FAIL = 2
    TIMEOUT = 3
    EXCEPTION = 4
    FILE_NOT_FOUND = 5
    TRIALS_EXCEEDED = 6


class BaseSubject(ABC):
    """High-level interface for interacting with the subject under test"""

    def __init__(self, timeout: Optional[int] = None, trials: Optional[int] = None,
                 quick_start: bool = True, name: Optional[str] = None, location: Optional[str] = None, **_):
        """
        Constructor for the Subject class.
        :param timeout: maximum time (in seconds) to wait for the program to complete the execution
        :param trials: maximum number of trials to execute the program with the same input
        :param quick_start: boolean flag to indicate if the constructor must call pre_execution method
        :param name: optional name of the subject, used for identification
        :param location: optional location of the subject, used for identification and forks
        """
        self.execute = self.__call__
        assert timeout is None or (isinstance(timeout, (int, float)) and timeout > 0), "timeout must be a positive integer or float or None."
        assert trials is None or (isinstance(trials, int) and trials > 0), "trials must be a positive integer or None."
        self._timeout = timeout
        self._trials = trials
        self._current_trial = 0
        self._input = None
        self._output = None
        self._name = name
        self._location = location
        self._state = ExecutionState.UNDEFINED
        self._error_msg_timeout = "ERROR: Timeout."
        self._error_msg_file = "ERROR: File not found."
        self._error_msg_exception = "ERROR: Exception occurred."
        self._error_msg_trials = "ERROR: Maximum number of trials exceeded."
        self._comparator: Optional[Callable[[Any, Any], Any]] = None
        self._encoder: Optional[Callable[[Any,], Any]] = None
        self._decoder: Optional[Callable[[Any,], Any]] = None

        if quick_start:
            self.pre_execution()

    def __del__(self):
        try:
            self.post_execution()
        except:
            pass

    @property
    def timeout(self) -> int | float | None:
        return self._timeout

    @timeout.setter
    def timeout(self, value: int | float | None):
        assert value is None or isinstance(value, int | float), f"timeout can be None or a number, got type={type(value)} instead."
        if value is not None:
            assert value > 0, f"timeout must be greater than 0, got value={value} instead."
        self._timeout = value

    @property
    def trials(self) -> int | None:
        return self._trials

    @trials.setter
    def trials(self, value: int | None):
        assert value is None or isinstance(value, int), f"trials can be an integer or None, got type={type(value)} instead."
        if value is not None:
            assert value > 0, f"trials must be greater than 0, got value={value} instead."
        self._trials = value
        if self._trials is not None:
            self._current_trial = 0

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def output(self) -> str | list:
        return self._output

    @property
    def input(self) -> str | list:
        return self._input

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def location(self) -> str | None:
        return self._location

    @property
    def comparator(self) -> Optional[Callable[[Any, Any], Any]]:
        return self._comparator

    @comparator.setter
    def comparator(self, func: Optional[Callable[[Any, Any], Any]]):
        if func is None:
            self._comparator = None
        else:
            assert callable(func), "comparator must be a callable function."
            signature = inspect.signature(func)
            positional_params = [
                p for p in signature.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            required = [p for p in positional_params if p.default is inspect.Parameter.empty]
            assert len(required) == 2, "comparator function must accept exactly two required positional arguments."
            self._comparator = func

    @property
    def encoder(self) -> Optional[Callable[[Any,], Any]]:
        return self._encoder

    @encoder.setter
    def encoder(self, func: Optional[Callable[[Any,], Any]]):
        if func is None:
            self._encoder = None
        else:
            assert callable(func), "encoder must be a callable function."
            signature = inspect.signature(func)
            positional_params = [
                p for p in signature.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            required = [p for p in positional_params if p.default is inspect.Parameter.empty]
            assert len(required) == 1, "encoder function must accept exactly one required positional argument."
            self._encoder = func

    @property
    def decoder(self) -> Optional[Callable[[Any,], Any]]:
        return self._decoder

    @decoder.setter
    def decoder(self, func: Optional[Callable[[Any,], Any]]):
        if func is None:
            self._decoder = None
        else:
            assert callable(func), "decoder must be a callable function."
            signature = inspect.signature(func)
            positional_params = [
                p for p in signature.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            required = [p for p in positional_params if p.default is inspect.Parameter.empty]
            assert len(required) == 1, "decoder function must accept exactly one required positional argument."
            self._decoder = func

    def compare_output(self, arg1: Any, arg2: Any) -> Any | None:
        """
        Function to compare the expected and actual output of the subject.
        Returns None if self.comparator object is None.
        Otherwise, it returns the result of self.comparator(arg1, arg2).
        :param arg1: first argument to compare
        :param arg2: second argument to compare
        :return: result of comparison or None
        """
        return None if self._comparator is None else self._comparator(arg1, arg2)

    def __getstate__(self):
        """
        Method to serialize the object. Required for pickling. Reimplement in subclass if needed.
        :return: dictionary of object attributes
        """
        return {
            "timeout": self._timeout,
            "trials": self._trials,
            "name": self._name,
        }

    def __setstate__(self, state):
        """
        Method to deserialize the object. Required for pickling. Reimplement in subclass if needed.
        :param state: dictionary of object attributes
        """
        self._timeout = state["timeout"]
        self._trials = state["trials"]
        self._name = state["name"]

    def reset(self):
        """Function to reset the state of the tested subject. By default, it does nothing. Reimplement in subclass if needed."""
        pass

    @abstractmethod
    def __execute__(self, data: Any | list[Any] | tuple[Any]) -> Any | list[Any]:
        """
        Executes the program under test with given input.
        This method should catch all exceptions and return output of the program or an error message.
        It should not raise any exceptions.
        :param data: input to the program
        :return program output or an error message
        """
        raise NotImplementedError("__execute__ method not implemented in the subclass")

    def __call__(self, data: Any | list[Any] | tuple[Any]) -> Any | list[Any]:
        """
        Resets the state of program under test and executes/triggers it with given input.
        :param data: input to the program
        :return program output or an error message
        """
        if data != self._input:
            self._input = data
            self._current_trial = 0

        if self._trials is not None:
            if self._current_trial >= self._trials:
                self._state = ExecutionState.TRIALS_EXCEEDED
                self._output = f"{self._error_msg_trials}\nInput: {data}"
                return self._output
            else:
                self._current_trial += 1

        self.reset()
        data = self.pre_processing(data)
        self._output = self.__execute__(data)
        self._output = self.post_processing(self._output)
        return self._output

    def pre_execution(self):
        """Optional function executed once in the beginning to initialize necessary services"""
        pass

    def post_execution(self):
        """Optional function executed once at the end of subject live to stop all running services"""
        pass

    def pre_processing(self, data: str | list | tuple) -> str | list | tuple:
        """Optional function to preprocess the input data before passing it to the subject"""
        return data

    def post_processing(self, data: str | list | tuple) -> str | list | tuple:
        """Optional function to postprocess the output data after receiving it from the subject"""
        return data

    @staticmethod
    def static_encoding(data: Any | Sequence[Any]):
        if isinstance(data, (list, tuple)):
            data = [val.replace(" ", SPACE_TOKEN) if isinstance(val, str) else val for val in data]
            data = " ".join(data)
        return data

    @staticmethod
    def static_decoding(data: Any):
        if isinstance(data, str) and SPACE_TOKEN in data:
            data = data.split(" ")
            data = [d.replace(SPACE_TOKEN, " ") for d in data]
        return data

    def get_encoder(self) -> Callable[[Any], Any]:
        return self._encoder if self._encoder is not None else self.static_encoding

    def get_decoder(self) -> Callable[[Any], Any]:
        return self._decoder if self._decoder is not None else self.static_decoding


class CallableSubject(BaseSubject):
    """This is a basic class to test callable objects"""

    def __init__(self,
                 callable_func: Callable,
                 timeout: Optional[int] = None,
                 trials: Optional[int] = None,
                 quick_start: bool = True,
                 name: Optional[str] = None,
                 location: Optional[str] = None,
                 **kwargs):
        super().__init__(timeout, trials, quick_start, name, location, **kwargs)
        self.__callable_func__ = callable_func

    def __getstate__(self):
        state = super().__getstate__()
        state["callable_func"] = Pickle.to_bytes(self.__callable_func__)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__callable_func__ = Pickle.from_bytes(state["callable_func"])

    def __execute__(self, data: str | list | tuple) -> str | list:
        try:
            if self._timeout is not None:
                self._output = func_timeout(self.timeout, self.__callable_func__, args=(data,))
            else:
                self._output = self.__callable_func__(data)
        except FunctionTimedOut:
            self._state = ExecutionState.TIMEOUT
            self._output = f"{self._error_msg_timeout}\nExecution was terminated after {self._timeout} seconds.\nInput: {data}"
        except Exception as e:
            self._state = ExecutionState.EXCEPTION
            self._output = f"{self._error_msg_exception}\n{e}\nInput: {data}"
        else:
            self._state = ExecutionState.PASS
        return self._output


class ShellSubject(BaseSubject):
    """This is a basic class to test shell programs"""
    def __init__(self,
                 program_path: str,
                 arguments: list[str],
                 *,
                 input_is_last_argument: bool = True,
                 timeout: Optional[int] = None,
                 trials: Optional[int] = None,
                 quick_start: bool = False,
                 name: Optional[str] = None,
                 location: Optional[str] = None,
                 **kwargs):
        """
        Initialize the ShellObjectWrapper object
        :param program_path: path to the program to be tested
        :param arguments: list of arguments to be passed to the program
        :param input_is_last_argument: boolean flag to indicate if the input data is the last argument to be passed
        :param timeout: maximum time (in seconds) to wait for the program to complete
        """
        super().__init__(timeout, trials, quick_start, name, location, **kwargs)
        if input_is_last_argument:
            self._command = [program_path, *arguments, None]
            self._input_index = len(self._command) - 1
        else:
            self._command = [program_path, None, *arguments]
            self._input_index = 1

    def __getstate__(self):
        state = super().__getstate__()
        state["command"] = Pickle.to_bytes(self._command)
        state["input_index"] = self._input_index
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._command = Pickle.from_bytes(state["command"])
        self._input_index = state["input_index"]

    def __execute__(self, data: str | list | tuple) -> str | list:
        self._input = data
        proc = None
        args = self._command.copy()
        if isinstance(data, (list, tuple)):
            args = args[:self._input_index] + [str(x) for x in data] + args[self._input_index + 1:]
        else:
            args[self._input_index] = str(data)
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            stdout, stderr = proc.communicate(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            self._state = ExecutionState.TIMEOUT
            self._output = f"{self._error_msg_timeout}\nExecution was terminated after {self._timeout} seconds."
        except FileNotFoundError as e:
            self._state = ExecutionState.FILE_NOT_FOUND
            self._output = f"{self._error_msg_file}\n{e}"
        except Exception as e:
            self._state = ExecutionState.EXCEPTION
            self._output = f"{self._error_msg_exception}\n{e}"
        else:
            rc = proc.returncode if proc else None
            self._state = ExecutionState.PASS if rc == 0 else ExecutionState.FAIL
            self._output = f"{stdout}\n{stderr}" if stderr else stdout
        finally:
            # Ensure cleanup only if still running
            if proc and proc.poll() is None:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                except ProcessLookupError:
                    pass
                except BaseException:
                    try:
                        proc.kill()
                        proc.wait()
                    except BaseException:
                        pass
        return self._output


class RemoteSubject(BaseSubject):
    """This is a basic class to test remote subjects by sending requests"""
    def __init__(self,
                 method: str = "get",
                 timeout: Optional[int] = None,
                 trials: Optional[int] = None,
                 quick_start: bool = True,
                 name: Optional[str] = None,
                 location: Optional[str] = None,
                 **kwargs):
        super().__init__(timeout, trials, quick_start, name, location, **kwargs)
        self._min_wait_time = 0.1
        self._max_wait_time = 0.5
        self._request_method = requests.get if method.lower() == "get" else requests.post

    def __getstate__(self):
        state = super().__getstate__()
        state["use_get_request"] = True if self._request_method == requests.get else False
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._request_method = requests.get if state["use_get_request"] else requests.post

    @property
    def min_wait_time(self):
        return self._min_wait_time

    @min_wait_time.setter
    def min_wait_time(self, value):
        assert isinstance(value, int | float), f"min_wait_time must be an integer or float, got type={type(value)} instead."
        assert value > 0, f"min_wait_time must be a positive integer or float, got value={value} instead."
        self._min_wait_time = value

    @property
    def max_wait_time(self):
        return self._max_wait_time

    @max_wait_time.setter
    def max_wait_time(self, value):
        assert isinstance(value, int | float), f"max_wait_time must be an integer or float, got type={type(value)} instead."
        assert value >= self._min_wait_time, f"max_wait_time must be greater than or equal to min_wait_time, got {value} >= {self._min_wait_time} instead."
        self._max_wait_time = value

    def __execute__(self, data: str | list | tuple) -> str | list:
        assert len(data) > 0, "Input is empty"
        response = self.__request__(data)
        while isinstance(response, requests.Response) and response.status_code != 200:
            response = self.__request__(data)
        return self._output

    def __request__(self, data: str | list) -> requests.Response | str:
        self._current_trial += 1
        self._state = ExecutionState.UNDEFINED
        if self._current_trial > 1:
            sleep(random_uniform(self._min_wait_time * self._current_trial, self._max_wait_time * self._current_trial))
        try:
            response = self._request_method(data, timeout=self._timeout)
        except requests.exceptions.Timeout:
            self._state = ExecutionState.TIMEOUT
            response = f"{self._error_msg_timeout}\nExecution was terminated after {self._timeout} seconds."
        except Exception as e:
            self._state = ExecutionState.EXCEPTION
            response = f"{self._error_msg_exception}\n{e}"
        else:
            if isinstance(response, requests.Response):
                self._output = response.text
                self._state = ExecutionState.PASS
        return response
