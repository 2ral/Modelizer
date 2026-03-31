import os
import sys
import ast
import time
import hmac
import random
import atexit
import base64
import socket
import ctypes
import zipfile
import hashlib
import platform
import tempfile
import threading
import contextlib

import logging
import logging.config

import torch
import psutil
import cloudpickle

from enum import Enum
from queue import Queue
from pathlib import Path
from threading import Lock
from collections import deque
from datetime import datetime
from importlib import import_module
from importlib.util import find_spec
from re import compile as re_compile
from xxhash import xxh3_64, xxh3_128
from collections.abc import Iterable
from random import seed as random_seed
from gc import collect as garbage_collect
from multiprocessing import cpu_count, Pool
from inspect import signature, getattr_static
from logging.handlers import QueueHandler, QueueListener
from typing import Union, Iterator, Optional, Sequence, Any, Literal, Collection

from tqdm.auto import tqdm
from shutil import rmtree as shutil_rmtree, move as shutil_move

from modelizer.configs import SEED

try:
    import wandb
except ImportError:
    wandb = None


########################################################################################################################
#                                              Singleton metaclass                                                     #
########################################################################################################################
class SingletonMeta(type):
    """An implementation of Singleton pattern."""
    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


########################################################################################################################
#                                          Utility Class for Logging Tasks                                             #
########################################################################################################################
class LoggingLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LoggerConfig:
    """Data class to store the configuration for the logger."""
    def __init__(self,
                 name: str = "debug",
                 level: LoggingLevel = LoggingLevel.INFO,
                 *,
                 log_to_file: bool = True,
                 log_to_console: bool = True,
                 log_to_wandb: bool = False,
                 is_global_logger: bool = False,
                 overwrite: bool = False,
                 use_async: bool = False,
                 root_dir: str | Path | None = None,
                 log_format: str = '%(asctime)s | %(levelname)-8s | %(message)s',
                 log_date_format: str = '%d-%m-%Y %H:%M:%S'):
        """
        Initializes the LoggerConfig with the specified parameters.
        :param name: Name of the logger.
        :param level: Logging level.
        :param log_to_file: Whether to log to a file.
        :param log_to_console: Whether to log to the console.
        :param log_to_wandb: Whether to log to Weights and Biases.
        :param is_global_logger: Whether this is a global logger.
        :param overwrite: Whether to overwrite the log file if it exists.
        :param use_async: Whether to use asynchronous logging.
        :param root_dir: Directory where the log file will be stored.
        :param log_format: Format of the log messages.
        :param log_date_format: Date format of the log messages.
        """
        self._name = name
        self._level = level.value
        self._log_format = log_format
        self._log_date_format = log_date_format
        self._root_dir = Path(root_dir) if root_dir else Path.cwd()
        self._overwrite = overwrite
        self._log_to_file = log_to_file
        self._log_to_console = log_to_console
        self._log_to_wandb = log_to_wandb
        self._is_global_logger = is_global_logger
        self._use_async = use_async
        self._loggers: list["Logger"] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def level(self) -> int:
        return self._level

    @property
    def log_format(self) -> str:
        return self._log_format

    @property
    def log_date_format(self) -> str:
        return self._log_date_format

    @property
    def root_dir(self) -> str | Path | None:
        return self._root_dir

    @property
    def overwrite(self) -> bool:
        return self._overwrite

    @property
    def log_to_file(self) -> bool:
        return self._log_to_file

    @log_to_file.setter
    def log_to_file(self, value: bool):
        assert value and isinstance(value, bool), f"log_to_file must be a boolean value, got {type(value)}"
        if self._log_to_file != value:
            self._log_to_file = value
            for logger in self._loggers:
                if value:
                    logger.add_file_handler()
                else:
                    logger.remove_file_handler()

    @property
    def log_to_console(self) -> bool:
        return self._log_to_console

    @log_to_console.setter
    def log_to_console(self, value: bool):
        assert value and isinstance(value, bool), f"log_to_console must be a boolean value, got {type(value)}"
        if self._log_to_console != value:
            self._log_to_console = value
            for logger in self._loggers:
                if value:
                    logger.add_console_handler()
                else:
                    logger.remove_console_handler()

    @property
    def log_to_wandb(self) -> bool:
        return self._log_to_wandb

    @log_to_wandb.setter
    def log_to_wandb(self, value: bool):
        assert value and isinstance(value, bool), f"log_to_wandb must be a boolean value, got {type(value)}"
        if self._log_to_wandb != value:
            self._log_to_wandb = value
            for logger in self._loggers:
                if value:
                    logger.add_wandb_handler()
                else:
                    logger.remove_wandb_handler()

    @property
    def is_global_logger(self) -> bool:
        return self._is_global_logger

    @property
    def use_async(self) -> bool:
        return self._use_async

    def register_logger(self, logger: "Logger"):
        if logger not in self._loggers:
            self._loggers.append(logger)

    def unregister_logger(self, logger: "Logger"):
        try:
            self._loggers.remove(logger)
        except ValueError:
            pass


class FlushStreamHandler(logging.StreamHandler):
    """A logging StreamHandler that flushes the stream after each log message."""
    def __init__(self, stream=sys.stdout):
        super().__init__(stream)

    def emit(self, record):
        super().emit(record)
        self.flush()  # Ensure flush immediately


class WandbHandler(logging.Handler):
    """A simple logging Handler that sends logs to Weights & Biases."""
    def __init__(self, key='logs', level=logging.INFO, debug: bool = False):
        super().__init__(level)
        self.key = key
        self.debug = debug

    def emit(self, record):
        log_entry = self.format(record)
        try:
            if wandb is not None and getattr(wandb, "run", None) is not None and not getattr(wandb.run, "finished", False):
                wandb.log({self.key: log_entry})
        except Exception as e:
            if self.debug:
                print(f"[WandbHandler] Error logging to wandb: {e}")


class Logger:
    """A flexible logger class that can log to console, file, and Weights & Biases."""
    _instances = {}
    _lock = Lock()

    def __new__(cls, config: LoggerConfig = None):
        if config and config.is_global_logger:
            with cls._lock:
                if "global" not in cls._instances:
                    cls._instances["global"] = super().__new__(cls)
                return cls._instances["global"]
        return super().__new__(cls)

    def __init__(self, config: LoggerConfig = None):
        if hasattr(self, '_initialized'):
            return  # avoid reinitializing

        if config is None or (not config.log_to_console and not config.log_to_file and not config.log_to_wandb):
            self.logger = logging.getLogger("__null_logger__")
            self.logger.handlers.clear()
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False
            self._handlers: list[logging.Handler] = []
            self._queue_handler = None
            self.listener = None
            self._initialized = True
            return

        self.config = config
        self._handlers: list[logging.Handler] = []
        self._queue_handler = None
        self.listener = None

        self.logger = logging.getLogger(config.name)
        self.logger.setLevel(config.level)
        self.logger.propagate = False
        self._detach_modelizer_handlers(self.logger)

        handlers = []
        formatter = logging.Formatter(config.log_format, config.log_date_format)

        if config.log_to_console:
            ch = FlushStreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(config.level)
            self._tag_handler(ch)
            handlers.append(ch)

        if config.log_to_file:
            config.root_dir.mkdir(parents=True, exist_ok=True)
            filepath = config.root_dir / f"{config.name}.log"
            mode = 'w' if config.overwrite else 'a'

            fh = logging.FileHandler(filepath, mode=mode, encoding='utf-8')
            fh.setFormatter(formatter)
            fh.setLevel(config.level)
            self._tag_handler(fh)
            handlers.append(fh)

        if config.log_to_wandb and wandb is not None:
            wh = WandbHandler(debug=False)
            wh.setFormatter(formatter)
            wh.setLevel(logging.INFO)
            self._tag_handler(wh)
            handlers.append(wh)

        if self.config.use_async and handlers:
            self.log_queue = Queue(-1)
            qh = QueueHandler(self.log_queue)
            self._tag_handler(qh, is_queue=True)
            self.logger.addHandler(qh)
            self._queue_handler = qh

            self.listener = QueueListener(self.log_queue, *handlers, respect_handler_level=True)
            self.listener.start()
            self._handlers = handlers  # real sinks owned by listener
        else:
            for h in handlers:
                self.logger.addHandler(h)
            self._handlers = handlers

        if not getattr(self, "_at_exit_registered", False):
            atexit.register(self.shutdown)
            self._at_exit_registered = True

        self.config.register_logger(self)
        self._initialized = True

    @property
    def is_null_logger(self):
        return self.logger.name == "__null_logger__"

    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)

    def shutdown(self):
        listener = getattr(self, 'listener', None)
        if listener:
            try:
                listener.stop()
            except Exception:
                pass
            self.listener = None

        if getattr(self, '_queue_handler', None) is not None:
            try:
                self.logger.removeHandler(self._queue_handler)
            except Exception:
                pass
            self._safe_close(self._queue_handler)
            self._queue_handler = None

        for h in list(self.logger.handlers):
            if getattr(h, "_modelizer_managed", False):
                try:
                    self.logger.removeHandler(h)
                except Exception:
                    pass
                self._safe_close(h)

        for h in getattr(self, "_handlers", []):
            self._safe_close(h)
        self._handlers = []
        self.config.unregister_logger(self)

    @staticmethod
    def forge(value):
        if isinstance(value, LoggerConfig):
            return Logger(value)
        elif isinstance(value, Logger):
            return value
        elif value is None:
            return Logger(None)
        else:
            raise TypeError(f"Unsupported logger type: {type(value)}")

    @staticmethod
    def _safe_close(handler):
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass

    @staticmethod
    def _detach_modelizer_handlers(logger):
        for h in list(logger.handlers):
            if getattr(h, "_modelizer_managed", False):
                try:
                    logger.removeHandler(h)
                except Exception:
                    pass
                Logger._safe_close(h)

    @staticmethod
    def _tag_handler(handler, is_queue: bool = False):
        setattr(handler, "_modelizer_managed", True)
        if is_queue:
            setattr(handler, "_modelizer_queue", True)

    def _build_formatter(self) -> logging.Formatter:
        return logging.Formatter(self.config.log_format, self.config.log_date_format)

    def _ensure_not_null(self):
        if self.is_null_logger:
            raise RuntimeError("Cannot modify handlers on a null logger")

    def _is_handler_present(self, handler_type: type[logging.Handler]) -> bool:
        """Check if any managed handler is an instance of `handler_type`."""
        return any(isinstance(h, handler_type) for h in self._handlers)

    def _register_handler(self, handler: logging.Handler) -> None:
        """Attach handler respecting async/non-async mode and track it."""
        self._tag_handler(handler)
        if self.config.use_async and self.listener is not None:
            # In async mode, handlers are owned by the listener
            self.listener.handlers = (*self.listener.handlers, handler)
        else:
            self.logger.addHandler(handler)
        self._handlers.append(handler)

    def _unregister_handler(self, handler_type: type[logging.Handler]) -> None:
        """
        Remove first handler of a given type from logger/listener and close it.
        `handler_type` is a handler *class*, `target` is an instance.
        """
        target: logging.Handler | None = None
        for h in list(self._handlers):
            if isinstance(h, handler_type):
                target = h
                break
        if target is None:
            return

        if self.config.use_async and self.listener is not None:
            self.listener.handlers = tuple(h for h in self.listener.handlers if h is not target)
        else:
            try:
                self.logger.removeHandler(target)
            except Exception:
                pass

        self._handlers.remove(target)
        self._safe_close(target)

    def add_console_handler(self) -> None:
        if not self.is_null_logger:
            if self._is_handler_present(FlushStreamHandler):
                raise RuntimeError("Console handler already exists for this logger")
    
            formatter = self._build_formatter()
            ch = FlushStreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(self.config.level)
            self._register_handler(ch)

    def remove_console_handler(self) -> None:
        if not self.is_null_logger:
            self._unregister_handler(FlushStreamHandler)

    def add_file_handler(self) -> None:
        if not self.is_null_logger:
            if self._is_handler_present(logging.FileHandler):
                raise RuntimeError("FileHandler already exists for this logger")
    
            formatter = self._build_formatter()
            self.config.root_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.config.root_dir / f"{self.config.name}.log"
            mode = "w" if self.config.overwrite else "a"
    
            fh = logging.FileHandler(filepath, mode=mode, encoding="utf-8")
            fh.setFormatter(formatter)
            fh.setLevel(self.config.level)
            self._register_handler(fh)

    def remove_file_handler(self) -> None:
        if not self.is_null_logger:
            self._unregister_handler(logging.FileHandler)

    def add_wandb_handler(self) -> None:
        if not self.is_null_logger:
            if wandb is None:
                raise RuntimeError("wandb is not available; cannot add WandbHandler")
    
            if self._is_handler_present(WandbHandler):
                raise RuntimeError("WandbHandler already exists for this logger")
    
            formatter = self._build_formatter()
            wh = WandbHandler(debug=False)
            wh.setFormatter(formatter)
            wh.setLevel(logging.INFO)
            self._register_handler(wh)

    def remove_wandb_handler(self) -> None:
        if not self.is_null_logger:
            self._unregister_handler(WandbHandler)


########################################################################################################################
#                           Utility classes and functions to save and load objects                                     #
########################################################################################################################
class Pickle(metaclass=SingletonMeta):
    _orig_torch_load = torch.load

    """Utility class that provides helper methods for saving and loading objects using pickle."""
    @staticmethod
    def dump(obj, path: str | Path):
        """
        Saves the object to a pickle file.
        :param obj: Object to be saved
        :param path: string or pathlib.Path object to save the object to
        """
        with open(path, "wb") as f:
            cloudpickle.dump(obj, f)

    @staticmethod
    def torch_load_cpu_helper(*args, **kwargs):
        """Internal helper to force torch.load to CPU unless map_location is explicitly given."""
        if "map_location" not in kwargs or kwargs["map_location"] is None:
            kwargs["map_location"] = torch.device("cpu")
        return Pickle._orig_torch_load(*args, **kwargs)

    @staticmethod
    def load(path: str | Path):
        """
        Loads the object from a pickle file.
        :param path: string or pathlib.Path object to load the object from
        :return: The loaded object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        try:
            torch.load = Pickle.torch_load_cpu_helper  # type: ignore[assignment]
            with open(path, "rb") as f:
                obj = cloudpickle.load(f)
        except:
            torch.load = Pickle._orig_torch_load
            raise
        else:
            torch.load = Pickle._orig_torch_load

        return obj

    @staticmethod
    def to_bytes(obj):
        """
        Converts the object to bytes.
        :param obj: Object to be converted
        :return: Bytes representation of the object
        """
        return cloudpickle.dumps(obj)

    @staticmethod
    def from_bytes(byte_stream: bytes):
        """
        Converts the bytes to an object.
        :param byte_stream: Bytes to be converted
        :return: The object
        """
        return cloudpickle.loads(byte_stream)


def load_module(input_dir: str | Path, logger: Optional[Logger] = None, force_cpu: bool = False):
    """
    Loads and initializes the module from the given directory.
    :param input_dir: The directory containing the module and config file
    :param logger: optional Logger object for logging
    :param force_cpu: Whether to force loading on CPU
    :return: The initialized module
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        error_msg = f"Directory not found: {input_dir}"
        if logger is not None:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    config = Pickle.load(input_dir / "config.pkl")
    factory = getattr(import_module(config["module"]), config["class"])
    method = "from_pretrained"  # check if the factory class defines a static method
    if isinstance(getattr_static(factory, method, None), staticmethod):
        factory = getattr(factory, method)
    if "logger" in signature(factory).parameters:
        if "force_cpu" in signature(factory).parameters:
            return factory(input_dir, logger=logger, force_cpu=force_cpu)
        return factory(input_dir, logger=logger)
    if "force_cpu" in signature(factory).parameters:
        return factory(input_dir, force_cpu=force_cpu)
    return factory(input_dir)


########################################################################################################################
#                                      Utility classes for Multiprocessing Tasks                                         #
########################################################################################################################
class Multiprocessing(metaclass=SingletonMeta):
    """Utility class that provides helper methods for multiprocess parallelization using multiprocessing.Pool."""
    CORE_COUNT = cpu_count() - 1 if cpu_count() > 2 else cpu_count()

    @staticmethod
    def parallel_run(function, data, text: str = None, n_jobs: int = -1, chunkify: bool = False, *, logger: Optional[Logger] = None) -> list:
        """Helper method that parallelizes execution of the specified function using multiprocessing.Pool."""
        assert len(data) > 0, "Cannot parallelize computation for an empty list"
        if n_jobs <= 0:
            n_jobs = len(data) if Multiprocessing.CORE_COUNT > len(data) else Multiprocessing.CORE_COUNT

        if chunkify:
            data = list(Multiprocessing.generator_fixed_size_chunks(data, n_jobs))

        if logger is not None:
            logger.info(f"Running {function.__name__} in parallel with {n_jobs} jobs. "
                        f"Data size: {len(data)}. Core Count: {Multiprocessing.CORE_COUNT}")

        if text is not None:
            data = list(tqdm(data, total=len(data), desc=text, leave=False))

        use_starmap = all(isinstance(item, Iterable) for item in data)
        with Pool(processes=n_jobs) as pool:
            if use_starmap:
                results = pool.starmap(function, data)
            else:
                results = pool.map(function, data)
        return results

    @staticmethod
    def generator_fixed_size_chunks(d_list: list[Any], elems_per_chunk: int = CORE_COUNT) -> Iterator[list[Any]]:
        """
        Generate chunks from the input list with a fixed number of elements per chunk.
        :param d_list: The list to be chunked.
        :param elems_per_chunk: The number of elements per chunk. Default is CORE_COUNT.
        :return: A generator yielding sublists of d_list.
        """
        for i in range(0, len(d_list), elems_per_chunk):
            yield d_list[i:i + elems_per_chunk]

    @staticmethod
    def generator_fixed_number_of_chunks(d_list: list[Any], n_chunks: int = CORE_COUNT) -> Iterator[list[Any]]:
        """
        Split the input list into a specified number of chunks as evenly as possible.
        :param d_list: The list to be split.
        :param n_chunks: The number of chunks. Default is CORE_COUNT.
        :return: A generator yielding sublists.
        """
        k, m = divmod(len(d_list), n_chunks)
        for i in range(n_chunks):
            yield d_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]


class StoppableThread(threading.Thread):
    """Cooperative stop via `stop_event`; CPython-only kill as last resort."""
    def __init__(self, target=None, args=(), kwargs=None, name=None, daemon=None):
        self.stop_event = threading.Event()
        self._user_target = target
        self._user_args = args or ()
        self._user_kwargs = dict(kwargs or {})
        super().__init__(target=self._run_with_stop, name=name, daemon=daemon)

    def _run_with_stop(self):
        target = self._user_target
        if target is None:
            return None
        kwargs = dict(self._user_kwargs)
        try:
            if 'stop_event' in signature(target).parameters and 'stop_event' not in kwargs:
                kwargs['stop_event'] = self.stop_event
        except Exception:
            pass
        return target(*self._user_args, **kwargs)

    def request_stop(self):
        """Signal cooperative stop to the running target."""
        self.stop_event.set()

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def stop(self, timeout: float | None = None, force: bool = False) -> bool:
        """
        Request stop and join. If still alive and force=True, attempt hard kill.
        Returns True if the thread is no longer alive.
        """
        self.stop_event.set()
        self.join(timeout)
        if self.is_alive() and force:
            self.kill()
            self.join(1.0)
        return not self.is_alive()

    def kill(self) -> bool:
        """
        Hard-kill the thread by raising SystemExit in its context.
        Use only as a last resort.
        """
        ident = self.ident
        if not ident or not self.is_alive():
            return False
        tid = ctypes.c_long(ident)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
        return res == 1


########################################################################################################################
#                                           Utility Class for Memory Usage                                             #
########################################################################################################################
class MemInfo(metaclass=SingletonMeta):
    """Utility class that provides helper methods for memory usage tracking."""
    @staticmethod
    def format_memory_usage(memory: int):
        """Converts memory usage to human-readable format."""
        if memory < 1024 ** 3:
            memory /= 1024 ** 2
            return f"{memory:.2f} MB"
        else:
            memory /= 1024 ** 3
            return f"{memory:.2f} GB"

    @staticmethod
    def clean_cache():
        """Cleans the PyTorch cache and runs garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        garbage_collect()

    @staticmethod
    def get_memory_usage() -> str:
        """Returns the memory usage of the current process in a human-readable format."""
        return MemInfo.format_memory_usage(psutil.Process(os.getpid()).memory_info().rss)

    @staticmethod
    def get_memory_usage_stats(task_type: str) -> str:
        if torch.cuda.is_available():
            message = (f"{task_type} | Memory Allocated: {MemInfo.format_memory_usage(torch.cuda.memory_allocated(0))}"
                       f" | Memory Reserved: {MemInfo.format_memory_usage(torch.cuda.memory_reserved(0))}")
        elif torch.backends.mps.is_available():
            message = (f"{task_type} | Memory Allocated: {MemInfo.format_memory_usage(torch.mps.current_allocated_memory())}"
                       f" | Memory Reserved: {MemInfo.format_memory_usage(torch.mps.driver_allocated_memory())}")
        else:
            message = f"{task_type} | Memory Allocated: {MemInfo.get_memory_usage()}"
        return message

    @staticmethod
    def get_used_memory(device: Optional[torch.device] = None) -> float:
        """
        Returns the used memory in MB. If cuda device is specified, limits the check scope.
        :param device: Optional torch.device object to specify the device type.
        :return: Used memory in MB.
        """
        try:
            if device is not None:
                if device.type == 'cuda':
                    if torch.cuda.is_available():
                        used = torch.cuda.memory_allocated(device) / 1024 ** 2
                    else:
                        used = 0.
                else:
                    used = psutil.Process().memory_info().rss / 1024 ** 2
            elif torch.cuda.is_available():
                device = torch.cuda.current_device()
                used = torch.cuda.memory_allocated(device) / 1024 ** 2
            else:
                used = psutil.Process().memory_info().rss / 1024 ** 2
        except Exception:
            used = 0.
        return used

    @staticmethod
    def get_available_memory(device: Optional[torch.device] = None) -> float:
        """
        Returns the available memory in MB. If cuda device is specified, limits the check scope.
        :param device: Optional torch.device object to specify the device type.
        :return: Available memory in MB.
        """
        try:
            if device is not None:
                if device.type == 'cuda':
                    if torch.cuda.is_available():
                        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                        available = free_bytes / 1024 ** 2
                    else:
                        available = 0.
                else:
                    vm = psutil.virtual_memory()
                    available = vm.available / 1024 ** 2
            elif torch.cuda.is_available():
                device = torch.cuda.current_device()
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                available = free_bytes / 1024 ** 2
            else:
                vm = psutil.virtual_memory()
                available = vm.available / 1024 ** 2
        except Exception:
            available = 0.
        return available


class MemoryTracker:
    """Utility class that tracks peak memory usage during a specific operation."""
    def __init__(self, device: Optional[Union[torch.device, str]] = None,
                 duration: Optional[int] = None, interval: float = 2.0, model_config=None):
        """
        Initializes the MemoryTracker.
        :param device: The device to monitor (e.g., 'cuda', 'mps', 'cpu', or torch.device). If None, auto-detects.
        :param duration: The duration (in seconds) to track memory usage. If None, tracks until stopped.
        :param interval: The interval (in seconds) between memory usage checks.
        :param model_config: Optional handler to the model configuration for memoru usage logging.
        """
        if isinstance(device, str):
            if "cuda" in device:
                if torch.cuda.is_available():
                    device = torch.device(device)
                else:
                    raise RuntimeError("CUDA is not available on this system.")
            elif device == "mps":
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    raise RuntimeError("MPS is not available on this system.")
            elif device == "cpu":
                device = torch.device("cpu")
            else:
                raise ValueError(f"Unsupported device string: {device}")
        elif isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system.")
            elif device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available on this system.")
        elif device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            raise TypeError(f"Unsupported device type: {type(device)} => {device}")

        self._device = device
        self._duration = duration
        self._interval = interval
        self._is_tracking = False
        self._peak_memory: float = 0.
        self._last_memory: float = 0.
        self._avg_memory: float = 0.
        self._num_samples: int = 0
        self.__model_config__ = model_config
        self.__memory_thread__: Optional[threading.Thread] = None

    @property
    def peak_memory_usage(self) -> float:
        """Returns the peak memory usage recorded during tracking in MB."""
        return self._peak_memory

    @property
    def last_memory_usage(self) -> float:
        """Returns the last recorded memory usage in MB."""
        return self._last_memory

    @property
    def average_memory_usage(self) -> float:
        """Returns the average memory usage recorded during tracking in MB."""
        return self._avg_memory

    @property
    def memory_usage_statistics(self) -> dict[str, float]:
        """Returns a dictionary containing peak, last, and average memory usage statistics in MB."""
        return {"peak": self._peak_memory, "last": self._last_memory, "average": self._avg_memory}

    @property
    def is_tracking(self) -> bool:
        """Returns whether memory tracking is currently active."""
        return self._is_tracking

    @property
    def device(self) -> torch.device:
        """Returns the device being monitored for memory usage."""
        return self._device

    @property
    def duration(self) -> Optional[int]:
        """Returns the duration for which memory tracking is set."""
        return self._duration

    @duration.setter
    def duration(self, value: int | float | None):
        """Sets the duration for which memory tracking is to be performed. If None, tracks indefinitely until stopped."""
        self._duration = value

    @property
    def interval(self) -> float:
        """Returns the interval between memory usage checks in seconds."""
        return self._interval

    @interval.setter
    def interval(self, value: float | int):
        """Sets the interval between memory usage checks in seconds."""
        assert isinstance(value, (float, int)), "Interval must be a float or int representing seconds."
        self._interval = float(value)

    def __track_memory__(self):
        start_time = time.time()

        while self._is_tracking and (self._duration is None or (time.time() - start_time < self._duration)):
            current_usage = self._get_current_memory()
            self._num_samples += 1
            self._avg_memory += (current_usage - self._avg_memory) / self._num_samples
            if self._device.type == 'cuda' and torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated(self._device) / 1024 ** 2
            else:
                peak_memory = 0.
            self._peak_memory = max(self._peak_memory, peak_memory, current_usage)
            if self.__model_config__ is not None and self.__model_config__.memory_requirements < self._peak_memory:
                self.__model_config__.memory_requirements = self._peak_memory
                self.__model_config__.kwargs["memory_statistics"] = self.memory_usage_statistics
            time.sleep(self._interval)

    def _get_current_memory(self):
        if self._device.type == 'cuda' and torch.cuda.is_available():
            usage = torch.cuda.memory_allocated(self._device) / 1024 ** 2  # MB
        elif self._device.type == 'mps' and torch.backends.mps.is_available():
            usage = psutil.Process().memory_info().rss / 1024 ** 2  # MB
        elif self._device.type == 'cpu':
            usage = psutil.Process().memory_info().rss / 1024 ** 2  # MB
        else:
            raise RuntimeError(f"Unsupported device type: {self._device}")
        self._last_memory = usage
        return usage

    def start_tracking(self):
        """Start tracking memory usage in a separate thread."""
        if self.__memory_thread__ is None:
            self._is_tracking = True
            self.__memory_thread__ = threading.Thread(target=self.__track_memory__)
            self.__memory_thread__.start()
        else:
            print("Memory tracking is already running.")

    def stop_tracking(self):
        """Stop tracking memory usage."""
        self._is_tracking = False
        if hasattr(self, 'memory_thread'):
            self.__memory_thread__.join()
        self.__memory_thread__ = None


########################################################################################################################
#                                           Utility Methods for Port Checking                                          #
########################################################################################################################
def check_port(port: int, host: str = 'localhost', timeout: int = 3):
    """
    Helper method to check if the port is available for binding.
    :param port: port number to check
    :param host: host address at which the port is to be checked
    :param timeout: timeout in seconds to wait before failing
    :return: True if the port is available, False otherwise
    """
    if not isinstance(port, int) or not (0 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        # `bind` doesn't use the timeout, but keeping it is harmless.
        sock.settimeout(timeout)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(start_port: int = 10000, max_port: int = 65535, host: str = "localhost") -> int:
    """Find the first available port on `host` starting from `start_port`."""
    if not (0 <= start_port <= 65535 and 0 <= max_port <= 65535 and start_port <= max_port):
        raise ValueError(f"Invalid port range: {start_port}..{max_port}")
    time.sleep(random.uniform(0.1, 3.0))
    for port in range(start_port, max_port + 1):
        if check_port(port, host=host):
            return port
    raise RuntimeError("No free ports available in the specified range")


########################################################################################################################
#                                       Utility Methods to Check Installation                                          #
########################################################################################################################
def _is_package_installed(package: str) -> bool:
    """Return True if *package* is importable, safely handling the case where
    the package is already loaded but its ``__spec__`` is ``None`` (which causes
    ``importlib.util.find_spec`` to raise ``ValueError``)."""
    try:
        return find_spec(package) is not None
    except ValueError:
        # Package is present in sys.modules but __spec__ is None; treat as installed.
        import sys
        return package in sys.modules


def check_installation(packages: Sequence[str] | str) -> dict[str, bool]:
    """Helper method that checks if a specified Python module is installed."""
    if isinstance(packages, str):
        return {packages: _is_package_installed(packages)}
    return {pck: _is_package_installed(pck) for pck in packages}


########################################################################################################################
#                                       Utility Methods to Check Execution                                             #
########################################################################################################################
def retrieve_init_arguments() -> str:
    filtered_args = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] in ['--huggingface-key', 'openai-key', '--wandb']:
            # Skip the current argument and the next one (the value)
            i += 2
        else:
            filtered_args.append(sys.argv[i])
            i += 1

    return " ".join(filtered_args)


########################################################################################################################
#                                  Utility Class for Operations with PyTorch                                           #
########################################################################################################################
class TorchHelpers(metaclass=SingletonMeta):
    """Utility class that provides helper methods for operations with PyTorch."""
    dtype_sequence = {
        torch.uint8: torch.int16,
        torch.int16: torch.uint16,
        torch.uint16: torch.int32,
        torch.int32: torch.uint32,
        torch.uint32: torch.int64,
        torch.int64: torch.uint64,
        torch.uint64: None
    }

    dtype_priority = {
        torch.uint8: 0,
        torch.int16: 1,
        torch.uint16: 2,
        torch.int32: 3,
        torch.uint32: 4,
        torch.int64: 5,
        torch.uint64: 6
    }

    @staticmethod
    def is_dtype_greater(left: torch.dtype, right: torch.dtype) -> bool:
        """
        This function checks if the left dtype is greater than the right dtype.
        :param left: The left dtype
        :param right: The right dtype
        :return: True if left is greater than right, False otherwise
        """
        return TorchHelpers.dtype_priority[left] > TorchHelpers.dtype_priority[right]

    @staticmethod
    def initialize_cuda(local_rank: Optional[int] = None):
        """
        This function initializes the CUDA backend for the model training and inference.
        :param local_rank: the local rank of the GPU. Default is None.
        :return: torch.device object
        """
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Error while emptying CUDA cache: {e}")
        device = torch.device("cuda", local_rank)
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision('high')
        return device

    @staticmethod
    def initialize_device(seed: int = SEED,
                          logger: Optional[Logger] = None,
                          *,
                          force_cpu: bool = False) -> torch.device:
        """
        This function initializes the torch.device for the model training and inference.
        :param seed: seed value for reproducibility
        :param logger: Logger object for logging
        :param force_cpu: enforces loading model to CPU
        :return: torch.device object
        """
        try:
            torch.manual_seed(seed)
        except RuntimeError as e:
            if logger is not None:
                logger.error(f"Error while setting seed: {e}")
        finally:
            random_seed(seed)

        if force_cpu:
            device = torch.device("cpu")
            if logger is not None:
                logger.info("Forcing CPU backend")

        elif torch.cuda.is_available():
            device = TorchHelpers.initialize_cuda()
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                if logger is not None:
                    logger.error("CUDA seed setting failed during device initialization")
            else:
                if logger is not None:
                    logger.info("CUDA backend enabled")

        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            if logger is not None:
                logger.info("MPS backend enabled")

        else:
            device = torch.device("cpu")
            if logger is not None:
                logger.info("No supported backend found, using CPU")

        return device

    @staticmethod
    def initialize_distributed(
            backend: str = "nccl",
            address: str = "localhost",
            port: int = 50005,
            world_size: int = 1,
            rank: int = 0,
            seed: int = SEED,
            logger: Optional[Logger] = None) -> torch.device:
        """
        This function initializes the distributed training environment for the model.
        :param backend: Backend type for distributed training. Possible values: "nccl", "gloo", "mpi"
        :param address: Address of the master node
        :param port: port Number for the master node
        :param world_size: Number of GPUs participating in distributed training
        :param rank: Unique identifier for each GPU in distributed training. 0 <= rank < world_size
        :param seed: Seed value for reproducibility. Default is configs.SEED
        :param logger: (Optional) Logger object for logging
        :return: torch.device object
        """

        assert all(isinstance(x, str) for x in (backend, address)), "Backend and address must be strings"
        assert all(isinstance(x, int) for x in (port, world_size, rank)), "port, world_size, and rank must be integers"

        old_port = port

        if not check_port(port):
            port = find_free_port()

        if port != old_port and logger is not None:
            logger.warning(f"Port {old_port} is already in use. Using port {port} instead.")

        os.environ.setdefault("MASTER_ADDR", address)
        os.environ.setdefault("MASTER_PORT", str(port))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if backend == "nccl":
            assert torch.cuda.is_available(), "nccl backend requires CUDA support"
            torch.cuda.set_device(local_rank)
            device = TorchHelpers.initialize_cuda(local_rank)

        elif backend in ("gloo", "mpi"):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = TorchHelpers.initialize_cuda(local_rank)
            else:
                device = torch.device("cpu")

        else:
            error_msg = "Invalid backend specified"
            if logger is not None:
                logger.error(error_msg)
            raise ValueError(error_msg)

        torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank)
        if logger is not None:
            logger.info(f"Distributed training enabled with {world_size} GPUs using {backend} backend")

        try:
            torch.manual_seed(seed)
        except RuntimeError as e:
            if logger is not None:
                logger.error(f"Error while setting seed: {e}")
        finally:
            random_seed(seed)
        try:
            import numpy as _np  # type: ignore
        except Exception:
            _np = None
        else:
            _np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return device

    @staticmethod
    def find_half_precision_weights_dtype() -> torch.dtype:
        """
        This functions finds the floating point dtype for half precision training.
        :return: torch.dtype object
        """
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @staticmethod
    def find_flash_attention_type() -> str:
        """This function finds the Flash Attention type based on the GPU architecture and the availability of the Flash Attention library."""
        return "flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and check_installation("flash-attn") else "eager"


########################################################################################################################
#                                       Utility functions for runtime logging                                          #
########################################################################################################################
def get_time_diff(start_time: datetime) -> str:
    """
    This function calculates the time difference between the current time and the start time.
    :param start_time: The start time as a datetime object
    """
    time_diff = datetime.now() - start_time
    hours = time_diff.seconds // 3600
    minutes = (time_diff.seconds % 3600) // 60
    seconds = time_diff.seconds % 60

    messages = []
    if hours > 0:
        messages.append(f"{hours} hours")
    if minutes > 0:
        messages.append(f"{minutes} minutes")
    if seconds > 0:
        messages.append(f"{seconds} seconds")
    if len(messages) > 0:
        total_time = ", ".join(messages)
    else:
        total_time = "0 seconds"

    return total_time


########################################################################################################################
#                                       Utility functions for data handling                                            #
########################################################################################################################
class DataHandlers(metaclass=SingletonMeta):
    """Utility class that provides helper methods for data handling."""
    REPLACE_WHITESPACE_PATTERN = re_compile(r"\s+")
    REPLACE_SPACE_COMMA_PATTERN = re_compile(r'(?<!,) ')

    @staticmethod
    def stringify(val):
        """
        This function converts the input value to a plain string.
         If the input is already a string, it tries to preprocess it as a Python literal.
        :param val: The input value to be converted
        :return: A plain string with list elements joined by spaces
        """
        if isinstance(val, str):
            try:
                val = ast.literal_eval(val)
            except Exception:
                val = [val]

        if isinstance(val, (list, tuple)):
            return " ".join(map(str, val))

        return str(val)

    @staticmethod
    def replace_spaces_except_after_comma(s: str) -> str:
        """
        Replace spaces with '<|spc|>'.
        Rules:
        - Replace every space with '<|spc|>', except space after a comma (", ") when it is outside parentheses AND outside angle brackets.
        - If the comma-space is inside parentheses "(...)" OR inside angle brackets "<...>", replace that space with '<|spc|>'.
        - Angle brackets are treated as non-nesting spans: '<' opens if not already inside, '>' closes only if currently inside.
        """
        out: list[str] = []
        paren_depth = 0
        in_angle = False
        prev = ""

        for ch in s:
            if ch == "(":
                paren_depth += 1
                out.append(ch)
            elif ch == ")":
                if paren_depth:
                    paren_depth -= 1
                out.append(ch)
            elif ch == "<":
                if not in_angle:
                    in_angle = True
                out.append(ch)
            elif ch == ">":
                if in_angle:
                    in_angle = False
                out.append(ch)
            elif ch == " ":
                if prev == "," and paren_depth == 0 and not in_angle:
                    out.append(" ")
                else:
                    out.append("<|spc|>")
            else:
                out.append(ch)

            prev = ch

        return "".join(out)

    @staticmethod
    def replace_whitespace(s):
        """
        This function replaces all whitespace characters in a string with '<|spc|>'.
        :param s: candidate string to replace whitespace characters in
        :return: updated string with whitespace characters replaced by '<|spc|>'
        """
        return DataHandlers.REPLACE_WHITESPACE_PATTERN.sub('<|spc|>', s)

    @staticmethod
    def post_formating(value):
        if isinstance(value, str):
            result = DataHandlers.replace_spaces_except_after_comma(value)
        elif isinstance(value, list | tuple):
            if not all(isinstance(elem, str) for elem in value):
                result = " ".join([DataHandlers.replace_spaces_except_after_comma(str(elem)) for elem in value])
            else:
                result = " ".join([DataHandlers.replace_spaces_except_after_comma(elem) for elem in value])
        else:
            result = DataHandlers.replace_spaces_except_after_comma(str(value))
        return result

    @staticmethod
    def deduplicate_keep_first(items: Iterable, exclude: Optional[Collection[Any]] = None) -> list:
        exclude_set = set(exclude if exclude is not None else tuple())
        seen = set()
        output = list()
        for x in items:
            if x in exclude_set:
                output.append(x)
                continue
            elif x in seen:
                continue
            else:
                output.append(x)
                seen.add(x)
        return output

    @staticmethod
    def locate_temp_dir() -> Path:
        return Path('/tmp' if platform.system() in ['Linux', 'Darwin'] else tempfile.gettempdir())

    @staticmethod
    def recursive_directory_lookup(directory: Path) -> Path | None:
        """
        Recursively searches for the first valid file in a directory tree,
        skipping '.DS_Store' and preferring top-level files first.
        :param directory: The root directory to start the search.
        :return: Path or None: The first valid file found, or None if no valid files exist.
        """
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a valid directory")

        stack = [directory]
        while stack:
            current_dir = stack.pop()
            try:
                entries = list(current_dir.iterdir())
            except Exception:
                continue
            files = [item for item in entries if item.is_file() and item.name.lower() != '.ds_store']
            if files:
                return files[0]
            subdirs = [item for item in entries if item.is_dir()]
            stack.extend(sorted(subdirs, reverse=True))
        return None

    @staticmethod
    def move_to_temp_folder(src_path: str | Path, subject_name: Optional[str] = None, *, clean: bool = False):
        """
        This utility function moves the specified file or folder to modelizer directory in the temp folder.
        :param src_path: Path to the source file or folder
        :param subject_name: Optional subject name to be used in the target directory name
        :param clean: If True, remove the existing modelizer directory before moving
        """
        src = Path(src_path)
        target_dir = DataHandlers.locate_temp_dir().joinpath("modelizer")
        assert src.exists(), f"src_path={src_path} does not exist"
        if src.is_file or len(list(src.iterdir())) > 0:  # src is a directory
            if subject_name is not None:
                target_dir = target_dir.joinpath(subject_name)

            if clean and target_dir.exists():
                if target_dir.is_file():
                    target_dir.unlink()
                else:
                    shutil_rmtree(target_dir)

            target_dir.mkdir(parents=True, exist_ok=True)
            target_dir = target_dir / src.name
            shutil_move(src, target_dir)

    @staticmethod
    def unzip(filepath: str | Path, use_tmp_directory: bool = True) -> Path | None:
        """
        Unzips a zip file and returns the first valid filepath found in the extracted directory.
        :param filepath: Path to the zip file
        :param use_tmp_directory: If True, uses a temporary directory for extraction; otherwise, extracts in the current working directory.
        :return: Path or None: The first valid file found, or None if no valid files exist.
        """
        filepath = Path(filepath).resolve()
        time_id = datetime.now().strftime("%H%M%S%f")
        if use_tmp_directory:
            output_dir = DataHandlers.locate_temp_dir().joinpath("modelizer_unzip").joinpath(f"unzipped_{filepath.name}_{time_id}")
        else:
            output_dir = Path.cwd().joinpath("modelizer_unzip").joinpath(f"unzipped_{filepath.stem}_{time_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        elif not filepath.is_file():
            raise FileExistsError(f"File not found: {filepath}")
        elif filepath.suffix != ".zip":
            raise ValueError(f"File is not a zip file: {filepath}")
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(output_dir)
            names = zf.namelist()
            files = [name for name in names if not name.endswith('/') and Path(name).name.lower() != '.ds_store']
            if len(files) == 1:
                single_file = output_dir / files[0]
                if single_file.exists() and single_file.is_file():
                    return single_file.resolve()
        extracted_dir = output_dir / filepath.stem
        if extracted_dir.exists():
            if extracted_dir.is_file():
                return extracted_dir
            elif extracted_dir.is_dir():
                result = DataHandlers.recursive_directory_lookup(extracted_dir)
                if result:
                    return result
        for file in output_dir.rglob("*"):
            if file.is_file() and file.name.lower() != '.ds_store':
                return file
        return None

    @staticmethod
    def zip(filepath: str | Path, zip_filepath: Optional[str | Path] = None) -> Path:
        """
        Zips a file or directory.
        :param filepath: Path to the file or directory to be zipped
        :param zip_filepath: Optional path for the output zip file. If None, the zip file will be created in the same directory as the input file/directory with a .zip extension.
        :return: Path to the created zip file
        """
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"File or directory not found: {filepath}")
        if not (filepath.is_file() or filepath.is_dir()):
            raise ValueError(f"Path is neither a file nor a directory: {filepath}")

        output_zip = Path(zip_filepath).resolve() if zip_filepath is not None else filepath.with_suffix('.zip')
        if output_zip.suffix != '.zip':
            raise ValueError(f"Output zip file must have a .zip extension: {output_zip}")
        output_zip.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            if filepath.is_file():
                zf.write(filepath, arcname=filepath.name)
            else:
                for root, dirs, files in os.walk(filepath):
                    root_path = Path(root)
                    rel_root = root_path.relative_to(filepath.parent)
                    zf.write(root_path, arcname=str(rel_root) + '/')
                    for file in files:
                        filepath = root_path / file
                        try:
                            if filepath.resolve() == output_zip.resolve():
                                continue
                        except Exception:
                            pass
                        arcname = filepath.relative_to(filepath.parent)
                        zf.write(filepath, arcname=str(arcname))
        return output_zip


########################################################################################################################
#                                   Helpler class for computing hashes                                                 #
########################################################################################################################
class HashingHelpers(metaclass=SingletonMeta):
    """Helper class providing methods for fast non-cryptographic and cryptographic hashing."""
    @staticmethod
    def _format(format_type: str, out: bytes) -> str | bytes | int:
        match format_type:
            case "bytes":
                return out
            case "int":
                return int.from_bytes(out, "big", signed=False)
            case "hex":
                return out.hex()
            case _:
                return base64.urlsafe_b64encode(out).rstrip(b"=").decode("ascii")

    @staticmethod
    def hash(
            data: str | bytes | bytearray | memoryview,
            *,
            nbytes: int = 16,
            seed: int = 0,
            encoding: str = "utf-8",
            fmt: Literal["b64", "hex", "bytes", "int"] = "b64",
    ) -> str | bytes | int:
        """
        Fast non-cryptographic hash for duplicate detection.
        Defaults to 128-bit for very low collision risk while remaining extremely fast.
        :param data: input data to hash
        :param nbytes: number of bytes in the output hash (1 to 16)
        :param seed: seed value for the hash function
        :param encoding: encoding to use if data is a string
        :param fmt: output format: "b64" (default), "hex", "bytes", or "int"
        :return: hash value in the specified format
        """
        assert 1 <= nbytes <= 16, "nbytes must be in [1, 16]"

        if isinstance(data, str):
            buf = data.encode(encoding)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            buf = memoryview(data)
        else:
            raise TypeError("data must be str, bytes, bytearray, or memoryview")

        if nbytes <= 8:
            digest = xxh3_64(buf, seed=seed).digest()  # 8 bytes
            out = digest[:nbytes]  # allow shorter truncation
        else:
            digest = xxh3_128(buf, seed=seed).digest()  # 16 bytes
            out = digest[:nbytes]

        return HashingHelpers._format(fmt, out)

    @staticmethod
    def cryptographic_hash(
            data: str | bytes | bytearray | memoryview,
            *,
            nbytes: int = 32,
            key: bytes | bytearray | memoryview | None = None,
            salt: bytes | bytearray | memoryview | None = None,
            encoding: str = "utf-8",
            fmt: Literal["b64", "hex", "bytes", "int"] = "b64",
            algo: Literal["auto", "blake2b", "sha256"] = "auto",
    ) -> str | bytes | int:
        """
        Cryptographic hash without external deps.
        Order: blake2b (default) -> sha256.
        - blake2b: 1..64 bytes
        - sha256/HMAC-SHA256: 1..32 bytes
        :param data: input data to hash
        :param nbytes: number of bytes in the output hash
        :param key: optional key for HMAC or keyed blake2b
        :param salt: optional salt to prepend to the data
        :param encoding: encoding to use if data is a string
        :param fmt: output format: "b64" (default), "hex", "bytes", or "int"
        :param algo: hashing algorithm: "auto" (default), "blake2b", or "sha256"
        :return: hash value in the specified format
        """
        if isinstance(data, str):
            buf = data.encode(encoding)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            buf = memoryview(data)
        else:
            raise TypeError("data must be str, bytes, bytearray, or memoryview")

        key_bytes = None if key is None else (bytes(key) if isinstance(key, (bytes, bytearray, memoryview)) else None)
        if key is not None and key_bytes is None:
            raise TypeError("key must be bytes-like")
        salt_mv = None if salt is None else (
            memoryview(salt) if isinstance(salt, (bytes, bytearray, memoryview)) else None)
        if salt is not None and salt_mv is None:
            raise TypeError("salt must be bytes-like")

        if algo in ("blake2b", "auto"):
            if not (1 <= nbytes <= 64):
                raise ValueError("nbytes must be in [1, 64] for blake2b")
            if key_bytes is not None and len(key_bytes) > 64:
                raise ValueError("blake2b key must be at most 64 bytes")
            h = hashlib.blake2b(digest_size=nbytes, key=key_bytes) if key_bytes else hashlib.blake2b(digest_size=nbytes)
            if salt_mv is not None:
                h.update(salt_mv)
                h.update(b"\x00")
            h.update(buf)
            return HashingHelpers._format(fmt, h.digest())

        if not (1 <= nbytes <= 32):
            raise ValueError("nbytes must be in [1, 32] for sha256")
        if key_bytes:
            h = hmac.new(key_bytes, digestmod=hashlib.sha256)
            if salt_mv is not None:
                h.update(salt_mv)
                h.update(b"\x00")
            h.update(buf)
            digest = h.digest()
        else:
            h = hashlib.sha256()
            if salt_mv is not None:
                h.update(salt_mv)
                h.update(b"\x00")
            h.update(buf)
            digest = h.digest()
        return HashingHelpers._format(fmt, digest[:nbytes])


########################################################################################################################
#                                           Directory Helpers                                                          #
########################################################################################################################
class DirectoryHelpers(metaclass=SingletonMeta):
    @staticmethod
    def find_folder(root: Union[str, Path], folder_name: str) -> Optional[Path]:
        """
        Return the first directory Path under root (including root itself) whose name
        equals folder_name. Uses pathlib.Path.rglob for recursion.
        Args:
        root: root directory to start searching from (str or Path).
        folder_name: directory name to search for (exact match, case-sensitive).

        Returns:
            Path to the first matching directory, or None if not found.

        Raises:
            FileNotFoundError: if `root` does not exist.
        """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"root {root!s} does not exist")

        if root.is_dir() and root.name == folder_name:
            return root

        for p in root.rglob(folder_name):
            if p.is_dir():
                return p

        return None

    @staticmethod
    def find_folders(root: Union[str, Path], folder_name: str, case_sensitive: bool = True,
                     follow_symlinks: bool = False, ) -> list[Path]:
        """ Return all directory Paths under root (including root itself) whose name equals folder_name.
        This implementation uses os.walk for explicit control over symlink-following and case-sensitivity,
        but returns pathlib.Path objects.
        Args:
                root: root directory to start searching from (str or Path).
                folder_name: directory name to search for.
                case_sensitive: whether match should be case-sensitive (default True).
                follow_symlinks: whether to follow symlinked directories (default False).

            Returns:
                List of Path objects for each matching directory (empty list if none).

            Raises:
                FileNotFoundError: if `root` does not exist.
            """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"root {root!s} does not exist")
        target = folder_name if case_sensitive else folder_name.lower()
        matches: list[Path] = []

        for dir_path, dir_names, _filenames in os.walk(root, followlinks=follow_symlinks):
            for d in dir_names:
                name = d if case_sensitive else d.lower()
                if name == target:
                    matches.append(Path(dir_path) / d)

        if root.is_dir():
            root_name = root.name if case_sensitive else root.name.lower()
            if root_name == target and (root not in matches):
                matches.insert(0, root)

        return matches

    @staticmethod
    def find_model_dir(
            root: Union[str, Path],
            *,
            return_all: bool = False,
            follow_symlinks: bool = False,
            max_depth: Optional[int] = None,
            filenames: list[str] = ("config.pkl", "model.pth"),
    ) -> Union[Optional[Path], list[Path]]:
        """
        Recursively search `root` and its subdirectories (using pathlib only) for a directory
        that contains all filenames listed in `filenames`. By default, returns the first match
        (top-down). If `return_all` is True, returns a list of all matches.

        Args:
            root: starting directory (str or Path).
            return_all: if True return all matching directories, otherwise return the first match or None.
            follow_symlinks: whether to follow symlinked directories while traversing.
            max_depth: maximum recursion depth (0 = only check root). None means unlimited.
            filenames: list of filenames that must be present in the same directory.

        Returns:
            If return_all is False: pathlib.Path of first match or None.
            If return_all is True: list of pathlib.Path objects (empty if none).

        Raises:
            FileNotFoundError: if root does not exist or is not a directory.
        """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"root {root!s} does not exist")
        if not root.is_dir():
            raise FileNotFoundError(f"root {root!s} is not a directory")

        required = tuple(filenames)
        matches: list[Path] = []
        queue = deque([(root, 0)])
        visited = set()

        while queue:
            cur_dir, depth = queue.popleft()

            if cur_dir.is_symlink() and not follow_symlinks:
                continue

            if follow_symlinks:
                try:
                    resolved = cur_dir.resolve()
                except Exception:
                    resolved = cur_dir
                if str(resolved) in visited:
                    continue
                visited.add(str(resolved))

            try:
                has_all = True
                for fn in required:
                    try:
                        if not (cur_dir / fn).is_file():
                            has_all = False
                            break
                    except PermissionError:
                        has_all = False
                        break
                if has_all:
                    if return_all:
                        matches.append(cur_dir)
                    else:
                        return cur_dir
            except PermissionError:
                pass

            if (max_depth is not None) and (depth >= max_depth):
                continue

            try:
                for child in cur_dir.iterdir():
                    try:
                        if child.is_dir():
                            if child.is_symlink() and not follow_symlinks:
                                continue
                            queue.append((child, depth + 1))
                    except PermissionError:
                        continue
            except PermissionError:
                continue

        return matches if return_all else None
