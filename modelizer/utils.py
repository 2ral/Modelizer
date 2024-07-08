import logging

from enum import Enum
from pathlib import Path
from math import ceil as math_ceil
from multiprocessing import Pool, cpu_count
from dateutil.parser import parse as date_parser
from pickle import dump, load, PicklingError, UnpicklingError, HIGHEST_PROTOCOL

from tqdm import tqdm


class SingletonMeta(type):
    __instances__ = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances__:
            instance = super().__call__(*args, **kwargs)
            cls.__instances__[cls] = instance
        return cls.__instances__[cls]


class LoggingLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class AbstractLogger:
    def __init__(self, logger: logging.Logger):
        self.__logger__ = logger

    def debug(self, message: str):
        self.__logger__.debug(message)

    def info(self, message: str):
        self.__logger__.info(message)

    def warning(self, message: str):
        self.__logger__.warning(message)

    def error(self, message: str):
        self.__logger__.error(message)

    def critical(self, message: str):
        self.__logger__.critical(message)


class Logger(AbstractLogger, metaclass=SingletonMeta):
    __instance__ = None

    def __init__(self, level: LoggingLevel, root_dir: str | Path | None = None,
                 filename: str = "debug.log", rewrite: bool = False, file_logger: bool = True, console_logger: bool = True):
        if Logger.__instance__ is None:
            handlers = []
            if console_logger:
                handlers.append(logging.StreamHandler())
            if file_logger:
                if root_dir is not None:
                    root_dir = Path(root_dir).resolve() if isinstance(root_dir, str) else root_dir
                    root_dir.mkdir(parents=True, exist_ok=True)
                    filepath = root_dir.joinpath(filename)
                else:
                    filepath = Path(filename).resolve()
                if rewrite and filepath.is_file():
                    filepath.unlink()
                handlers.append(logging.FileHandler(filepath.as_posix()))
            logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', handlers=handlers,
                                datefmt='%d-%m-%Y %H:%M:%S', level=level.value, force=True,)
            logger = logging.getLogger()
            super(Logger, self).__init__(logger)
            Logger.__instance__ = self
        else:
            if Logger.__instance__.__logger__.level > level.value:
                Logger.__instance__.__logger__.setLevel(level.value)
            super(Logger, self).__init__(Logger.__instance__.__logger__)


class FileLogger(AbstractLogger):
    __instances__ = {}

    def __init__(self, logger_name: str, level: LoggingLevel, root_dir: str | Path | None = None,
                 filename: str | None = None, rewrite: bool = False):
        assert len(logger_name) > 0, "Logger name cannot be empty"
        if logger_name not in FileLogger.__instances__:
            if root_dir is not None:
                root_dir = Path(root_dir).resolve() if isinstance(root_dir, str) else root_dir
                root_dir.mkdir(parents=True, exist_ok=True)
                filepath = root_dir.joinpath(f"{logger_name}.log" if filename is None else filename)
            else:
                filepath = Path(f"{logger_name}.log" if filename is None else filename)
            if rewrite and filepath.is_file():
                filepath.unlink()
            logger = logging.getLogger(logger_name)
            logger.setLevel(level.value)
            formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
            fh = logging.FileHandler(filepath.as_posix())
            fh.setLevel(level.value)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            super(FileLogger, self).__init__(logger)
            FileLogger.__instances__[logger_name] = self
        else:
            FileLogger.__instances__[logger_name].__logger__.setLevel(level.value)
            super(FileLogger, self).__init__(FileLogger.__instances__[logger_name].__logger__)


# Parallel processing utility function
class Multiprocessing:
    CORE_COUNT = cpu_count()

    @staticmethod
    def parallel_run(function, data, text=None, n_jobs: int = 0, chunkify: bool = False) -> list:
        n_jobs = n_jobs if n_jobs else Multiprocessing.CORE_COUNT
        chunkify = chunkify if n_jobs > 1 else False
        data = list(Multiprocessing.chunk_generator(data, n_jobs)) if chunkify else data
        with Pool(n_jobs) as pool:
            if text is None:
                result = list(pool.imap(function, data))
            else:
                result = list(tqdm(pool.imap(function, data), total=len(data), desc=text))
            pool.close()
            pool.join()
        return result

    @staticmethod
    def chunk_generator(d_list: list, elems_per_chunk: int = CORE_COUNT):
        for i in range(0, len(d_list), elems_per_chunk):
            yield d_list[i:i + elems_per_chunk]


def chunkify_list(data: list, n_chunks: int) -> list[list]:
    chunk_size = math_ceil(len(data) / n_chunks)
    return list(map(lambda i: data[i * chunk_size:i * chunk_size + chunk_size], list(range(n_chunks))))


# Pickling utility functions
def pickle_dump(data, filepath: str | Path) -> int:
    filepath = filepath.as_posix() if isinstance(filepath, Path) else filepath
    assert data is not None, "Data cannot be None"
    try:
        dump(data, open(filepath, "wb"), protocol=HIGHEST_PROTOCOL)
    except (PicklingError, OSError):
        return 1
    return 0


def pickle_load(filepath: str | Path):
    filepath = filepath.as_posix() if isinstance(filepath, Path) else filepath
    try:
        data = load(open(filepath, "rb"))
    except (UnpicklingError, OSError, FileNotFoundError):
        data = None
    return data


# subject mapping utility functions
def infer_subject(source: str, target: str) -> str | None:
    if source in ["html", "markdown"] and target in ["html", "markdown"]:
        subject = "markdown"
    elif source in ["mathml", "latex", "ascii_math"] and target in ["mathml", "latex", "ascii_math"]:
        subject = "mathml"
    elif source in ["sql", "kql"] and target in ["sql", "kql"]:
        subject = "sql"
    elif source in ["expression", "latex"] and target in ["expression", "latex"]:
        subject = "expression"
    else:
        subject = f"{source}_{target}"
    return subject


def get_time_from_log(debug_file: str | Path, search_pattern: str = "Model trained in ", time_format: str = "h") -> float:
    if isinstance(debug_file, str):
        debug_file = Path(debug_file)
    debug_lines = debug_file.read_text()
    pos = debug_lines.find(search_pattern) + len(search_pattern)
    time_str = debug_lines[pos:]
    if ", " in time_str:
        day_str, time_str = time_str.split(", ", 1)
        hours = int(day_str.split(" day")[0]) * 24
    else:
        hours = 0
    time = date_parser(time_str)
    if time_format == "h":
        result = hours + round(time.hour + time.minute / 60 + time.second / 3600, 2)
    elif time_format == "m":
        result = hours * 60 + round(time.hour * 60 + time.minute + time.second / 60, 2)
    else:
        result = hours * 3600 + round(time.hour * 3600 + time.minute * 60 + time.second, 2)
    return result


def parse_model_name(filename: str | Path) -> tuple[str, str, str, bool, bool]:
    if isinstance(filename, Path):
        filename = filename.name
    simplified_tokens = "simplified" in filename
    name = (filename.replace("simplified", "").
            replace("enumerated", "").
            replace("_____", "").lower())
    dataset_name, name = name.split("___", 1)
    dataset_size = dataset_name.replace("data_", "")
    if "nop" in dataset_size:
        partitioned = False
        dataset_size = dataset_size.replace("nop_", "")
    else:
        partitioned = True
    name = name.replace("modelizer_", "")
    name = name.replace("ascii_math", "ascii-math")
    if name.count("_") > 1:
        name = name.replace("_", "|", 1)
        source, target = name.split("_", 1)
        source = source.replace("|", "_")
    else:
        source, target = name.split("_")
    if "ascii-math" in source:
        source = source.replace("ascii-math", "ascii_math")
    elif "ascii-math" in target:
        target = target.replace("ascii-math", "ascii_math")
    return dataset_size, source, target, partitioned, simplified_tokens
