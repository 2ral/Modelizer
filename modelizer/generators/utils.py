import modelizer.utils as utils
from modelizer.generators.grammars import MARKER_MAPPING

from enum import Enum
from typing import Type
from pathlib import Path
from itertools import chain, repeat
from re import compile as re_compile
from random import randint, shuffle


class PlaceholderDataType(Enum):
    # Markdown
    TEXT = "TEXT"
    URL = "URL"
    # SQL
    VALUE = "VALUE"
    COLUMN = "COLUMN"
    TABLE = "TABLE"
    # MathML
    NUMBER = "NUMBER"
    IDENTIFIER = "IDENTIFIER"
    # Python Math Expression
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    # LINQ - Not Supported
    SOURCE = "SOURCE"
    FUNCTION = "FUNCTION"
    EXPRESSION = "EXPRESSION"


class PlaceholderProcessor:
    def __init__(self, placeholders: list | tuple | str):
        placeholders = MARKER_MAPPING[placeholders] if isinstance(placeholders, str) else placeholders
        self.placeholders = [p[0] if isinstance(p, list | tuple) else p for p in placeholders]
        self.__num_counter__ = randint(1000, 10000)
        self.__str_counter__ = 97
        self.__last_str__ = "".join([chr(randint(97, 122)) for _ in range(randint(5, 10))])
        self.__filters__ = {p: re_compile(rf"{p}_\d+") for p in self.placeholders}

    def deduplicate_placeholders(self, data: str) -> str:
        for p in self.placeholders:
            counter, pos = 0, data.find(p, 0)
            while pos != -1:
                counter += 1
                replacement = f"{p}_{counter}"
                data = f"{data[:pos]}{replacement}{data[pos + len(p):]}"
                pos = data.find(p, pos + len(replacement))
        return data

    def generalize_placeholders(self, data: str | list) -> str:
        if isinstance(data, list):
            for i, v in enumerate(data):
                for p in self.__filters__:
                    to_clean = self.__filters__[p].findall(v)
                    for p_i in to_clean:
                        data[i] = data[i].replace(p_i, p)
        else:
            for p in self.__filters__:
                for p_i in self.__filters__[p].findall(data):
                    data = data.replace(p_i, p)
        return data

    def replace_placeholders(self, data: str) -> tuple[str, dict]:
        mapping = {p: {} for p in self.placeholders}
        for p in self.placeholders:
            while p in data:
                start_pos = data.find(p)
                end_pos = start_pos + len(p)
                if data[end_pos] == "_":
                    end_pos += 1
                while data[end_pos].isdigit():
                    end_pos += 1
                p_i = data[start_pos:end_pos]
                i = self.generate_value(PlaceholderDataType(p))
                mapping[p][p_i] = i
                data = data.replace(p_i, i)
        return data, mapping

    def recover_placeholders(self, data: str, mapping: dict) -> str:
        for p in self.placeholders:
            for p_i, i in mapping[p].items():
                data = data.replace(i, p_i)
        return data

    def generate_number(self):
        self.__num_counter__ += 1
        return str(self.__num_counter__)

    def generate_string(self):
        if self.__str_counter__ == 97:
            self.__last_str__ += chr(self.__str_counter__)
            self.__str_counter__ += 1
        elif self.__str_counter__ == 122:
            self.__last_str__ = self.__last_str__[:-1] + chr(self.__str_counter__)
            self.__str_counter__ = 97
        else:
            self.__last_str__ = self.__last_str__[:-1] + chr(self.__str_counter__)
            self.__str_counter__ += 1
        return self.__last_str__

    def generate_value(self, data_type: PlaceholderDataType):
        # Reimplement this method to support custom data types
        match data_type:
            case PlaceholderDataType.TEXT | PlaceholderDataType.VALUE \
                 | PlaceholderDataType.COLUMN | PlaceholderDataType.TABLE \
                 | PlaceholderDataType.IDENTIFIER:
                return self.generate_string()
            case PlaceholderDataType.NUMBER | PlaceholderDataType.INTEGER:
                return self.generate_number()
            case PlaceholderDataType.FLOAT:
                return f"{self.generate_number()}.{self.generate_number()}"
            case PlaceholderDataType.URL:
                return f"https://{self.generate_string()}.{self.generate_string()}"
            case _:
                raise ValueError(f"Unsupported placeholder data type {data_type}\n"
                                 f"Reimplement this method to support custom data types.")

    def split_tokens_with_placeholders(self, tokens: list[str]) -> list[str]:
        def iterate_list(start_pos: int, data: list) -> tuple[int, list[str]]:
            for i, t in enumerate(data):
                if i < start_pos:
                    continue
                for p in self.placeholders:
                    if p in t:
                        token_repetition_count = t.count(p)
                        new_tokens = list()
                        t_left, t_right = t.split(p, 1)
                        if t_left:
                            new_tokens.append(t_left)
                        if t_right:
                            p_prime = p
                            if t_right[0] == "_":
                                p_prime += "_"
                                t_right = t_right[1:]
                                # eliminate the digit prefix
                                while t_right and t_right[0].isdigit():
                                    p_prime += t_right[0]
                                    t_right = t_right[1:]
                                new_tokens.append(p_prime)
                                if t_right:
                                    # remainder is a token
                                    new_tokens.append(t_right)
                            else:
                                # all right side is a token
                                new_tokens.append(p_prime)
                                new_tokens.append(t_right)
                        new_tokens_count = len(new_tokens)
                        if new_tokens_count > 1:
                            new_index = i + new_tokens_count if token_repetition_count == 1 else i
                            return new_index, data[:i] + new_tokens + data[i+1:]
                        else:
                            return i + 1, data
            return -1, data

        status, tokens = iterate_list(0, tokens)
        while status != -1:
            status, tokens = iterate_list(status, tokens)
        return tokens


def __placeholder_updator__(params: tuple[list[str], list[str], Type[PlaceholderProcessor]]):
    subjects, markers, factory = params
    processor = factory(markers)
    return [processor.deduplicate_placeholders(subject) for subject in subjects]


def placeholder_updator(subjects: list[str], markers: list[str], factory: Type[PlaceholderProcessor]) -> list[str]:
    subjects = list(utils.Multiprocessing.chunk_generator(subjects))
    markers = list(repeat(markers, len(subjects)))
    factories = list(repeat(factory, len(subjects)))
    params = list(zip(subjects, markers, factories))
    return list(chain(*utils.Multiprocessing.parallel_run(__placeholder_updator__, params, text="Updating placeholders")))


def generate_random_marker_mapping(subject: str, placeholders: list | tuple | None,  filepath: Path) -> list[list[str]]:
    loaded = utils.pickle_load(filepath)
    placeholders = MARKER_MAPPING.setdefault(subject, None) if placeholders is None else placeholders
    assert placeholders is not None, "Placeholders are not defined and not found. Pass a valid list of placeholders."
    if isinstance(loaded, dict) and subject in loaded:
        result = loaded[subject]
    else:
        generated = {m[0]: [f"{m[0]}_{i}" for i in range(250)] for m in placeholders}
        combined = list(chain(*generated.values()))
        result = []
        for i in range(100):
            shuffle(combined)
            result.append(combined.copy())
        data = {} if loaded is None else loaded
        data[subject] = result
        utils.pickle_dump(data, filepath)
    return result


def finalize_latex(latex: str):
    # helper function to finalize latex formulas for display
    return r"{}{}{}".format('\\begin{align*}\n', latex, '\n\\end{align*}')
