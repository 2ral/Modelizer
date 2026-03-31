import re
import random
import string

from sys import maxsize
from typing import Optional, Union

from modelizer.dependencies.fuzzingbook import Grammar, EarleyParser


class PlaceholderProcessor:
    def __init__(self,
                 placeholder_mapping: dict[str, str],
                 grammar: Optional[Union[dict, Grammar]] = None,
                 *,
                 min_int: int = -maxsize + 1,
                 max_int: int = maxsize,
                 min_string_length: int = 1,
                 max_string_length: int = 128,
                 allowed_chars: str = string.ascii_letters + string.digits,
                 custom_generators: Optional[dict] = None):
        """
        Initializes the PlaceholderProcessor with the given parameters.
        :param placeholder_mapping: dictionary mapping non-terminal symbols to placeholders. "<rule_name>": "placeholder"
        :param grammar: A Grammar object or a dictionary defining the grammar rules. <rule_name>: ["<rule_content>"]
        :param min_int: minimum integer value for generated integers. Default is -sys.maxsize - 1.
        :param max_int: maximum integer value for generated integers. Default is sys.maxsize.
        :param min_string_length: minimum length for generated strings. Default is 1.
        :param max_string_length: maximum length for generated strings. Default is 128.
        :param allowed_chars: string of characters allowed in generated strings. Default is alphanumeric characters.
        :param custom_generators: Optional dictionary of custom generators for specific placeholders.
        """
        assert isinstance(placeholder_mapping, dict), "Placeholders must be a dictionary mapping non-terminal symbols to placeholders."
        self._placeholders = [str(p) for p in set(placeholder_mapping.values())]
        self._filters = {p: re.compile(rf"{p}_\d+") for p in self._placeholders}
        self._patterns = {p: re.compile(rf'\b{re.escape(p)}\b') for p in self._placeholders}
        self._replacement_counter = 0
        self._max_int = max_int
        self._min_int = min_int
        self._min_string_length = min_string_length
        self._max_string_length = max_string_length
        self._allowed_chars = allowed_chars
        self._value_generators = {}
        self._register_default_generators()

        self._protocols = ("http", "https", "ftp", "sftp", "ftps")

        if custom_generators:
            for key, gen_func in custom_generators.items():
                self.register_value_generator(key, gen_func)

        if grammar is not None:
            if ((hasattr(grammar, '__getitem__') and hasattr(grammar, 'keys')) or
                    (hasattr(grammar, '__class__') and grammar.__class__.__name__ == 'Grammar')):
                self._parser = EarleyParser(grammar, placeholder_mapping)
            else:
                raise AssertionError("grammar must be a Grammar object or Python dictionary.")
        else:
            self._parser = None

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def max_int(self):
        return self._max_int

    @max_int.setter
    def max_int(self, value):
        if not isinstance(value, int):
            raise ValueError(f"max_int must be an integer, got {type(value)} instead.")
        self._max_int = value

    @property
    def min_int(self):
        return self._min_int

    @min_int.setter
    def min_int(self, value):
        if not isinstance(value, int):
            raise ValueError(f"min_int must be an integer, got {type(value)} instead.")
        if value <= -maxsize - 1:
            raise ValueError(f"min_int must be greater than {-maxsize - 1}, got {value} instead.")
        if value > self._max_int:
            raise ValueError(f"min_int must be less than or equal to max_int, got {value} > {self._max_int} instead.")
        self._min_int = value

    @property
    def min_string_length(self):
        return self._min_string_length

    @min_string_length.setter
    def min_string_length(self, new_value):
        if not isinstance(new_value, int):
            raise ValueError(f"min_string_length must be an integer, got {type(new_value)} instead.")
        self._min_string_length = new_value

    @property
    def max_string_length(self):
        return self._max_string_length

    @max_string_length.setter
    def max_string_length(self, value):
        if not isinstance(value, int):
            raise ValueError(f"max_string_length must be an integer, got {type(value)} instead.")
        if value < self._min_string_length:
            raise ValueError(
                f"max_string_length must be greater than or equal to min_string_length, got {value} < {self._min_string_length} instead."
            )
        self._max_string_length = value

    @property
    def allowed_chars(self):
        return self._allowed_chars

    @allowed_chars.setter
    def allowed_chars(self, value):
        if not isinstance(value, str):
            raise ValueError(f"allowed_chars must be a string, got {type(value)} instead.")
        self._allowed_chars = value

    def deduplicate_placeholders(self, data: str) -> str:
        """
        Add unique identifiers to each placeholder occurrence in the given data string.
        :param data: input data string with placeholders
        :return: updated data string with unique identifiers for each placeholder occurrence
        """
        for p in self._placeholders:
            self._replacement_counter = 0
            data = self._patterns[p].sub(self.__placeholder_replacement__, data)
        return data

    def __placeholder_replacement__(self, match):
        self._replacement_counter += 1
        return f"{match.group(0)}_{self._replacement_counter}"

    def generalize_placeholders(self, data: str) -> str | list:
        """
        Remove unique identifiers from each placeholder occurrence in the given data string or list.
        :param data: String containing placeholders with identifiers
        :return: String with abstracted placeholders
        """
        for p, pattern in self._filters.items():
            data = pattern.sub(p, data)
        return data

    def insert_placeholders(self, data: str, deduplicate: bool = True):
        """
        This method implements the logic to replace data fragments with corresponding placeholders.
        :param data: input data string
        :param deduplicate: Boolean flag indicating whether to deduplicate placeholders
        :return: abstracted data string with placeholders
        """
        if self._parser is None:
            return data
        abstracted = self._parser.abstract(data)
        if deduplicate:
            abstracted = self.deduplicate_placeholders(abstracted)
        return abstracted

    def insert_mapped_placeholders(self, data: str) -> tuple[str, dict[str, str]]:
        """
        This method implements the logic to replace data fragments with corresponding placeholders.
        :param data: input data string
        :return: tuple containing the modified data string and a dictionary mapping placeholders to the replaced values
        """
        if self._parser is None:
            return data, {}
        abstracted, mapping = self._parser.abstract_mapped(data)
        return abstracted, mapping

    @staticmethod
    def remove_placeholders(data: str, mapping: dict) -> str:
        """
        Replace all occurrences of placeholders in the given data string with the corresponding values from the mapping.
        :param data: String with placeholders
        :param mapping: Dictionary with placeholder->value mapping
        :return: String without placeholders
        """
        if not mapping:
            return data
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, mapping.keys())) + r')\b')
        return pattern.sub(lambda match: mapping[match.group(0)], data)

    def split_tokens_with_placeholders(self, tokens: list[str]) -> list[str]:
        """
        Split tokens that contain placeholders into sub tokens.
        :param tokens: tokenized data as list of strings
        :return: sequence of tokens where each placeholder occurrence is guaranteed to be a separate token
        """
        placeholder_patterns = [rf"{re.escape(p)}(?:_\d+)?" for p in self._placeholders]
        combined_regex = "(" + "|".join(placeholder_patterns) + ")"

        new_tokens = []
        for token in tokens:
            parts = re.split(combined_regex, token)
            new_tokens.extend([part for part in parts if part])

        return new_tokens

    def synthesize_values(self, data: str) -> tuple[str, dict]:
        """
        Replace all occurrences of each placeholder in the given data string.
        If a placeholder occurrence (possibly with a trailing underscore and digits)
        is seen more than once, the same generated value is used.
        :param data: input data string
        :return:
            - updated data string with placeholders replaced by generated values
            - dictionary mapping placeholders to generated values
        """
        mapping = {p: {} for p in self._placeholders}
        # Build combined regex for all placeholders followed by optional _digits.
        pattern = re.compile("(" + "|".join(map(re.escape, self._placeholders)) + r")(_\d+)?")

        def callback(match):
            placeholder = match.group(1)
            occurrence = match.group(0)
            if occurrence in mapping[placeholder]:
                return mapping[placeholder][occurrence]
            else:
                generated_val = self.generate_value(placeholder)
                mapping[placeholder][occurrence] = generated_val
                return generated_val

        data = pattern.sub(callback, data)
        return data, mapping

    def generate_integer(self) -> str:
        return str(random.randint(self._min_int, self._max_int))

    def generate_float(self) -> str:
        int_part = random.randint(self._min_int if self._min_int >= 0 else 0, min(self._max_int, 999999))
        decimal_part = random.randint(0, 999999)
        return f"{int_part}.{decimal_part}"

    def generate_number(self) -> str:
        return random.choice([self.generate_integer, self.generate_float])()

    def generate_string(self) -> str:
        return ''.join(random.choice(self._allowed_chars) for _ in range(random.randint(self._min_string_length, self._max_string_length)))

    def generate_url(self) -> str:
        return f"{random.choice(self._protocols)}://{self.generate_string()}.{self.generate_string()}"

    def _register_default_generators(self):
        for key in ("_TXT_", "_VAL_", "_COL_", "_TABLE_", "_ID_"):
            self.register_value_generator(key, self.generate_string)
        self.register_value_generator("_FLOAT_", self.generate_float)
        self.register_value_generator("_NUM_", self.generate_number)
        self.register_value_generator("_INT_", self.generate_integer)
        self.register_value_generator("_URL_", self.generate_url)

    def register_value_generator(self, placeholder: str, func):
        """
        Registers (or overrides) a generator function for a specific data type.
        placeholder: must a unique string that represents the Placeholder.
        func: A callable (with no parameters) that returns a generated value.
        """
        if not callable(func):
            raise ValueError("Generator must be callable.")
        if not isinstance(placeholder, str):
            raise ValueError("Placeholder must be a string.")
        if placeholder in self._value_generators:
            raise ValueError(f"Generator for placeholder {placeholder!r} is already registered.")
        self._value_generators[placeholder] = func

    def generate_value(self, data_type) -> str:
        """
        Generate a value for the given data type.
        :param data_type: Placeholder string representing the data type.
        :return: Generated value.
        """
        generator = self._value_generators.get(data_type)
        if generator is None:
            raise ValueError(f"Unsupported placeholder data type: {data_type!r}. "
                             "Register a custom generator via register_value_generator().")
        return generator()
