from enum import Enum
from abc import ABC, abstractmethod
from re import compile as re_compile

from collections import ChainMap, OrderedDict


def mine_placeholders(grammar: dict) -> tuple:
    placeholders = list()
    for k, v in grammar.items():
        if len(v) > 1:
            continue
        elif "<" in v[0] and ">" in v[0] and len(v[0]) > 2:
            continue
        else:
            placeholders.append(v[0])
    return tuple(placeholders)


class MappingPolicy(Enum):
    SIMPLIFIED = 0
    OPTIMIZING = 1
    EXHAUSTIVE = 2


class AbstractTokenizer(ABC):
    def __init__(self, grammar: dict | None = None, placeholders: list[str] | list[str] | tuple[str] = ()):
        # Do not update self.buffer or self.__mappings__ outside the feed method
        def feed_wrapper(func):
            def wrapper(*args, **kwargs):
                self.buffer.clear()
                func(*args, **kwargs)

                if self.__mapping_policy__ == MappingPolicy.SIMPLIFIED:
                    for i in range(len(self.buffer)):
                        for p in self.__filter_rules__:
                            for candidate in self.__filter_rules__[p].findall(self.buffer[i]):
                                self.buffer[i] = self.buffer[i].replace(candidate, p)

                return self.buffer.copy()

            return wrapper

        self.buffer = []
        if grammar is not None:
            self.__placeholders__ = mine_placeholders(grammar)
        else:
            self.__placeholders__ = placeholders
        self.__mappings__ = {p: dict() for p in placeholders}
        self.__filter_rules__ = {p: re_compile(rf"{p}_\d+") for p in placeholders}
        self.__inverse_mapping__ = dict()
        self.__mask_mapping__ = False
        self.__mapping_policy__ = MappingPolicy.OPTIMIZING
        self.feed = feed_wrapper(self.feed)
        self.tokenize = self.feed
        self.mapped_tokenize = self.mapped_feed

    def __mask_token__(self, token: str, placeholder: str) -> str:
        match self.__mapping_policy__:
            case MappingPolicy.SIMPLIFIED:
                tag = placeholder
                self.__mappings__[placeholder][f"{placeholder}_{len(self.__mappings__[placeholder]) + 1}"] = token
            case MappingPolicy.OPTIMIZING:
                if token in self.__inverse_mapping__:
                    tag = self.__inverse_mapping__[token]
                else:
                    tag = f"{placeholder}_{len(self.__mappings__[placeholder]) + 1}"
                    self.__mappings__[placeholder][tag] = token
                    self.__inverse_mapping__[token] = tag
            case _:
                tag = f"{placeholder}_{len(self.__mappings__[placeholder]) + 1}"
                self.__mappings__[placeholder][tag] = token
                self.__inverse_mapping__[token] = tag
        return tag

    @staticmethod
    def find_merge_candidate(tokens: list[str], search_keys: tuple, start_pos: int = -1) -> int:
        found = False
        for i, token in enumerate(tokens):
            if i <= start_pos:
                continue
            for key in search_keys:
                found = key in token
            if found:
                return i
        return -1

    @abstractmethod
    def feed(self, data):
        # Use only this method to update the content of the self.buffer and self.__mappings__ variables
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, tokens: list[str]) -> str:
        # re-declare this method in the child class if more complex logic is needed
        raise NotImplementedError

    def set_mapping_policy(self, policy: int | MappingPolicy):
        self.__mapping_policy__ = MappingPolicy(policy) if isinstance(policy, int) else policy

    def mapped_feed(self, data: str) -> tuple[list[str], dict[str, str]]:
        for name in self.__mappings__:
            self.__mappings__[name].clear()
        self.__inverse_mapping__.clear()
        self.__mask_mapping__ = True
        self.feed(data)
        self.__mask_mapping__ = False
        return self.buffer.copy(), dict(ChainMap(*[self.__mappings__[name] for name in self.__mappings__]))

    def mapped_reconstruct(self, tokens: list[str], mapping: dict[str, str]) -> str:
        self.__mask_mapping__ = True
        match self.__mapping_policy__:
            case MappingPolicy.SIMPLIFIED:
                result = self.limited_reconstruct(tokens, mapping)
            case _:
                for i, v in enumerate(tokens):
                    if v in mapping:
                        tokens[i] = mapping[v]
                result = self.reconstruct(tokens)
                keys = list(mapping.keys())
                keys.sort(reverse=True)
                for k in keys:
                    result = result.replace(k, mapping[k])
        self.__mask_mapping__ = False
        return result

    def limited_reconstruct(self, tokens: list[str], mapping: dict[str, str]) -> str:
        placeholders = list(self.__mappings__.keys())
        keys = {p: [] for p in placeholders}
        for k in mapping:
            for p in placeholders:
                if k.startswith(p):
                    keys[p].append(k)
                    break
        for k in keys:
            keys[k].sort()
        result = self.reconstruct(tokens)
        keys = {k: v for k, v in keys.items() if len(v)}
        for k in keys:
            while k in result and len(keys[k]):
                result = result.replace(k, mapping[keys[k].pop(0)], 1)
        return result

    def early_reconstruct(self, tokens: list[str], mapping: dict[str, str]) -> str:
        placeholders = list(mapping.keys())
        for i, token in enumerate(tokens):
            replaced = list()
            for p in placeholders:
                if p in token:
                    tokens[i] = token.replace(p, mapping[p])
                    replaced.append(p)

            for p in replaced:
                placeholders.remove(p)
                del mapping[p]

        return self.mapped_reconstruct(tokens, mapping)

    @staticmethod
    def eliminate_mapping_duplicates(tokens: list[str], mapping: dict[str, str]) -> tuple[list[str], dict[str, str]]:
        inverted_mapping = dict()
        for k, v in mapping.items():
            if v in inverted_mapping:
                inverted_mapping[v].append(k)
            else:
                inverted_mapping[v] = [k]

        sorted_candidates = [sorted(v) for v in inverted_mapping.values() if len(v) > 1]
        candidates = dict(ChainMap(*[{k: s[0] for k in s[1:]} for s in sorted_candidates]))

        for i, token in enumerate(tokens):
            if token in candidates:
                tokens[i] = candidates[token]
                del mapping[token]

        return tokens, mapping

    @staticmethod
    def reduce_mapping_distance(tokens: list[str], mapping: dict[str, str]) -> tuple[list[str], dict[str, str]]:
        def get_numeric_part(data: str) -> int:
            underscore_pos = data.find("_")
            return int(data[underscore_pos + 1:])

        def forge_token(data: str, sequence_num: int) -> str:
            underscore_pos = data.find("_")
            return f"{data[:underscore_pos]}_{sequence_num}"

        collection_of_keys = {}
        for k in mapping:
            pos = k.find("_")
            if pos > 0 and k[pos + 1:].isdigit():
                beginning = k[:pos]
                if beginning in collection_of_keys:
                    collection_of_keys[beginning].append(k)
                else:
                    collection_of_keys[beginning] = [k]

        ordered_mapping = OrderedDict()
        for collection in collection_of_keys.values():
            if len(collection) > 1:
                collection.sort()
                smallest_digit = get_numeric_part(collection[0])
                for i, token in enumerate(collection):
                    digit = get_numeric_part(token)
                    if digit > smallest_digit + 1:
                        smallest_digit += 1
                        ordered_mapping[token] = forge_token(token, smallest_digit)
            else:
                digit = get_numeric_part(collection[0])
                if digit != 1:
                    ordered_mapping[collection[0]] = forge_token(collection[0], 1)

        ordered_mapping = OrderedDict(sorted(list(ordered_mapping.items())))
        for k, v in ordered_mapping.items():
            mapping[v] = mapping[k]
            del mapping[k]

        for i, token in enumerate(tokens):
            if token in ordered_mapping:
                tokens[i] = ordered_mapping[token]

        return tokens, mapping


class WhitespaceTokenizer(AbstractTokenizer):
    def __init__(self, delimiter: str = " ", grammar: dict | None = None, placeholders: tuple = ()):
        super(WhitespaceTokenizer, self).__init__(grammar=grammar, placeholders=placeholders)
        self.__delimiter__ = delimiter

    def feed(self, data):
        self.buffer = data.split(self.__delimiter__)

    def reconstruct(self, tokens: list[str]) -> str:
        return self.__delimiter__.join(tokens)
