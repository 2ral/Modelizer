import random

from enum import Enum

from modelizer.tokenizers.encoder import (
    EncoderTokenizer,
    Tensor,
    Path,
    Any,
    Sequence,
    Iterable,
    Optional,
)

from modelizer.configs import UNKNOWN_FEATURE
from modelizer.utils import HashingHelpers, DataHandlers, Pickle


class EncodingPolicy(Enum):
    POSITIVE = "positive"
    NON_NEGATIVE = "non-negative"
    FULL = "full"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN

    @classmethod
    def valid_policies(cls) -> set[str]:
        values: set[str] = {str(member.value) for member in cls if member is not cls.UNKNOWN}
        return values


class ForgingPolicy(Enum):
    SPARSE = "sparse"
    UNSET = "unset"
    RANDOM = "random"
    REFERENCE = "reference"
    MUTATIONS = "mutations"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN

    @classmethod
    def valid_policies(cls) -> set[str]:
        values: set[str] = {str(member.value) for member in cls if member is not cls.UNKNOWN}
        return values


class FeatureEncoder:
    def __init__(self, features: Sequence[Any] | Iterable[Any],
                 encoding: str | EncodingPolicy = EncodingPolicy.POSITIVE,
                 forging: str | ForgingPolicy = ForgingPolicy.SPARSE,
                 ):
        """
        This class is responsible to automatically encode feature vectors into the tokenizable textual representation.
        :param features: sequence of features to be encoded. Features could be binary (e.g., "b_feature1", "b_feature2"),
                         ternary (e.g., "t_feature1", "t_feature2"), or any other string.
                         If the feature does not start with "b_" or "t_", it will be prefixed with "b_".
        :param encoding: str - The encoding type, either 'positive', 'non-negative', or 'full'.
        :param forging: str - The forging type, either 'sparse', 'unset', 'random', 'reference', or 'mutations'.
        """
        if isinstance(encoding, str):
            encoding = EncodingPolicy(encoding)
        assert encoding != EncodingPolicy.UNKNOWN, "encoding can be either 'positive', 'non-negative', or 'full'"

        if isinstance(forging, str):
            forging = ForgingPolicy(forging)
        assert forging != ForgingPolicy.UNKNOWN, "forging can be one of 'sparse', 'unset', 'random', 'reference' or 'mutations'"

        self._raw_features = tuple(features)
        self._features = self.__format_features__(features)
        assert len(self._features) > 0, "features must not be empty"
        assert len(set(self._features)) == len(self._features), "features must have unique values"

        self._encoding = encoding
        self.encode = self.__set_encoding_method__(encoding)
        self._forging = forging
        self.forge = self.__set_forging_method__(forging)
        self._filter_duplicates = False

    def __getstate__(self):
        return {
            "_raw_features": self._raw_features,
            "features": self._features,
            "encoding": self._encoding,
            "forging": self._forging,
        }

    def __setstate__(self, state):
        self._raw_features = state["_raw_features"]
        self._features = state["features"]
        self._encoding = state["encoding"]
        self._forging = state["forging"]
        self.forge = self.__set_forging_method__(state["forging"])
        self.encode = self.__set_encoding_method__(state["encoding"])

    @property
    def raw_features(self) -> tuple:
        return self._raw_features

    @property
    def features(self) -> tuple:
        return self._features

    @property
    def encoding(self) -> EncodingPolicy:
        return self._encoding

    @encoding.setter
    def encoding(self, value: str | EncodingPolicy):
        """
        Set the encoding type for the feature vector.
        :param value: str - The encoding type, either 'positive', 'non-negative', 'full'.
        """
        if isinstance(value, str):
            value = EncodingPolicy(value)
        if value == EncodingPolicy.UNKNOWN:
            raise ValueError("encoding must be either 'positive', 'non-negative', or 'full'")
        self.encode = self.__set_encoding_method__(value)
        self._encoding = value

    @property
    def forging(self) -> ForgingPolicy:
        return self._forging

    @forging.setter
    def forging(self, value: str | ForgingPolicy):
        """
        Set the forging type for the feature vector.
        :param value: str - The forging type, either 'sparse', 'unset', 'random', or 'reference'.
        """
        if isinstance(value, str):
            value = ForgingPolicy(value)
        if value == ForgingPolicy.UNKNOWN:
            raise ValueError("forging must be one of 'sparse', 'unset', 'random', 'reference', or 'mutations'")
        self.forge = self.__set_forging_method__(value)
        self._forging = value

    @property
    def filter_duplicates(self) -> bool:
        return self._filter_duplicates

    @filter_duplicates.setter
    def filter_duplicates(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError(f"filter_duplicates must be a boolean value, got {value} instead")
        self._filter_duplicates = value

    def __set_encoding_method__(self, value: EncodingPolicy):
        """
        Find the encoding type for the feature vector.
        :param value: str - The encoding type, either 'positive', 'non-negative', 'full'.
        :return: The corresponding encoding method (callable).
        """
        match value:
            case EncodingPolicy.POSITIVE:
                callable_func = self.encode_positive
            case EncodingPolicy.NON_NEGATIVE:
                callable_func = self.encode_non_negative
            case EncodingPolicy.FULL:
                callable_func = self.encode_full
            case _:
                raise ValueError("encoding must be either 'positive', 'non-negative', 'full'")
        return callable_func

    def __set_forging_method__(self, value: ForgingPolicy):
        """
        find the forging type for the feature vector.
        :param value: str - The forging type, either 'sparse', 'unset', 'random', 'reference' or 'mutations'.
        :return: The corresponding forging method (callable).
        """
        match value:
            case ForgingPolicy.UNSET:
                callable_func = self.forge_query_dense_unset
            case ForgingPolicy.RANDOM:
                callable_func = self.forge_query_dense_random
            case ForgingPolicy.REFERENCE | ForgingPolicy.MUTATIONS:
                callable_func = self.forge_query_dense_reference
            case ForgingPolicy.SPARSE:
                callable_func = self.forge_query_sparse
            case _:
                raise ValueError("forging must be one of 'sparse', 'unset', 'random', 'reference', or 'mutations'")
        return callable_func

    @staticmethod
    def __format_features__(features: Sequence[Any]) -> tuple[str, ...]:
        # Preserve leading '!' / '?' while auto-prefixing underlying feature names.
        formatted = []
        for f in features:
            if f == UNKNOWN_FEATURE:
                formatted.append(UNKNOWN_FEATURE)
                continue
            f = str(f)
            if f.startswith(("!", "?")):
                marker = f[0]
                f = f[1:]
            else:
                marker = ""
            if not f.startswith("b_") and not f.startswith("t_"):
                f = f"b_{f}"
            formatted.append(f"{marker}{f}")
        return tuple(formatted)

    @staticmethod
    def __decode_features__(features: Sequence[str] | Iterable[str] | str, to_string: bool = False) -> list[str] | str:
        if isinstance(features, str):
            features = features.split()
        decoded = []
        for f in features:
            if f == UNKNOWN_FEATURE:
                decoded.append(UNKNOWN_FEATURE)
                continue
            marker = ""
            name = f
            if name.startswith(("!", "?")):
                marker = name[0]
                name = name[1:]
            if name.startswith(("b_", "t_")):
                name = name.split("_", 1)[1]
            decoded.append(f"{marker}{name}")
        decoded = list(dict.fromkeys(decoded))
        return " ".join(decoded) if to_string else decoded

    def __update__(self, features: Sequence[Any] | Iterable[Any]):
        new_raw_features = self.__decode_features__(features)
        new_features = self.__format_features__(new_raw_features)
        self._raw_features = tuple(set(self._raw_features).union(set(new_raw_features)))
        self._features = tuple(set(self._features).union(set(new_features)))

    def __len__(self) -> int:
        """
        Get the number of features in the feature vector.
        :return: int - The number of features.
        """
        return len(self._features)

    def index(self, feature: str) -> int:
        """
        Get the index of a feature in the feature vector.
        """
        if feature.startswith(("!", "?")):
            feature = feature[1:]
        return -1 if feature not in self._features else self._features.index(feature)

    def __getitem__(self, index: int) -> Any:
        """
        Get the feature at the specified index.
        :param index: int - The index of the feature to retrieve.
        :return: str - The feature at the specified index.
        """
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)} => {index} instead")
        if index == -1:
            return UNKNOWN_FEATURE
        elif index < -1 or index >= len(self._features):
            raise IndexError(f"Index out of range. This index range is [0, {len(self._features) - 1}] but got {index} instead")
        return self._features[index]

    def encode_positive(self, features: Sequence[int | str], to_string: bool = False) -> list[str] | str:
        """
        Encode a vector into a compact representation, filtering out non-positive features.
        :param features:
          - Sequence[str] - A sequence of feature names (strings) indicating the presence or absence of features. !feature indicates absence, ?feature indicates undefined. This input type is recommended as less error-prone.
          - Sequence[int] - A sequence of integers or strings indicating the presence (1) or absence (0) of features or (-1) for undefined features.
        :param to_string: bool - If True, return a string representation; otherwise, return a list of feature names.
        :return: list[str] - A list of feature names that are present (i.e., have a value greater than 0).
        """
        features = list(features)
        if len(features) == 0:
            raise ValueError("vector must not be empty")
        first_elem_type = type(features[0])
        if not all(type(x) is first_elem_type for x in features) or first_elem_type not in (int, str):
            raise TypeError("all elements in the vector must be either string or integer")
        elif isinstance(features[0], int):
            if len(features) != len(self._features):
                raise ValueError(f"vector length {len(features)} does not match features length {len(self._features)}")
            result = [f for f, v in zip(self._features, features) if v > 0]
        else:
            features = self.__format_features__(features)
            result = [f for f in features if not f.startswith("!") and not f.startswith("?")]
        return " ".join(result) if to_string else result

    def encode_non_negative(self, features: Sequence[int | str], to_string: bool = False) -> list[str] | str:
        """
        Encode a vector into a compact representation, filtering out negative features.
        :param features:
          - Sequence[str] - A sequence of feature names (strings) indicating the presence or absence of features. !feature indicates absence, ?feature indicates undefined. This input type is recommended as less error-prone.
          - Sequence[int] - A sequence of integers or strings indicating the presence (1) or absence (0) of features or (-1) for undefined features.
        :param to_string: bool - If True, return a string representation; otherwise, return a list of feature names.
        :return: list[str] - A list of feature names that are present (i.e., have a value greater than 0).
        """
        features = list(features)
        if len(features) == 0:
            raise ValueError("vector must not be empty")
        first_elem_type = type(features[0])
        if not all(type(x) is first_elem_type for x in features) or first_elem_type not in (int, str):
            raise TypeError("all elements in the vector must be either string or integer")
        elif isinstance(features[0], int):
            if len(features) != len(self._features):
                raise ValueError(f"vector length {len(features)} does not match features length {len(self._features)}")
            result = [f if v > 0 else f"!{f}" for f, v in zip(self._features, features) if v >= 0]
        else:
            features = self.__format_features__(features)
            result = [f for f in features if not f.startswith("?")]
        return " ".join(result) if to_string else result

    def encode_full(self, features: Sequence[int | str], to_string: bool = False) -> list[str] | str:
        """
        Encode a vector into a full representation, including features with zeros or negative values.
        :param features:
          - Sequence[str] - A sequence of feature names (strings) indicating the presence or absence of features. !feature indicates absence, ?feature indicates undefined. This input type is recommended as less error-prone.
          - Sequence[int] - A sequence of integers or strings indicating the presence (1) or absence (0) of features or (-1) for undefined features.
        :param to_string: bool - If True, return a string representation; otherwise, return a list of feature names.
        :return: list[str] - A list of feature names, with absent features prefixed by '!' to indicate their absence or '?' to indicate the undefined status of the feature.
        """
        features = list(features)
        if len(features) == 0:
            raise ValueError("vector must not be empty")
        first_elem_type = type(features[0])
        if not all(type(x) is first_elem_type for x in features) or first_elem_type not in (int, str):
            raise TypeError("all elements in the vector must be of the same type either string or integer")
        elif isinstance(features[0], int):
            if len(features) != len(self._features):
                raise ValueError(f"vector length {len(features)} does not match features length {len(self._features)}")
            result = [f if v > 0 else f"?{f}" if v < 0 and f.startswith("t_") else f"!{f}" for f, v in zip(self._features, features)]
        else:
            result = self.__format_features__(features)
        return " ".join(result) if to_string else result

    def __pre_process_feature_vector__(self, features: Sequence[int | str]) -> list[str]:
        features = list(features)
        if len(features) == 0:
            # this is a fix to resolve an issue when monitored sequences are empty
            # raise ValueError("features must not be empty")
            return list()
        first_elem_type = type(features[0])
        if not all(type(x) is first_elem_type for x in features) or first_elem_type not in (int, str):
            raise TypeError("all elements in the vector must be of the same type either string or integer")

        if isinstance(features[0], int):
            if self._filter_duplicates:
                features = DataHandlers.deduplicate_keep_first(features)
            if len(features) != len(self._features):
                raise ValueError(f"vector length {len(features)} does not match features length {len(self._features)}")
            features = [f if v > 0 else f"?{f}" if v < 0 and f.startswith("t_") else f"!{f}" for f, v in zip(self._features, features)]
        else:
            if self._filter_duplicates:
                features = DataHandlers.deduplicate_keep_first(features, [UNKNOWN_FEATURE])
            else:
                cleaned_features = [f for f in features if f != UNKNOWN_FEATURE]
                if len(cleaned_features) != len(set(cleaned_features)):
                    raise ValueError(f"all features except {UNKNOWN_FEATURE} must be unique when duplicates are disallowed,\ngot {cleaned_features}")
            features = list(self.__format_features__(features))

        for idx, f in enumerate(features):
            if f.startswith("?"):
                f = f[1:]
                if not f.startswith("t_"):
                    raise ValueError(f"feature '{f}' is a binary feature and can not be undefined")
                if f in features:
                    raise ValueError(f"feature '{f}' can not be set and undefined at the same time")
                if f"!{f}" in features:
                    raise ValueError(f"feature '{f}' can not be undefined and unset at the same time")
            elif f.startswith("!"):
                f = f[1:]
                if f in features:
                    raise ValueError(f"feature '{f}' can not be set and unset at the same time")
                if f.startswith("t_") and f"?{f}" in features:
                    raise ValueError(f"feature '{f}' can not be unset and undefined at the same time")
            if f not in self._features:
                # raise ValueError(f"feature '{f}' not found in the predefined feature vector")
                features[idx] = UNKNOWN_FEATURE
        return features

    def forge_query_sparse(self, features: Sequence[int | str], *, to_string: bool = False, **_) -> list[str] | str:
        """
        Forge a sparse query from the given features.
        :param features:
          - Sequence[str] - A sequence of feature names (strings) indicating the presence or absence of features. !feature indicates absence, ?feature indicates undefined. This input type is recommended as less error-prone.
          - Sequence[int] - A sequence of integers or strings indicating the presence (1) or absence (0) of features or (-1) for undefined features.
        :param to_string: bool - If True, return a string representation; otherwise, return a list of features.
        :return: list[str] | str - A list of features or a string representation of the query.
        """
        features = self.__pre_process_feature_vector__(features)
        result = sorted(features, key=lambda f: (
            self._features.index(f[1:]) if (f.startswith("!") or f.startswith("?")) and f[1:] in self._features
            else self._features.index(f) if f in self._features
            else len(self._features),  # it is <|UNKNOWN_FEATURE|>
        ))
        return " ".join(result) if to_string else result

    def forge_query_dense_unset(self, features: Sequence[int | str], *, to_string: bool = False, **_) -> list[str] | str:
        """
        Forge a dense query with unspecified features set to unset (!feature).
        :param features:
          - Sequence[str] - A sequence of feature names (strings) indicating the presence or absence of features. !feature indicates absence, ?feature indicates undefined. This input type is recommended as less error-prone.
          - Sequence[int] - A sequence of integers or strings indicating the presence (1) or absence (0) of features or (-1) for undefined features.
        :param to_string: bool - If True, return a string representation; otherwise, return a list of features.
        :return: list[str] | str - A list of features with unspecified features marked as unset, or a string representation of the query.
        """
        features = self.__pre_process_feature_vector__(features)
        unknown_count = features.count(UNKNOWN_FEATURE)
        filtered_features = set([f for f in features if f != UNKNOWN_FEATURE])
        result = [f"?{f}" if f.startswith(f"t_") and f not in filtered_features and f"!{f}" not in filtered_features else f"!{f}" if f"!{f}" in filtered_features or f not in filtered_features else f for f in self._features]
        result = result + [UNKNOWN_FEATURE] * unknown_count
        return " ".join(result) if to_string else result

    def forge_query_dense_random(self, features: Sequence[int | str], *, to_string: bool = False, **_) -> list[str] | str:
        """
        Forge a dense query with random state for unspecified features.
        :param features:
          - Sequence[str] - A sequence of feature names (strings) indicating the presence or absence of features. !feature indicates absence, ?feature indicates undefined. This input type is recommended as less error-prone.
          - Sequence[int] - A sequence of integers or strings indicating the presence (1) or absence (0) of features or (-1) for undefined features.
        :param to_string: bool - If True, return a string representation; otherwise, return a list of features.
        :return: list[str] | str - A list of features with unspecified features randomly set to either the feature or its unset version, or a string representation of the query.
        """
        features = self.__pre_process_feature_vector__(features)
        unknown_count = features.count(UNKNOWN_FEATURE)
        filtered_features = set([f for f in features if f != UNKNOWN_FEATURE])
        result = [f if f in filtered_features else f"!{f}" if f"!{f}" in filtered_features else f"?{f}" if f"?{f}" in filtered_features else random.choice((f, f"!{f}", f"?{f}") if f.startswith("t_") else (f, f"!{f}")) for f in self._features]
        result = result + [UNKNOWN_FEATURE] * unknown_count
        return " ".join(result) if to_string else result

    def forge_query_dense_reference(self, features: Sequence[int | str], *, reference: Sequence[int | str],
                                    max_mutations: int = 0, to_string: bool = False) -> list[str] | str:
        """
        Forge a dense query based on a reference vector, replacing features in the reference with those from the feature vector.
        :param features:
          - Sequence[str] - A sequence of feature names (strings) indicating the presence or absence of features. !feature indicates absence, ?feature indicates undefined. This input type is recommended as less error-prone.
          - Sequence[int] - A sequence of integers or strings indicating the presence (1) or absence (0) of features or (-1) for undefined features.
        :param reference: Sequence[int | str] - A reference vector to base the query on.
        :param max_mutations: int - The maximum number of features to mutate in the reference vector. If zero, no mutations are applied.
        :param to_string: bool - If True, return a string representation; otherwise, return a list of features.
        :return: list[str] | str - A list of features with specified features replaced in the reference, or a string representation of the query.
        """
        reference: list = self.__pre_process_feature_vector__(reference)
        if len(reference) != len(self._features):
            raise ValueError(f"reference length {len(reference)} does not match features length {len(self._features)}")

        features: list = self.__pre_process_feature_vector__(features)
        unknown_count = features.count(UNKNOWN_FEATURE)
        features = [f for f in features if f != UNKNOWN_FEATURE]

        if not isinstance(max_mutations, int) or max_mutations < 0:
            raise ValueError("num_mutations must be a non-negative integer")

        # locating the features in the reference vector
        mapping = dict()
        for f in features:
            query_f = f[1:] if f.startswith("!") or f.startswith("?") else f
            index = self.index(query_f)
            if index < 0:
                raise ValueError(f"feature '{query_f}' not found in the predefined feature vector")
            mapping[index] = f

        # mutating the reference vector
        if 0 < max_mutations < len(reference) - len(mapping):
            all_indices = set(range(len(reference)))
            excluded_indices = set(mapping.keys())
            remaining_indices = list(all_indices - excluded_indices)
            num_mutations = random.randint(1, max_mutations)
            indices_to_mutate = random.sample(remaining_indices, num_mutations)

            for idx in indices_to_mutate:
                current_value = reference[idx]
                is_ternary = self._features[idx].startswith("t_")
                if is_ternary:
                    if current_value.startswith("?"):
                        selection_options = (self._features[idx], f"!{self._features[idx]}")
                    elif current_value.startswith("!"):
                        selection_options = (self._features[idx], f"?{self._features[idx]}")
                    else:
                        selection_options = (f"!{self._features[idx]}", f"?{self._features[idx]}")
                    reference[idx] = random.choice(selection_options)
                else:
                    reference[idx] = current_value[1:] if current_value.startswith("!") else f"!{current_value}"
        else:
            print(f"Too many mutations requested, skipping the mutation step."
                  f"\nTotal features count: {len(self._features)}"
                  f"\nSelected features count: {len(mapping)}"
                  f"\nNumber of requested mutations: {max_mutations}"
                  f"\nMaximum possible mutations: {len(reference) - len(mapping)}")

        # updating the reference vector with the mapping
        for idx, value in mapping.items():
            reference[idx] = value

        reference = reference + [UNKNOWN_FEATURE] * unknown_count
        return " ".join(reference) if to_string else reference

    def forge_undefined_feature_vector(self):
        """
        Forge a feature vector with all features set to undefined.
        :return: list[str] - A list of features with all features set to undefined.
        """
        return [f"?{f}" if f.startswith("t_") else f"!{f}" for f in self._features]


class FeatureTokenizer(EncoderTokenizer):
    """This class is responsible for creating a tokenizer for program execution features."""
    def __init__(self, path: Optional[str | Path] = None, feature_encoder: Optional[FeatureEncoder] = None):
        super().__init__(path)
        if self.path is not None and self.path.joinpath("feature.pkl").exists():
            loaded = Pickle.load(self.path / "feature.pkl")
            self._feature_encoder = loaded.get("feature_encoder", None)
            self._filter_feature_duplicates = loaded.get("filter_feature_duplicates", False)
            if self._feature_encoder is not None:
                self._feature_encoder.filter_duplicates = self._filter_feature_duplicates
            self.references = loaded.get("references", list())
            self._seen_patterns = loaded.get("seen_patterns", set())
            self._max_mutations = loaded.get("max_mutations", 0)
            self._unknown_feature_buffer_size = loaded.get("unknown_feature_buffer", 20)
        elif feature_encoder is not None and isinstance(feature_encoder, FeatureEncoder):
            self._feature_encoder = feature_encoder
            self.references = list()
            self._seen_patterns = set()
            self._max_mutations = 0
            self._unknown_feature_buffer_size: Optional[int] = None
            self._filter_feature_duplicates = False if self._feature_encoder is None else self._feature_encoder.filter_duplicates
        else:
            self._feature_encoder = None
            self.references = list()
            self._seen_patterns = set()
            self._max_mutations = 0
            self._unknown_feature_buffer_size: Optional[int] = None
            self._filter_feature_duplicates = False

    @property
    def feature_encoder(self) -> FeatureEncoder:
        return self._feature_encoder

    @property
    def forging_policy(self) -> ForgingPolicy:
        return self._feature_encoder.forging

    @forging_policy.setter
    def forging_policy(self, value: str | ForgingPolicy):
        if self._feature_encoder is None:
            raise ValueError("feature_encoder is not initialized. Please train the tokenizer first.")
        self._feature_encoder.forging = value

    @property
    def encoding_policy(self) -> EncodingPolicy:
        return self._feature_encoder.encoding

    @encoding_policy.setter
    def encoding_policy(self, value: str | EncodingPolicy):
        if self._feature_encoder is None:
            raise ValueError("feature_encoder is not initialized. Please train the tokenizer first.")
        self._feature_encoder.encoding = value

    @property
    def valid_forging_policies(self) -> set[str]:
        return ForgingPolicy.valid_policies()

    @property
    def valid_encoding_policies(self) -> set[str]:
        return EncodingPolicy.valid_policies()

    @property
    def max_mutations(self) -> int:
        return self._max_mutations

    @max_mutations.setter
    def max_mutations(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError("num_mutations must be a non-negative integer")
        self._max_mutations = value

    @property
    def max_sequence_length(self) -> int:
        buffer = self._unknown_feature_buffer_size if self._unknown_feature_buffer_size is not None else 0
        return self._max_sequence_length + buffer + 2

    @property
    def unknown_feature_buffer_size(self) -> int:
        return self._unknown_feature_buffer_size

    @unknown_feature_buffer_size.setter
    def unknown_feature_buffer_size(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"unknown_feature_buffer must be a non-negative integer, got {value} instead")
        if self._unknown_feature_buffer_size is not None:
            print(f"Warning: changing the size of unknown_feature_buffer after tokenized was trained may lead to unexpected behaviors. Previous value: {self._unknown_feature_buffer_size}, new value: {value}")
        self._unknown_feature_buffer_size = value

    @property
    def filter_feature_duplicates(self) -> bool:
        return self._filter_feature_duplicates

    @filter_feature_duplicates.setter
    def filter_feature_duplicates(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError(f"filter_feature_duplicates must be a boolean value, got {value} instead")
        self._filter_feature_duplicates = value
        if self._feature_encoder is not None:
            self._feature_encoder.filter_duplicates = self._filter_feature_duplicates

    def __getstate__(self):
        state = super().__getstate__()
        state["feature_encoder"] = Pickle.to_bytes(self._feature_encoder) if self._feature_encoder is not None else None
        state["references"] = self.references
        state["seen_patterns"] = self._seen_patterns
        state["max_mutations"] = self._max_mutations
        state["filter_feature_duplicates"] = self._filter_feature_duplicates
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._filter_feature_duplicates = state.get("filter_feature_duplicates", False)
        if state["feature_encoder"] is not None:
            self._feature_encoder = Pickle.from_bytes(state["feature_encoder"])
            if not isinstance(self._feature_encoder, FeatureEncoder):
                raise TypeError("feature_encoder must be an instance of FeatureEncoder")
            self._feature_encoder.filter_duplicates = self._filter_feature_duplicates
        else:
            self._feature_encoder = None
        self.references = state.get("references", list())
        self._seen_patterns = state.get("seen_patterns", set())
        self._max_mutations = state.get("max_mutations", 0)

    def stretch_vocabulary(self,
                           tokens: Sequence[Any] | Iterable[Any] | Any,
                           *,
                           save: bool = True) -> dict[Any, int]:
        if self._feature_encoder is None:
            raise ValueError("feature_encoder is not initialized. Please train the tokenizer first.")

        known_features = set(self._feature_encoder.features)
        normalized_tokens = self.normalize_stretch_tokens(tokens)
        self._feature_encoder.__update__(normalized_tokens)
        new_features = [feature for feature in self._feature_encoder.features if feature not in known_features]

        if not new_features:
            return {}

        stretch_tokens: list[str] = []
        for feature in new_features:
            stretch_tokens.append(feature)
            stretch_tokens.append(f"!{feature}")
            if feature.startswith("t_"):
                stretch_tokens.append(f"?{feature}")

        if self.references:
            self.references = [tuple(self._feature_encoder.forge_query_dense_unset(reference, to_string=False))
                               for reference in self.references]

        added = super().stretch_vocabulary(stretch_tokens, save=False)
        if save and self.path is not None:
            self.save(self.path)
        return added

    def save(self, path: str | Path):
        if path is None:
            print("path argument is None, so tokenizer is not saved to disk.")
        else:
            super().save(path)
            data = {
                "feature_encoder": self._feature_encoder,
                "references": self.references,
                "seen_patterns": self._seen_patterns,
                "max_mutations": self._max_mutations,
                "filter_feature_duplicates": self._filter_feature_duplicates,
            }
            Pickle.dump(data, Path(path).resolve() / "feature.pkl")

    def add_references(self, references: list[Sequence[str]]):
        """
        This method adds new reference feature sequences to the tokenizer.
        :param references: list[Sequence[str]] - A list of reference feature sequences to be added.
        """
        new_refs = set(tuple(ref) for ref in references)
        current = set(tuple(ref) for ref in self.references)
        self.references = list(current | new_refs)

    def train(self, data: Sequence[Sequence[str] | str] | Iterable[Sequence[str] | str],
              encoding_policy: str | EncodingPolicy = EncodingPolicy.POSITIVE,
              forging_policy: str | ForgingPolicy = ForgingPolicy.SPARSE,
              max_input_length: Optional[int] = None,
              separator: Optional[str] = None,
              unknown_feature_buffer_size: int = 25,
              *, legacy_padding_mode: bool = False, **_):
        """
        This method trains the tokenizer on the provided data.
        :param data: Training data, a sequence or iterable of sequences of features (strings).
        :param encoding_policy: a string or EncodingPolicy enum value indicating the encoding policy to use. Could be one of 'positive', 'non-negative', or 'full'.
        :param forging_policy: a string or ForgingPolicy enum value indicating the forging policy to use. Could be one of 'sparse', 'unset', 'random', 'reference', or 'mutations'.
        :param max_input_length: Optional maximum input length for the tokenizer.
        :param separator: Optional separator to use for splitting the input data,
        :param unknown_feature_buffer_size: the buffer size for unknown features.
        :param legacy_padding_mode: if True, uses the legacy padding mode, where padding token == eos token.
        """

        samples = []
        data = list(data)
        unique_tokens = {UNKNOWN_FEATURE, }

        if legacy_padding_mode:
            self._pad_token = self._eos_token

        if self._unknown_feature_buffer_size is None:
            self.unknown_feature_buffer_size = unknown_feature_buffer_size

        for idx, value in enumerate(data):
            if isinstance(value, str):
                tokens = tuple(value.split())
            else:
                tokens = tuple(value)
            data[idx] = tokens
            samples.append(tokens)
            unique_tokens.update(tokens)

        if self._feature_encoder is None:
            self._feature_encoder = FeatureEncoder(unique_tokens, encoding_policy, forging_policy)
        else:
            features_set = set(self._feature_encoder.raw_features)
            if not unique_tokens.issubset(features_set):
                unique_tokens |= features_set
                self._feature_encoder = FeatureEncoder(unique_tokens, self.feature_encoder.encoding, self.feature_encoder.forging)

        if self._encoder is not None:
            features_set = set(self._feature_encoder.features)
            encoder_set = set(self._encoder.classes_)
            if not features_set.issubset(encoder_set):
                data.append(self._feature_encoder.__decode_features__(features_set - encoder_set, to_string=True))
                self._encoder = None
            if not encoder_set.issubset(features_set):
                self._feature_encoder.__update__(encoder_set - features_set)

        dense_refs = []
        for sample in samples:
            dense = tuple(self._feature_encoder.forge_query_dense_unset(sample, to_string=False))
            dense_refs.append(dense)
        self.add_references(dense_refs)

        if self._encoder is None:
            data = [self._feature_encoder.encode(d, to_string=True) for d in data]
            self._seen_patterns.update([HashingHelpers.hash(d, fmt="hex") for d in data])
            super().train(data, max_input_length, separator)

    def __call__(self, data: Sequence[str] | str,
                 truncation: bool = True,
                 padding: bool = True,
                 return_tensors: bool = True,
                 reference: Optional[Sequence[str] | str] = None,
                 **kwargs) -> dict[str, list | Tensor]:
        """
        This method tokenizes the input data.
        :param data: data to be tokenized
        :param truncation: if True, truncates the input data to the maximum sequence length
        :param padding: if True, pads the input data to the maximum sequence length
        :param return_tensors: if True, returns the tokenized data as a tensor
        :param reference: an optional reference sequence of features. By default, it is None.
        :param kwargs: additional keyword arguments
        :return: a dictionary containing the tokenized input data, attention mask, etc.
        """
        if reference is None and len(self.references) > 0:
            reference = random.choice(self.references)

        for preprocessor in self._preprocessors:
            data = preprocessor(data)

        if isinstance(data, str):
            data = data.split()
        data = self._feature_encoder.forge(data, reference=reference, max_mutations=self.max_mutations, to_string=False)

        unknown_feature_count = data.count(UNKNOWN_FEATURE)
        if unknown_feature_count > self._unknown_feature_buffer_size:
            print(f"Warning: The number of unknown features in the input data exceeds the configured buffer size. "
                  f"Found {unknown_feature_count}. Trimming {unknown_feature_count - self._unknown_feature_buffer_size} unknown features to fit the buffer.")
            count = 0
            data = [d for d in data if d != UNKNOWN_FEATURE or (count := count + 1) <= self._unknown_feature_buffer_size]

        data = " ".join(data)
        mutations = 0 if self._feature_encoder.forging != ForgingPolicy.MUTATIONS else self.max_mutations
        if mutations > 0:
            hash_value = HashingHelpers.hash(data, fmt="hex")
            trials = 10
            while hash_value in self._seen_patterns and trials > 0:
                trials -= 1
                data = self._feature_encoder.forge(data, reference=reference, max_mutations=mutations, to_string=True)
                hash_value = HashingHelpers.hash(data, fmt="hex")
        tokenized = self.__tokenize__(data, truncation, padding, **kwargs)
        return {k: self._to_tensor(v) for k, v in tokenized.items()} if self._torch_dtype is not None and return_tensors else tokenized
