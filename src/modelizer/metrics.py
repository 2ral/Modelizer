import numpy

from torch import Tensor
from pandas import DataFrame, Series, concat as pd_concat

from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from json import dumps as json_dumps
from typing import Optional, Callable, Union, Any, Sequence, Tuple, List

from jiwer import wer, mer, wip
from nltk import download as nltk_download
from nltk import translate as nltk_translate
from Levenshtein import distance as lev_distance, ratio as lev_ratio

from modelizer.configs import UNKNOWN_FEATURE
from modelizer.utils import SingletonMeta, Pickle


########################################################################################################################
#                                             Results DataClasses                                                      #
########################################################################################################################
@dataclass
class ValidationResults:
    """A container class that stores the results of the validation."""

    def __init__(self, backward_mode: bool,
                 model_input: Any,
                 model_input_tokens: Sequence[int] | Tensor,
                 model_output: Any,
                 model_output_tokens: Sequence[int] | Tensor,
                 ground_truth: Any,
                 ground_truth_tokens: Sequence[int] | Tensor):
        """
        Initializes the ValidationResults class.
        :param model_input: The model inputs
        :param model_output: The model prediction
        :param ground_truth: The ground truth data
        :param model_output_tokens: The model prediction tokenized
        :param ground_truth_tokens: The ground truth tokenized
        """
        self.used_trials = 0
        self.backward_mode = backward_mode
        self.model_input = model_input
        self.model_input_tokens = self.tensor_to_list(model_input_tokens)
        self.model_output = model_output
        self.model_output_tokens = self.tensor_to_list(model_output_tokens)
        self.ground_truth = ground_truth
        self.ground_truth_tokens = self.tensor_to_list(ground_truth_tokens)

        if self.backward_mode:
            self.__is_equal_raw__ = self.ground_truth == self.model_input
            self.__is_equal_tokenized__ = self.ground_truth_tokens == self.model_input_tokens
            self.__is_subset_tokenized__ = set(self.model_input_tokens).issubset(set(self.ground_truth_tokens))
        else:
            self.__is_equal_raw__ = ground_truth == model_output
            self.__is_equal_tokenized__ = self.ground_truth_tokens == self.model_output_tokens
            self.__is_subset_tokenized__ = set(self.model_output_tokens).issubset(set(self.ground_truth_tokens))

    @property
    def is_evaluable(self) -> bool:
        return len(self.ground_truth_tokens) > 0

    def __eq__(self, other):
        return (
            self.model_input_tokens == other.model_input_tokens and
            self.model_output_tokens == other.model_output_tokens and
            self.ground_truth_tokens == other.ground_truth_tokens
        )

    def __ne__(self, other):
        return (
            self.model_input_tokens != other.model_input_tokens or
            self.model_output_tokens != other.model_output_tokens or
            self.ground_truth_tokens != other.ground_truth_tokens
        )

    def __hash__(self):
        return hash((json_dumps(self.model_input_tokens), json_dumps(self.model_output_tokens), json_dumps(self.ground_truth_tokens)))

    def __str__(self):
        return f"Input: {self.model_input_tokens}\nOutput: {self.model_output_tokens}\nGround Truth: {self.ground_truth_tokens}"

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        return {
            "used_trials": self.used_trials,
            "backward_mode": self.backward_mode,
            "is_equal_raw": self.__is_equal_raw__,
            "is_equal_tokenized": self.__is_equal_tokenized__,
            "is_subset_tokenized": self.__is_subset_tokenized__,
            "model_input": Pickle.to_bytes(self.model_input),
            "model_input_tokens": Pickle.to_bytes(self.model_input_tokens),
            "model_output": Pickle.to_bytes(self.model_output),
            "model_output_tokens": Pickle.to_bytes(self.model_output_tokens),
            "ground_truth": Pickle.to_bytes(self.ground_truth),
            "ground_truth_tokens": Pickle.to_bytes(self.ground_truth_tokens),
        }

    def __setstate__(self, state):
        self.used_trials = state["used_trials"]
        self.backward_mode = state["backward_mode"]
        self.__is_equal_raw__ = state["is_equal_raw"]
        self.__is_equal_tokenized__ = state["is_equal_tokenized"]
        self.__is_subset_tokenized__ = state["is_subset_tokenized"]
        self.model_input = Pickle.from_bytes(state["model_input"])
        self.model_input_tokens = Pickle.from_bytes(state["model_input_tokens"])
        self.model_output = Pickle.from_bytes(state["model_output"])
        self.model_output_tokens = Pickle.from_bytes(state["model_output_tokens"])
        self.ground_truth = Pickle.from_bytes(state["ground_truth"])
        self.ground_truth_tokens = Pickle.from_bytes(state["ground_truth_tokens"])

    def is_equal(self, strict: bool = True):
        """
        Check if the model output is equal to the ground truth.
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: True if the model output is equal to the ground truth, False otherwise.
        """
        if strict:
            return self.__is_equal_raw__ or self.__is_equal_tokenized__
        else:
            return self.__is_subset_tokenized__ or self.__is_equal_raw__ or self.__is_equal_tokenized__

    def get_training_data(self) -> Tuple[Any, Any]:
        if self.backward_mode:
            return self.ground_truth, self.model_output
        else:
            return self.model_input, self.ground_truth

    def get_subject_input_tokenized(self):
        return self.model_output_tokens if self.backward_mode else self.model_input_tokens

    def get_model_input(self):
        return self.model_input

    def get_edit_distance(self, strict: bool = True) -> int:
        """
        Computes the edit distance between the model prediction and program output.
        For the forward models, the comparison is between the model prediction and the program output.
        For the backward models, the comparison is between the model input and the program output.
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: The edit distance between the model prediction and program output.
        """
        y_pred = self.model_input if self.backward_mode else self.model_output
        return 0 if self.is_equal(strict) else Metrics.edit_distance(y_pred, self.ground_truth, strict)

    @staticmethod
    def tensor_to_list(value):
        """
        Converts a tensor or nested list tensor to a flat list.
        For a tensor, first applies tolist() and then unwraps nested single-item lists.
        """
        if isinstance(value, dict):
            value = value["input_ids"]
        if hasattr(value, 'tolist'):
            value = value.tolist()
        while isinstance(value, list) and value and isinstance(value[0], list) and len(value) == 1:
            value = value[0]
        return value


@dataclass(frozen=True)
class FeatureComparisonResults:
    """A container class that stores the results of the feature comparison.

    Supports:
    - positive tokens: normal features
    - negative tokens: string starting with `!`, meaning the feature must **not** be present.
    """
    common: int                 # satisfied positive features
    missing: int                # missing positive features (false negatives)
    true_size: int              # number of positive requested features
    is_subset: bool = False     # all positive requested ⊆ monitored and no negative violated

    neg_satisfied: int = 0      # negative features correctly absent in monitored
    neg_violated: int = 0       # negative features incorrectly present in monitored
    neg_size: int = 0           # total negative requested features

    @property
    def fraction(self) -> float:
        """Overall fraction of satisfied requested features (positive and negative)."""
        total_requested = self.true_size + self.neg_size
        total_satisfied = self.common + self.neg_satisfied
        return 1.0 if total_requested == 0 else round(total_satisfied / total_requested, 4)

    @property
    def pos_fraction(self) -> float:
        """Fraction of satisfied **positive** requested features."""
        return 1.0 if self.true_size == 0 else round(self.common / self.true_size, 4)

    @property
    def neg_fraction(self) -> float:
        """Fraction of satisfied **negative** requested features."""
        return 1.0 if self.neg_size == 0 else round(self.neg_satisfied / self.neg_size, 4)

    def __str__(self) -> str:
        return ", ".join(
            f"{k.replace('_', '').capitalize()}: {v}"
            for k, v in self.as_dict().items()
        )

    def as_dict(self):
        return {
            "fraction": self.fraction,
            "common": self.common,
            "missing": self.missing,
            "true_size": self.true_size,
            "is_subset": self.is_subset,
            "pos_fraction": self.pos_fraction,
            "neg_fraction": self.neg_fraction,
            "neg_satisfied": self.neg_satisfied,
            "neg_violated": self.neg_violated,
            "neg_size": self.neg_size,
        }

    @staticmethod
    def _split_pos_neg(tokens: List[str]) -> tuple[set[str], set[str]]:
        """Split tokens into (positive, negative_without_bang)."""
        pos, neg = set(), set()
        for t in tokens:
            if isinstance(t, str) and t.startswith("!"):
                if len(t) > 1:
                    neg.add(t[1:])
            else:
                pos.add(t)
        return pos, neg

    @staticmethod
    def compare_features(monitored_tokens: list[str], requested_tokens: list[str]) -> "FeatureComparisonResults":
        """
        Compare `requested_tokens` (may include negative tokens like "!x")
        to `monitored_tokens` (actually observed during execution).

        - Positive tokens (e.g. "A") must be present in `monitored_tokens`.
        - Negative tokens (e.g. "!B") require that "B" is *not* in `monitored_tokens`.
        """
        pred_set = set(monitored_tokens)

        # requested tokens: positives and negatives
        true_pos, true_neg = FeatureComparisonResults._split_pos_neg(requested_tokens)

        # positive part
        common_pos = len(pred_set & true_pos)
        missing_true = len(true_pos - pred_set)  # false negatives

        # negative part: a negative feature "!x" condition is *satisfied* if "x" is NOT in pred_set
        neg_size = len(true_neg)
        neg_violated = len(pred_set & true_neg)
        neg_satisfied = neg_size - neg_violated

        is_subset = False if neg_violated > 0 else true_pos.issubset(pred_set)

        return FeatureComparisonResults(
            common=common_pos,
            missing=missing_true,
            true_size=len(true_pos),
            is_subset=is_subset,
            neg_satisfied=neg_satisfied,
            neg_violated=neg_violated,
            neg_size=neg_size,
        )

    @staticmethod
    def empty(requested_tokens: list[str]) -> "FeatureComparisonResults":
        true_pos, true_neg = FeatureComparisonResults._split_pos_neg(requested_tokens)
        return FeatureComparisonResults(
            common=0,
            missing=len(true_pos),
            true_size=-1,
            is_subset=False,
            neg_satisfied=0,
            neg_violated=len(true_neg),
            neg_size=-1,
        )


@dataclass
class FeatureResults:
    """A container class that stores the results of the model predictions and monitored executions."""
    program_input: Any  # program input, which could be given by the user or generated by the model
    input_tokens: list[str]  # tokenized representation of either program input or meta-input tokens
    features: list[str]  # a set of execution features given as input to the model or predicted by the model
    feature_tokens: list[str]  # tokenized representation of the features
    monitored: list[str]  # features collected from the program after execution
    monitored_tokens: list[str]  # tokenized representation of the monitored features
    meta_input: Optional[Any] = None  # optional pattern

    is_backward: bool = False
    used_trials = 0
    computed_results: FeatureComparisonResults = None

    @property
    def is_evaluable(self) -> bool:
        return len(self.monitored_tokens) > 0

    def __eq__(self, other):
        if not isinstance(other, FeatureResults):
            return NotImplemented
        return (
            self.is_backward == other.is_backward and
            self.input_tokens == other.input_tokens and
            self.features == other.features and
            self.feature_tokens == other.feature_tokens and
            self.monitored == other.monitored and
            self.monitored_tokens == other.monitored_tokens
        )

    def __hash__(self):
        return hash((
            self.is_backward,
            json_dumps(self.input_tokens, ensure_ascii=False),
            json_dumps(self.features, ensure_ascii=False),
            json_dumps(self.feature_tokens, ensure_ascii=False),
            json_dumps(self.monitored, ensure_ascii=False),
            json_dumps(self.monitored_tokens, ensure_ascii=False),
        ))

    def compare_features(self, comparator_func: Optional[Callable[[list[str], list[str]], FeatureComparisonResults]] = None):
        if len(self.monitored_tokens) > 0:
            if comparator_func is None:
                comparator_func = FeatureComparisonResults.compare_features
            self.computed_results = comparator_func(self.monitored_tokens, self.feature_tokens)
        else:
            self.computed_results = FeatureComparisonResults.empty(self.feature_tokens)
        return self.computed_results

    def is_subset(self) -> bool:
        """Returns True if FeatureTokens ⊆ MonitoredTokens."""
        if self.computed_results is None:
            self.compare_features()
        return self.computed_results.is_subset if self.is_evaluable else False

    def is_exact_match(self) -> bool:
        """Returns True iff FeatureTokens == MonitoredTokens."""
        return set(self.feature_tokens) == set(self.monitored_tokens) if self.is_evaluable else False

    def is_equal(self, strict: bool = True) -> bool:
        """strict=True -> exact match, strict=False -> subset."""
        if self.is_evaluable:
            return self.is_exact_match() if strict else self.is_subset()
        else:
            return False

    def found_unknown_features(self):
        return UNKNOWN_FEATURE in set(self.monitored_tokens) if self.is_evaluable else False

    def get_training_data(self):
        if self.is_backward:
            return self.monitored, self.program_input if self.meta_input is None else self.meta_input
        else:
            return self.program_input if self.meta_input is None else self.meta_input, self.monitored

    def get_subject_input_tokenized(self):
        return self.input_tokens

    def get_model_input(self):
        return self.features if self.is_backward else self.program_input


#########################################################################################################################
#                                             Feature Metrics Class                                                     #
#########################################################################################################################
class FeatureMetrics:
    """
        Set-overlap metrics for predicted features vs. monitored features.

        - Excellent result: Features ⊆ Monitored.
        - Works with either List[FeatureResults] or a pre-built dataframe
          with columns: ["Input", "Features", "Monitored", "FeatureTokens", "MonitoredTokens", "Is subset",
                         "Found unknown", "Common", "Missing", "Fraction", "TrueSizeTokens", "Backward"].

        -- recall - fraction of requested features that were present in execution.
        -- precision - fraction of exhibited features that were actually requested.
        -- excellent accuracy will mean a fraction of cases where all requested features were present in execution.
    """

    @staticmethod
    def to_computed_results(results: List["FeatureResults"], comparator: Callable[[Any, Any], FeatureComparisonResults] = None, recompute: bool = False) -> DataFrame:
        """Build the canonical computed_results dataframe from a list of FeatureResults."""
        rows = []
        for res in results:
            if res.computed_results is None or recompute:
                res.compare_features(comparator)
            rows.append({
                "Input": res.program_input,
                "Features": res.features,
                "Monitored": res.monitored,
                "FeatureTokens": res.feature_tokens,
                "MonitoredTokens": res.monitored_tokens,
                "Is subset": res.is_subset(),
                "Found unknown": res.found_unknown_features(),
                "Common": res.computed_results.common,
                "Missing": res.computed_results.missing,
                "Fraction": res.computed_results.pos_fraction,
                "TrueSizeTokens": res.computed_results.true_size,
                "Backward": res.is_backward,
            })
        return DataFrame(rows)

    @classmethod
    def _ensure_df(cls, data: Union[List["FeatureResults"], DataFrame]) -> DataFrame:
        if isinstance(data, DataFrame):
            df = data.copy()
        elif isinstance(data, list):
            df = cls.to_computed_results(data)
        else:
            raise TypeError("data must be a pandas.DataFrame or List[FeatureResults]")

        if "Monitored" in df.columns:
            def _monitor_key(v):
                if isinstance(v, (list, tuple)):
                    return tuple(v)
                return (v,) if v is not None else tuple()

            keys = df["Monitored"].apply(_monitor_key)
            counts = keys.value_counts()
            df["Unique monitored"] = keys.map(lambda k: counts.get(k, 0) == 1)
        else:
            df["Unique monitored"] = False

        if "Found unknown" in df.columns:
            found_unknown = df["Found unknown"].astype(bool)
        else:
            found_unknown = Series([False] * len(df), index=df.index)
        df["Found unknown and unique monitored"] = found_unknown & df["Unique monitored"]

        return df

    @staticmethod
    def _as_set(value: Any) -> set:
        if value is None:
            return set()
        elif isinstance(value, set):
            return value
        elif isinstance(value, (list, tuple)):
            return set(value)
        else:
            return {value}

    @staticmethod
    def _per_row_stats(row: Series) -> Series:
        true_set = FeatureMetrics._as_set(row["FeatureTokens"] if "FeatureTokens" in row and row["FeatureTokens"] is not None else row.get("Features"))
        pred_set = FeatureMetrics._as_set(row["MonitoredTokens"] if "MonitoredTokens" in row and row["MonitoredTokens"] is not None else row.get("Monitored"))

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        union = len(pred_set | true_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        jaccard = tp / union if union else 0.0

        exact = 1.0 if pred_set == true_set else 0.0
        excellent_flag = 1.0 if true_set and true_set.issubset(pred_set) else 0.0

        return Series({
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Jaccard": jaccard,
            "ExactMatch": exact,
            "Excellent": excellent_flag,
            "PredSize": len(pred_set),
            "TrueSize": len(true_set),
        })

    @classmethod
    def _summarize(cls, df: DataFrame) -> dict:
        stats = df.apply(cls._per_row_stats, axis=1) if len(df) else DataFrame()
        unknown_found_cases = int(df["Found unknown"].sum()) if "Found unknown" in df.columns else 0

        if stats.empty:
            return {
                "n": 0,
                "excellent_accuracy": 0.0,
                "subset_accuracy": 0.0,
                "exact_match_accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "macro_jaccard": 0.0,
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "micro_jaccard": 0.0,
                "weighted_fraction": 0.0,
                "unknown_found_cases": 0,
            }

        macro = stats[["Precision", "Recall", "F1", "Jaccard"]].mean().to_dict()
        tp, fp, fn = stats["TP"].sum(), stats["FP"].sum(), stats["FN"].sum()
        micro_precision = tp / (tp + fp) if (tp + fp) else 0.0
        micro_recall = tp / (tp + fn) if (tp + fn) else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
        micro_jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

        if "Fraction" in df.columns and "TrueSize" in stats.columns and stats["TrueSize"].sum():
            weighted_fraction = float((df["Fraction"] * stats["TrueSize"]).sum() / stats["TrueSize"].sum())
        elif "Fraction" in df.columns and "TrueSizeTokens" in df.columns and df["TrueSizeTokens"].sum():
            weighted_fraction = float((df["Fraction"] * df["TrueSizeTokens"]).sum() / df["TrueSizeTokens"].sum())
        else:
            weighted_fraction = 0.0

        excellent_acc = float(stats["Excellent"].mean())
        exact_acc = float(stats["ExactMatch"].mean())

        return {
            "n": int(len(df)),
            "excellent_accuracy": excellent_acc,
            "subset_accuracy": excellent_acc,
            "exact_match_accuracy": exact_acc,
            "macro_precision": float(macro["Precision"]),
            "macro_recall": float(macro["Recall"]),
            "macro_f1": float(macro["F1"]),
            "macro_jaccard": float(macro["Jaccard"]),
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
            "micro_jaccard": float(micro_jaccard),
            "weighted_fraction": weighted_fraction,
            "unknown_found_cases": unknown_found_cases,
        }

    @classmethod
    def compute_distinguished_metrics(cls, data: Union[List["FeatureResults"], DataFrame]) -> dict:
        """Returns overall metrics and per-'Backward' metrics."""
        df = cls._ensure_df(data)
        overall = cls._summarize(df)
        by_mode = {
            str(mode): cls._summarize(group.copy())
            for mode, group in df.groupby("Backward", dropna=False)
        }
        return {"overall": overall, "by_mode": by_mode}

    @classmethod
    def compute_metrics(cls, data: Union[List["FeatureResults"], DataFrame]) -> DataFrame:
        """Returns a single dataframe with appended summary rows first, followed by detailed rows."""
        df = cls._ensure_df(data)

        # per-row stats
        row_stats = df.apply(cls._per_row_stats, axis=1) if len(df) else DataFrame()
        rows_df = pd_concat([df.reset_index(drop=True), row_stats], axis=1)
        rows_df["RowType"] = "row"
        rows_df["Group"] = "row"

        # overall summary
        overall = cls._summarize(df)
        overall_row = {"RowType": "summary", "Group": "overall", **overall}

        # per-Backward summaries
        by_mode_rows = []
        for mode, group in df.groupby("Backward", dropna=False):
            mode_summary = cls._summarize(group.copy())
            # label by Backward mode value (e.g., True/False/NaN as string)
            by_mode_rows.append({"RowType": "summary", "Group": f"Backward={mode}", **mode_summary})

        # columns unification
        summary_df = DataFrame([overall_row] + by_mode_rows)
        all_cols = list(dict.fromkeys(list(summary_df.columns) + list(rows_df.columns)))
        summary_df = summary_df.reindex(columns=all_cols)
        rows_df = rows_df.reindex(columns=all_cols)

        return pd_concat([summary_df, rows_df], ignore_index=True)


#########################################################################################################################
#                                                  Metrics Class                                                       #
#########################################################################################################################
class Metrics(metaclass=SingletonMeta):
    def __init__(self, round_precision: int = 4):
        """
        Initializes the Metrics class.
        Attempts to find the WordNet corpus and downloads it if not present.
        Sets up various metric functions and a smoothing function for BLEU.
        """
        assert isinstance(round_precision, int) and round_precision > 0, "round_precision must be a positive integer."
        if not Path.home().joinpath('nltk_data', 'corpora', 'wordnet.zip').is_file():
            nltk_download('wordnet')
        self._round_precision = round_precision
        self._jiwer_wer = wer
        self._jiwer_mer = mer
        self._jiwer_wip = wip
        self._nltk_meteor = nltk_translate.meteor_score
        self._nltk_bleu = nltk_translate.bleu_score
        self._nltk_nist = nltk_translate.nist_score
        self._nltk_chrf = nltk_translate.chrf_score
        self._nltk_gleu = nltk_translate.gleu_score
        self._smoothing = self._nltk_bleu.SmoothingFunction()

    @property
    def round_precision(self) -> int:
        return self._round_precision

    @round_precision.setter
    def round_precision(self, value: int):
        assert isinstance(value, int), f"round_precision must be an integer, got type={type(value)} instead."
        assert value > 0, f"round_precision must be greater than 0, got {value} instead."
        self._round_precision = value

    def __getstate__(self):
        """
        Returns the state of the Metrics class for serialization.
        :return: A dictionary containing the state of the Metrics class.
        """
        return {
            "round_precision": self._round_precision,
            "_jiwer_wer": self._jiwer_wer,
            "_jiwer_mer": self._jiwer_mer,
            "_jiwer_wip": self._jiwer_wip,
            "_nltk_meteor": self._nltk_meteor,
            "_nltk_bleu": self._nltk_bleu,
            "_nltk_nist": self._nltk_nist,
            "_nltk_chrf": self._nltk_chrf,
            "_nltk_gleu": self._nltk_gleu,
            "_smoothing": self._smoothing
        }

    def __setstate__(self, state):
        """
        Restores the state of the Metrics class from a serialized state.
        :param state: A dictionary containing the state of the Metrics class.
        """
        self._round_precision = state["round_precision"]
        self._jiwer_wer = state["_jiwer_wer"]
        self._jiwer_mer = state["_jiwer_mer"]
        self._jiwer_wip = state["_jiwer_wip"]
        self._nltk_meteor = state["_nltk_meteor"]
        self._nltk_bleu = state["_nltk_bleu"]
        self._nltk_nist = state["_nltk_nist"]
        self._nltk_chrf = state["_nltk_chrf"]
        self._nltk_gleu = state["_nltk_gleu"]
        self._smoothing = state["_smoothing"]

    ####################################################################################################################
    #                                               Helper Methods                                                     #
    ####################################################################################################################
    @staticmethod
    def corpus_tokens_to_strings(data: Sequence[Sequence[Sequence[str]]]) -> Tuple[List[str], List[str]]:
        """
        Prepares joined strings for the token sequences from the data.
        :param data: List of tuples (reference tokens, prediction tokens).
        :return: A tuple with two lists (y_trye, y_pred) of joined strings.
        """
        y_true = [" ".join(ref) for ref, _ in data]
        y_pred = [" ".join(pred) for _, pred in data]
        return y_true, y_pred

    ####################################################################################################################
    #                                       Sentence-level score metrics                                               #
    ####################################################################################################################
    def bleu_score(self, y_true: Sequence[str], y_pred: Sequence[str]) -> float:
        """
        Computes BLEU score for a single sentence pair.
        :param y_true: List of reference tokens.
        :param y_pred: List of predicted tokens.
        :return: The sentence-level BLEU score.
        """
        # noinspection PyTypeChecker
        # Inspection does not recognize the smoothing_function argument, but it is valid, according to the NLTK documentation.
        return self._nltk_bleu.sentence_bleu([y_true], y_pred, smoothing_function=self._smoothing.method1)

    def gleu_score(self, y_true: Sequence[str], y_pred: Sequence[str]) -> float:
        """
        Computes GLEU score for a single sentence pair.
        :param y_true: List of reference tokens.
        :param y_pred: List of predicted tokens.
        :return: The sentence-level GLEU score.
        """
        return self._nltk_gleu.sentence_gleu([y_true], y_pred)

    def nist_score(self, y_true: Sequence[str], y_pred: Sequence[str]) -> float:
        """
        Computes NIST score for a single sentence pair. Catches ZeroDivisionError.
        :param y_true: List of reference tokens.
        :param y_pred: List of predicted tokens.
        :return: The sentence-level NIST score, or 0.0 if a ZeroDivisionError occurs.
        """
        try:
            return self._nltk_nist.sentence_nist([y_true], y_pred)
        except ZeroDivisionError:
            return 0.0

    def chrf_score(self, y_true: Sequence[str], y_pred: Sequence[str]) -> float:
        """
        Computes chrF score for a single sentence pair.
        :param y_true: List of reference tokens.
        :param y_pred: List of predicted tokens.
        :return: The sentence-level chrF score.
        """
        return self._nltk_chrf.sentence_chrf(y_true, y_pred)

    def meteor_score(self, y_true: Sequence[str], y_pred: Sequence[str]) -> float:
        """
        Computes METEOR score for a single sentence pair.
        :param y_true: List of reference tokens.
        :param y_pred: List of predicted tokens.
        :return: The sentence-level METEOR score.
        """
        return self._nltk_meteor.single_meteor_score(y_true, y_pred)

    ####################################################################################################################
    #                                           Corpus-level score methods                                             #
    ####################################################################################################################
    def bleu_corpus(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes BLEU score on a corpus level.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The corpus-level BLEU score.
        """
        references = [[y_true] for y_true, _ in data]
        predictions = [y_pred for _, y_pred in data]
        # noinspection PyTypeChecker
        # Inspection does not recognize the smoothing_function argument, but it is valid, according to the NLTK documentation.
        return self._nltk_bleu.corpus_bleu(references, predictions, smoothing_function=self._smoothing.method1)
    
    def bleu_corpus2(self, references: Sequence[Sequence[Sequence[str]]], predictions: Sequence[Sequence[str]]) -> float:
        """
        Computes BLEU score on a corpus level.
        :param references: A sequence of sequences of reference tokens sequences.
        :param predictions: A sequence of predicted sequences of tokens.
        :return: The corpus-level BLEU score.
        """
        # noinspection PyTypeChecker
        # Inspection does not recognize the smoothing_function argument, but it is valid, according to the NLTK documentation.
        return self._nltk_bleu.corpus_bleu(references, predictions, smoothing_function=self._smoothing.method1)

    def gleu_corpus(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes GLEU score on a corpus level.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The corpus-level GLEU score.
        """
        references = [[y_true] for y_true, _ in data]
        predictions = [y_pred for _, y_pred in data]
        return self._nltk_gleu.corpus_gleu(references, predictions)
    
    def gleu_corpus2(self, references: Sequence[Sequence[Sequence[str]]], predictions: Sequence[Sequence[str]]) -> float:
        """
        Computes GLEU score on a corpus level.
        :param references: A sequence of sequences of a reference sequence of tokens.
        :param predictions: A sequence of predicted sequences of tokens.
        :return: The corpus-level BLEU score.
        """
        return self._nltk_gleu.corpus_gleu(references, predictions)

    def nist_corpus(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes NIST score on a corpus level.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The corpus-level NIST score.
        """
        references = [[y_true] for y_true, _ in data]
        predictions = [y_pred for _, y_pred in data]
        try:
            score = self._nltk_nist.corpus_nist(references, predictions)
        except ZeroDivisionError:
            score = 0.0
        return score
    
    def nist_corpus2(self, references: Sequence[Sequence[Sequence[str]]], predictions: Sequence[Sequence[str]]) -> float:
        """
        Computes NIST score on a corpus level.
        :param references: A sequence of sequences of a reference sequence of tokens.
        :param predictions: A sequence of predicted sequences of tokens.
        :return: The corpus-level NIST score.
        """
        try:
            score = self._nltk_nist.corpus_nist(references, predictions)
        except ZeroDivisionError:
            score = 0.0
        return score

    def chrf_corpus(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes chrF score on a corpus level.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The corpus-level chrF score.
        """
        # For chrF, the corpus function expects a list of joined strings.
        y_true, y_pred = self.corpus_tokens_to_strings(data)
        return self._nltk_chrf.corpus_chrf(y_true, y_pred)

    def meteor_corpus(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes the average METEOR score on a corpus level.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The average corpus-level METEOR score.
        """
        scores = [self._nltk_meteor.single_meteor_score(ref, pred) for ref, pred in data]
        return sum(scores) / len(scores) if scores else 0.0

    ####################################################################################################################
    #                                        Error and difference metrics                                              #
    ####################################################################################################################
    def match_error_rate(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes the Match Error Rate (MER): the percentage of tokens that were mis-predicted and inserted. Lower is better.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The MER percentage rounded to round_precision.
        """
        y_true, y_pred = self.corpus_tokens_to_strings(data)
        return round(self._jiwer_mer(y_true, y_pred) * 100, self._round_precision)

    def word_error_rate(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes the Word Error Rate (WER): the percentage of words that were mis-predicted. Lower is better.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The WER percentage rounded to round_precision.
        """
        y_true, y_pred = self.corpus_tokens_to_strings(data)
        return round(self._jiwer_wer(y_true, y_pred) * 100, self._round_precision)

    def word_info_preserved(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes Word Information Preserved (WIP): the percentage of correctly predicted words. Higher is better.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The WIP percentage rounded to round_precision.
        """
        y_true, y_pred = self.corpus_tokens_to_strings(data)
        return round(self._jiwer_wip(y_true, y_pred) * 100, self._round_precision)

    def word_info_lost(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes Word Information Lost (WIL): the percentage of incorrectly predicted words.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The WIL percentage rounded to round_precision.
        """
        return round(100 - self.word_info_preserved(data), self._round_precision)

    def edit_distance_distribution(self, data: Sequence[Tuple[Sequence[str], Sequence[str]]], strict: bool = True) -> dict[str, Union[int, float]]:
        """
        Returns a distribution of edit distances for the provided data.
        'exact_match_cases' counts zero distances,
        'close_match_cases' includes distances of 0 and 1,
        'far_match_cases' are cases with an edit distance greater than 1.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: A dictionary with edit distance distribution metrics.
        """
        if not data:
            return {
                "exact_match_cases": 0,
                "exact_match_proportion": 0.0,
                "close_match_cases": 0,
                "close_match_proportion": 0.0,
                "far_match_cases": 0,
                "far_match_proportion": 0.0,
                "average_edit_distance": 0.0,
                "average_similarity_ratio": 0.0,
            }

        distances = [self.edit_distance(ref, pred, strict) for ref, pred in data]
        ratios = [self.similarity_ratio(ref, pred, strict) for ref, pred in data]
        avg_ratio = sum(ratios) / len(ratios)
        avg_distance = sum(distances) / len(distances)
        counts = Counter(distances)
        total = len(data)
        exact_match = counts.get(0, 0)
        close_match = counts.get(1, 0) + exact_match
        far_match = total - close_match
        result = {
            "exact_match_cases": exact_match,
            "exact_match_proportion": round(exact_match / total * 100, self._round_precision),
            "close_match_cases": close_match,
            "close_match_proportion": round(close_match / total * 100, self._round_precision),
            "far_match_cases": far_match,
            "far_match_proportion": 0,
            "average_edit_distance": round(avg_distance, self._round_precision),
            "average_similarity_ratio": round(avg_ratio, self._round_precision),
        }
        result["far_match_proportion"] = round(100 - result["close_match_proportion"], self._round_precision)
        return result

    ####################################################################################################################
    #                                           Standard Error functions                                               #
    ####################################################################################################################
    def standard_edit_error(self, data: Sequence[Sequence[Sequence[str]]], strict: bool = False) -> float:
        """
        Computes the standard error for edit distances.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: The standard error is computed on the edit distances.
        """
        return self.standard_error(self.edit_distance, data, strict=strict)

    def standard_bleu_error(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes the standard error using sentence-level BLEU scores.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The standard error computed for BLEU scores.
        """
        return self.standard_error(self.bleu_score, data)

    def standard_gleu_error(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes the standard error using sentence-level GLEU scores.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The standard error computed for GLEU scores.
        """
        return self.standard_error(self.gleu_score, data)

    def standard_nist_error(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes the standard error using sentence-level NIST scores.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The standard error computed for NIST scores.
        """
        return self.standard_error(self.nist_score, data)

    def standard_chrf_error(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes the standard error using sentence-level chrF scores.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The standard error is computed for chrF scores.
        """
        return self.standard_error(self.chrf_score, data)

    def standard_meteor_error(self, data: Sequence[Sequence[Sequence[str]]]) -> float:
        """
        Computes the standard error using sentence-level METEOR scores.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The standard error computed for METEOR scores.
        """
        return self.standard_error(self.meteor_score, data)

    @staticmethod
    def standard_error(error_func: Callable[..., Union[int, float]],
                       data: Sequence[Sequence[Sequence[str]]], **kwargs) -> float:
        """
        Calculates the standard error across a list of computed errors.
        Requires at least 2 data points.
        :param error_func: A function that computes an error between two lists of strings.
        :param data: A list of tuples with (reference tokens, prediction tokens).
        :return: The standard error value.
        :raises ValueError: If data contains fewer than 2 elements.
        """
        if len(data) < 2:
            raise ValueError("Data list must contain at least 2 elements")
        errors = numpy.array([error_func(ref, pred, **kwargs) for ref, pred in data])
        return float(numpy.std(errors, ddof=1) / numpy.sqrt(errors.size))

    @staticmethod
    def cosine_similarity(y_true: Union[Sequence[str], str],
                          y_pred: Union[Sequence[str], str],
                          floating_precision: int = 6) -> float:
        """
        Calculates cosine similarity between two token sequences or strings.
        :param y_true: A list of tokens or a string representing reference.
        :param y_pred: A list of tokens or a string representing model prediction.
        :param floating_precision: The number of decimal places for rounding the result.
        :return: The cosine similarity as a floating point number.
        """
        u_true, counts_true = numpy.unique(y_true, return_counts=True)
        u_pred, counts_pred = numpy.unique(y_pred, return_counts=True)

        common_tokens, idx_true, idx_pred = numpy.intersect1d(u_true, u_pred, return_indices=True)
        dot = numpy.dot(counts_true[idx_true], counts_pred[idx_pred])

        norm_true = float(numpy.linalg.norm(counts_true))
        norm_pred = float(numpy.linalg.norm(counts_pred))

        if norm_true == 0 or norm_pred == 0:
            return 0.0

        similarity = dot / (norm_true * norm_pred)
        return round(similarity, floating_precision)

    @staticmethod
    def __validate_inputs__(y_true: Union[Sequence[str], str], y_pred: Union[Sequence[str], str]) -> tuple[list[Union[Sequence[str], str]], list[Union[Sequence[str], str]]]:
        if not isinstance(y_true, (str, list, tuple)):
            raise ValueError(
                f"Unsupported type for y_true ({type(y_true).__name__}); "
                "only str, list or tuple are supported."
            )

        if not isinstance(y_pred, (str, list, tuple)):
            raise ValueError(
                f"Unsupported type for y_pred ({type(y_pred).__name__}); "
                "only str, list or tuple are supported."
            )

        if isinstance(y_true, str) != isinstance(y_pred, str):
            raise ValueError("y_true and y_pred must be of the same type.")

        if isinstance(y_true, tuple):
            y_true = list(y_true)
            y_pred = list(y_pred)

        return y_true, y_pred

    @staticmethod
    def edit_distance(y_true: Union[Sequence[str], str], y_pred: Union[Sequence[str], str], strict: bool = True) -> int:
        """
        Computes the edit distance between two sequences (either strings or lists).
        :param y_true: The first sequence (reference); must be of type str or list.
        :param y_pred: The second sequence (prediction); must be the same type as y_true.
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: Edit distance as an integer.
        :raises ValueError: If input types are not supported or mismatched.
        """
        y_true, y_pred = Metrics.__validate_inputs__(y_true, y_pred)

        if strict:
            if y_true == y_pred:
                return 0
            elif isinstance(y_true, str):
                return lev_distance(y_pred, y_true)
            else:
                return lev_distance(" ".join(y_pred), " ".join(y_true))

        # non-strict branch
        if isinstance(y_true, str):
            if y_true in y_pred:
                return 0
            else:
                return lev_distance(y_pred, y_true)
        elif set(y_true).issubset(set(y_pred)):
            return 0
        else:
            return lev_distance(" ".join(y_pred), " ".join(y_true))

    @staticmethod
    def similarity_ratio(y_true: Union[Sequence[str], str], y_pred: Union[Sequence[str], str], strict: bool = True) -> float:
        """
        Computes the similarity ratio between two sequences (either strings or lists).
        :param y_true: The first sequence (reference); must be of type str or list.
        :param y_pred: The second sequence (prediction); must be the same type as y_true.
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: Similarity ratio as a float.
        :raises ValueError: If input types are not supported or mismatched.
        """
        y_true, y_pred = Metrics.__validate_inputs__(y_true, y_pred)

        if strict:
            if y_true == y_pred:
                return 1.0
            elif isinstance(y_true, str):
                return lev_ratio(y_pred, y_true)
            else:
                return lev_ratio(" ".join(y_pred), " ".join(y_true))

        # non-strict branch
        if isinstance(y_true, str):
            return 1.0 if y_true in y_pred else lev_ratio(y_pred, y_true)
        elif set(y_true).issubset(set(y_pred)):
            return 1.0
        else:
            return lev_ratio(" ".join(y_pred), " ".join(y_true))

    @staticmethod
    def compute_validity_rate(data: Sequence[ValidationResults], strict: bool = True) -> float:
        """
        Computes the percentage of valid predictions in a list of tuples.
        :param data: A list of tuples with shape [IsEqual, GroundTruth, Prediction, ModelInput]
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: The percentage of valid predictions.
        """
        valid = sum(d.is_equal(strict) for d in data)
        return round(valid / len(data) * 100, 2) if len(data) > 0 else 0.0

    @staticmethod
    def compute_avg_validity_distance(data: Sequence[ValidationResults], strict: bool = True) -> float:
        """
        Computes the average edit distance for a list of tuples.
        :param data: A list of tuples with [IsEqual, GroundTruth, Prediction, ModelInput] pairs.
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: The average edit distance.
        """
        distances = [d.get_edit_distance(strict) for d in data]
        return round(sum(distances) / len(distances), 2) if len(distances) > 0 else 0.0

    @staticmethod
    def compute_avg_validity_ratio(data: Sequence[ValidationResults], strict: bool = True) -> float:
        """
        Computes the average similarity ratio for a list of tuples.
        :param data: A list of tuples with [IsEqual, GroundTruth, Prediction, ModelInput] pairs.
        :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
        :return: The average similarity ratio.
        """
        ratios = [Metrics.similarity_ratio(d.ground_truth, d.model_output, strict) for d in data]
        return round(sum(ratios) / len(ratios), 2) if ratios else 0.0


########################################################################################################################
#                                       Score computation helper functions                                             #
########################################################################################################################
def _bleu_corpus(references, predictions):
    try:
        return Metrics().bleu_corpus2(references, predictions)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _gleu_corpus(references, predictions):
    try:
        return Metrics().gleu_corpus2(references, predictions)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _nist_corpus(references, predictions):
    try:
        return Metrics().nist_corpus2(references, predictions)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _chrf_corpus(data):
    try:
        return Metrics().chrf_corpus(data)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _meteor_corpus(data):
    try:
        return Metrics().meteor_corpus(data)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _standard_edit_error(data, strict):
    try:
        return Metrics().standard_edit_error(data, strict)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _standard_bleu_error(data):
    try:
        return Metrics().standard_bleu_error(data)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _standard_meteor_error(data):
    try:
        return Metrics().standard_meteor_error(data)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _match_error_rate(data):
    try:
        return Metrics().match_error_rate(data)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def _word_error_rate(data):
    try:
        return Metrics().word_error_rate(data)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return -1


def compute_metrics(data: Sequence[Tuple[Sequence[str], Sequence[str]]],
                    source: str,
                    target: str,
                    evaluation_type: str,
                    strict: bool = True, **_, ) -> dict[str, Any]:
    """
    Computes a dictionary with metrics for one evaluation type.
    :param data: A list of tuples with (reference tokens, prediction tokens).
    :param source: A string representing the name of the source language.
    :param target: A string representing the name of the target language.
    :param evaluation_type: A string indicating the evaluation type (e.g. "Synthetic", "Real", "Tuned").
    :param strict: If True, checks for strict equality; if False, checks for subset equality. Default is True.
    :return: A dictionary with scores and error metrics.
    """
    if not data:
        return {
            "Source": source,
            "Target": target,
            "Evaluation": evaluation_type,
            "Total Records": 0,
            "Minimum Expected Tokens": 0,
            "Maximum Expected Tokens": 0,
            "Average Expected Tokens": 0.0,
            "Minimum Predicted Tokens": 0,
            "Maximum Predicted Tokens": 0,
            "Average Predicted Tokens": 0.0,
            "BLEU": -1,
            "GLEU": -1,
            "NIST": -1,
            "chrF": -1,
            "METEOR": -1,
            "Edit Error": -1,
            "BLEU Error": -1,
            "METEOR Error": -1,
            "MER": -1,
            "WER": -1,
            "WIP": 0.0,
            "WIL": 100.0,
            "Exact match cases": 0,
            "Exact match proportion": 0.0,
            "Close match cases": 0,
            "Close match proportion": 0.0,
            "Far match cases": 0,
            "Far match proportion": 0.0,
            "Average edit distance": 0.0,
            "Average similarity ratio": 0.0,
        }

    metrics = Metrics()
    references = [[y_true] for y_true, _ in data]
    predictions = [y_pred for _, y_pred in data]
    expected_token_counts = numpy.array([len(ref) for ref, _ in data])
    predicted_token_counts = numpy.array([len(pred) for _, pred in data])

    score: dict[str, Any] = {
        "Source": source,
        "Target": target,
        "Evaluation": evaluation_type,
        "Total Records": len(data),
        "Minimum Expected Tokens": int(expected_token_counts.min()),
        "Maximum Expected Tokens": int(expected_token_counts.max()),
        "Average Expected Tokens": float(numpy.mean(expected_token_counts)),
        "Minimum Predicted Tokens": int(predicted_token_counts.min()),
        "Maximum Predicted Tokens": int(predicted_token_counts.max()),
        "Average Predicted Tokens": float(numpy.mean(predicted_token_counts)),
        "BLEU": _bleu_corpus(references, predictions),
        "GLEU": _gleu_corpus(references, predictions),
        "NIST": _nist_corpus(references, predictions),
        "chrF": _chrf_corpus(data),
        "METEOR": _meteor_corpus(data),
        "Edit Error": _standard_edit_error(data, strict),
        "BLEU Error": _standard_bleu_error(data),
        "METEOR Error": _standard_meteor_error(data),
        "MER": _match_error_rate(data),
        "WER": _word_error_rate(data),
        "WIP": metrics.word_info_preserved(data)
    }

    score["WIL"] = round(100 - score["WIP"], 4)
    edit_dist_dist = metrics.edit_distance_distribution(data, strict)
    score.update({k.replace("_", " ").capitalize(): v for k, v in edit_dist_dist.items()})
    return score
