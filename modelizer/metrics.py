import nltk.translate as nltk_translate

from typing import Callable
from statistics import stdev
from collections import Counter
from gc import collect as garbage_collect
from math import floor as math_floor, sqrt as math_sqrt

from pylev import levenshtein
from jiwer import wer, mer, wip
from nltk import download as nltk_download
from Levenshtein import distance as string_edit_distance
nltk_download('wordnet')


class Scores:
    def __init__(self):
        self.round_precision = 4
        self.__jiwer_wer__ = wer
        self.__jiwer_mer__ = mer
        self.__jiwer_wip__ = wip
        self.__nltk_meteor__ = nltk_translate.meteor_score
        self.__nltk_bleu__ = nltk_translate.bleu_score
        self.__nltk_nist__ = nltk_translate.nist_score
        self.__nltk_chrf__ = nltk_translate.chrf_score
        self.__nltk_gleu__ = nltk_translate.gleu_score
        self.__smoothing__ = nltk_translate.bleu_score.SmoothingFunction().method1

    def bleu_score(self, y_true: list[str], y_pred: list[str]) -> float:
        return self.__nltk_bleu__.sentence_bleu([y_true], y_pred, smoothing_function=self.__smoothing__)

    def bleu_corpus(self, data: list[tuple[list[str], list[str]]]) -> float:
        references = [[y_true] for y_true, _ in data]
        predictions = [y_pred for _, y_pred in data]
        return self.__nltk_bleu__.corpus_bleu(references, predictions, smoothing_function=self.__smoothing__)

    def gleu_score(self, y_true: list[str], y_pred: list[str]) -> float:
        return self.__nltk_gleu__.sentence_gleu([y_true], y_pred)

    def gleu_corpus(self, data: list[tuple[list[str], list[str]]]) -> float:
        references = [[y_true] for y_true, _ in data]
        predictions = [y_pred for _, y_pred in data]
        return self.__nltk_gleu__.corpus_gleu(references, predictions)

    def nist_score(self, y_true: list[str], y_pred: list[str]) -> float:
        return self.__nltk_nist__.sentence_nist([y_true], y_pred)

    def nist_corpus(self, data: list[tuple[list[str], list[str]]]) -> float:
        references = [[y_true] for y_true, _ in data]
        predictions = [y_pred for _, y_pred in data]
        return self.__nltk_nist__.corpus_nist(references, predictions)

    def chrf_score(self, y_true: list[str], y_pred: list[str]) -> float:
        return self.__nltk_chrf__.sentence_chrf(y_true, y_pred)

    def chrf_corpus(self, data: list[tuple[list[str], list[str]]]) -> float:
        references = [y_true for y_true, _ in data]
        predictions = [y_pred for _, y_pred in data]
        return self.__nltk_chrf__.corpus_chrf(references, predictions)

    def meteor_score(self, y_true: list[str], y_pred: list[str]) -> float:
        return self.__nltk_meteor__.single_meteor_score(y_true, y_pred)

    def meteor_corpus(self, data: list[tuple[list[str], list[str]]]) -> float:
        return sum(self.__nltk_meteor__.single_meteor_score(y_true, y_pred) for y_true, y_pred in data) / len(data)

    def match_error_rate(self, data: list[tuple[list[str], list[str]]]) -> float:
        # This value indicates the percentage of tokens that were incorrectly predicted and inserted.
        # The lower the value, the better the performance with a MER of 0 being a perfect score.
        y_true = [" ".join(y_true) for y_true, _ in data]
        y_pred = [" ".join(y_pred) for _, y_pred in data]
        return round(self.__jiwer_mer__(y_true, y_pred) * 100, self.round_precision)

    def word_error_rate(self, data: list[tuple[list[str], list[str]]]) -> float:
        # This value indicates the percentage of words that were incorrectly predicted.
        # The lower the value, the better the performance with a WER of 0 being a perfect score.
        y_true = [" ".join(y_true) for y_true, _ in data]
        y_pred = [" ".join(y_pred) for _, y_pred in data]
        return round(self.__jiwer_wer__(y_true, y_pred) * 100, self.round_precision)

    def word_info_preserved(self, data: list[tuple[list[str], list[str]]]) -> float:
        # This value indicates the percentage of words that were correctly predicted between a set of ground-truth sentences and a set of hypothesis sentences.
        # The higher the value, the better the performance with a WordInfoPreserved of 1 being a perfect score.
        y_true = [" ".join(y_true) for y_true, _ in data]
        y_pred = [" ".join(y_pred) for _, y_pred in data]
        return round(self.__jiwer_wip__(y_true, y_pred) * 100, self.round_precision)

    def word_info_lost(self, data: list[tuple[list[str], list[str]]]) -> float:
        # This value indicates the percentage of words that were incorrectly predicted between a set of ground-truth sentences and a set of hypothesis sentences.
        # The lower the value, the better the performance with a WordInfoLost of 0 being a perfect score.
        return round(100 - self.word_info_preserved(data), self.round_precision)

    def edit_distance_distribution(self, data: list[tuple[list[str], list[str]]]):
        edit_distances = Counter([self.edit_distance(y_true, y_pred) for y_true, y_pred in data])
        total = len(data)
        exact_match = edit_distances.setdefault(0, 0)
        close_match = edit_distances.setdefault(1, 0) + exact_match
        far_match = total - close_match
        results = {
            "exact_match_cases": exact_match,
            "exact_match_proportion": round(exact_match / total * 100, self.round_precision),
            "close_match_cases": close_match,
            "close_match_proportion": round(close_match / total * 100, self.round_precision),
            "far_match_cases": far_match,
        }
        results["far_match_proportion"] = round(100 - results["close_match_proportion"], self.round_precision)
        return results

    def standard_edit_error(self, data: list[tuple[list[str], list[str]]]) -> float:
        return self.standard_error(self.edit_distance, data)

    def standard_blue_error(self, data: list[tuple[list[str], list[str]]]) -> float:
        return self.standard_error(self.bleu_score, data)

    def standard_gleu_error(self, data: list[tuple[list[str], list[str]]]) -> float:
        return self.standard_error(self.gleu_score, data)

    def standard_nist_error(self, data: list[tuple[list[str], list[str]]]) -> float:
        return self.standard_error(self.nist_score, data)

    def standard_chrf_error(self, data: list[tuple[list[str], list[str]]]) -> float:
        return self.standard_error(self.chrf_score, data)

    def standard_meteor_error(self, data: list[tuple[list[str], list[str]]]) -> float:
        return self.standard_error(self.meteor_score, data)

    @staticmethod
    def standard_error(error_func: Callable[[list[str], list[str]], int | float],
                       data: list[tuple[list[str], list[str]]]) -> float:
        assert len(data) > 1, "Data list must contain at least 2 elements"
        errors = [error_func(y_true, y_pred) for y_true, y_pred in data]
        return stdev(errors) / math_sqrt(len(errors))

    @staticmethod
    def cosine_similarity(y_true: list[str] | str, trg: list[str] | str, floating_precision: int = 6) -> float:
        true_counter, pred_counter = Counter(y_true), Counter(trg)
        tokens = set(true_counter.keys()).union(set(pred_counter.keys()))
        true_vector = [true_counter.setdefault(token, 0) for token in tokens]
        pred_vector = [pred_counter.setdefault(token, 0) for token in tokens]
        len_true = sum(s * s for s in true_vector) ** 0.5
        len_pred = sum(t * t for t in pred_vector) ** 0.5
        dot = sum(s * t for s, t in zip(true_vector, pred_vector))
        return round(dot / (len_true * len_pred), floating_precision)

    @staticmethod
    def edit_distance(y_true: list | str, y_pred: list | str) -> int:
        if not isinstance(y_true, (str, list)):
            raise ValueError(f"Unsupported type for y_true: {type(y_true)} | Only str and list are supported")
        if not isinstance(y_pred, (str, list)):
            raise ValueError(f"Unsupported type for y_pred: {type(y_pred)} | Only str and list are supported")
        if type(y_true) is not type(y_pred):
            raise ValueError(f"y_true and y_pred must be of the same type")
        if y_true == y_pred:
            return 0
        elif isinstance(y_true, str):
            return string_edit_distance(y_true, y_pred)
        else:
            return levenshtein(y_true, y_pred)


def __compute_score__(predictions: list, size: str, partitioned: bool, simplified: bool, source: str, target: str,
                      eval_type: str):
    score_calculator = Scores()
    data = [(predictions[i]["target_tokens"], predictions[i]["b1"]) for i in range(len(predictions))]
    all_expected_tokens = [len(predictions[i]["target_tokens"]) for i in range(len(predictions))]
    all_predicted_tokens = [len(predictions[i]["b1"]) for i in range(len(predictions))]
    score = {
        "Source": source,
        "Target": target,
        "Train Set": size,
        "Partitioned": partitioned,
        "Enumerated": not simplified,
        "Total Records": len(predictions),
        "Evaluation Type": eval_type,
        "Minimum Expected Tokens": min(all_expected_tokens),
        "Maximum Expected Tokens": max(all_expected_tokens),
        "Average Expected Tokens": math_floor(sum(all_expected_tokens) / len(predictions)),
        "Minimum Predicted Tokens": min(all_predicted_tokens),
        "Maximum Predicted Tokens": max(all_predicted_tokens),
        "Average Predicted Tokens": math_floor(sum(all_predicted_tokens) / len(predictions)),
    }
    word_info_preserved = score_calculator.word_info_preserved(data)
    score.update({k.replace("_", " ").capitalize(): v for k, v in score_calculator.edit_distance_distribution(data).items()})
    score["BLEU"] = score_calculator.bleu_corpus(data)
    score["GLEU"] = score_calculator.gleu_corpus(data)
    score["NIST"] = score_calculator.nist_corpus(data)
    score["METEOR"] = score_calculator.meteor_corpus(data)
    score["chrF"] = score_calculator.chrf_corpus(data)
    score["Edit Error"] = score_calculator.standard_edit_error(data)
    score["BLEU Error"] = score_calculator.standard_blue_error(data)
    score["METEOR Error"] = score_calculator.standard_blue_error(data)
    score["MER"] = score_calculator.match_error_rate(data)
    score["WER"] = score_calculator.word_error_rate(data)
    score["WIP"] = word_info_preserved
    score["WIL"] = round(100 - word_info_preserved, 4)
    return score


def compute_score(records: dict):
    size = records["size"]
    source = records["source"]
    target = records["target"]
    partitioned = records["partitioned"]
    simplified = records["simplified_tokens"]
    result = [__compute_score__(records["syn_results"], size, partitioned, simplified, source, target, "Synthetic"),
              __compute_score__(records["real_results"], size, partitioned, simplified, source, target, "Real")]
    if records.get("tuned_results", None) is not None:
        result.append(
            __compute_score__(records["tuned_results"], size, partitioned, simplified, source, target, "Tuned"))
    garbage_collect()
    return result


def compute_error_distribution(arguments: tuple[dict, str] | list[dict, str]):
    records, evaluation_column = arguments
    result = {"Size": records["size"], "Source": records["source"], "Target": records["target"],
              "Partitioned": records["partitioned"], "Simplified": records["simplified_tokens"],
              "Evaluation Type": evaluation_column.split("_")[0].capitalize(), "Total": len(records[evaluation_column])}

    edit_dist = [(Scores.edit_distance(x["target_tokens"], x["b1"]), len(x["target_tokens"])) for x in
                 records[evaluation_column]]
    distribution = {
        "0": [e[1] for e in edit_dist if e[0] == 0],
        "1": [e[1] for e in edit_dist if e[0] == 1],
        "2-5": [e[1] for e in edit_dist if 2 <= e[0] < 5],
        "6-10": [e[1] for e in edit_dist if 5 <= e[0] <= 10],
        "11+": [e[1] for e in edit_dist if e[0] >= 11],
    }

    result["Average Expected Length"] = math_floor(sum([e[1] for e in edit_dist]) / len(edit_dist))
    result["Minimum Expected Length"] = min([e[1] for e in edit_dist])
    result["Maximum Expected Length"] = max([e[1] for e in edit_dist])

    for k, v in distribution.items():
        result[f"{k} Errors Proportion"] = round(len(v) / len(records[evaluation_column]) * 100, 2)
    for k, v in distribution.items():
        result[f"{k} Errors"] = sorted(set(v))
    return result
