# Minimal implementation based on scikit-learn LabelEncoder, now with support for mixed sequences of integers and floats
# https://raw.githubusercontent.com/scikit-learn/scikit-learn/6a0838c416c7c2a6ee7fe4562cd34ae133674b2e/sklearn/preprocessing/_label.py

import numpy

from typing import Any
from math import isnan as math_isnan


class LabelEncoder:
    """Encode target labels with values between 0 and n_classes - 1."""

    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None
        self._nan_marker = object()

    @property
    def class_to_index(self) -> dict:
        return self.class_to_index_

    @property
    def classes(self) -> numpy.ndarray:
        return self.classes_

    def is_fitted(self) -> bool:
        return self.classes_ is not None

    def fit(self, y):
        """
        Fit the label encoder to the provided labels.

        :param y: List-like of shape (n_samples,) with labels to fit.
        :note: If y is a numeric NumPy array, mixed ints/floats may already be
               coerced to floats before encoding (e.g., array([1, 1.0]) -> float64).
        :return: self
        """
        y = numpy.array(y, dtype=object).ravel()
        seen = dict()
        for val in y:
            seen.setdefault(self.__normalize_label__(val), val)
        self.classes_ = numpy.array(list(seen.values()), dtype=object)
        self.class_to_index_ = dict()

        for idx, cls in enumerate(self.classes_):
            key = self.__normalize_label__(cls)
            self.class_to_index_[key] = idx

        return self

    def __normalize_label__(self, value: Any) -> Any:
        if isinstance(value, numpy.generic):
            value = value.item()
        if isinstance(value, float):
            is_nan = math_isnan(value)
        else:
            try:
                is_nan = bool(numpy.isnan(value))
            except (TypeError, ValueError):
                is_nan = False
        return self._nan_marker if is_nan else (type(value), value)

    def transform(self, y):
        """
        Transform labels to normalized integer encoding.

        :param y: List-like of shape (n_samples,) with labels to encode.
        :return: List of integers representing encoded labels.
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder instance is not fitted yet. Call 'fit' first.")

        y = numpy.array(y, dtype=object).ravel()
        encoded = []
        for label in y:
            key = self.__normalize_label__(label)
            try:
                encoded.append(self.class_to_index_[key])
            except KeyError as e:
                raise ValueError(f"y contains new labels: {label!r}\n{e}") from None
        return encoded

    def fit_transform(self, y):
        """
        Fit label encoder and return encoded labels.

        :param y: List-like of shape (n_samples,) with labels to fit and transform.
        :return: List of integers representing encoded labels.
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform normalized encoding back to original labels.

        :param y: List-like of integers representing encoded labels.
        :return: List of original label values.
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder instance is not fitted yet. Call 'fit' first.")

        y = numpy.array(y, dtype=int)
        if numpy.any((y < 0) | (y >= len(self.classes_))):
            raise ValueError("y contains invalid class indices.")

        return [self.classes_[index] for index in y]
