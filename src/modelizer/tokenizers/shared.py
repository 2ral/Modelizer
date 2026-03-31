from abc import ABC, abstractmethod
from typing import Optional, Iterable, Generic, TypeVar
from collections.abc import KeysView, ItemsView, ValuesView, MutableMapping, Iterator


K = TypeVar("K")
V = TypeVar("V")


class _EfficientVocabulary(dict, ABC):
    """
    Shared base for efficient vocabularies with an implicit identity mapping for a key range.
    Subclasses must implement `_is_identity_key(key)` and may override iteration/length.
    """

    def __init__(self, *, deleted=None):
        self._deleted_identity_keys: set = set(deleted or [])
        super().__init__()

    @abstractmethod
    def _is_identity_key(self, key) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError("_is_identity_key must be implemented by subclasses")

    def __contains__(self, key) -> bool:
        if self._is_identity_key(key):
            return key not in self._deleted_identity_keys
        return super().__contains__(key)

    def __getitem__(self, key):
        if self._is_identity_key(key):
            if key in self._deleted_identity_keys:
                raise KeyError(key)
            return key
        return super().__getitem__(key)

    def get(self, key, default=None):
        if self._is_identity_key(key):
            return default if key in self._deleted_identity_keys else key
        return super().get(key, default)

    def setdefault(self, key, default=None):
        if self._is_identity_key(key):
            if key in self._deleted_identity_keys:
                if default is None or default == key:
                    self._deleted_identity_keys.discard(key)
                    return key
                raise ValueError(f"Cannot setdefault non-identity value for {key}.")
            return key
        return super().setdefault(key, default)

    def __setitem__(self, key, value) -> None:
        if self._is_identity_key(key):
            if value != key:
                raise ValueError(f"Cannot override identity mapping for {key} (must map to itself).")
            self._deleted_identity_keys.discard(key)
            return None
        return super().__setitem__(key, value)

    def update(self, other=None, /, **kwargs) -> None:
        if other is not None:
            iterable = other.items() if hasattr(other, "items") else other
            for k, v in iterable:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def pop(self, key, default=None):
        if self._is_identity_key(key):
            if key in self._deleted_identity_keys:
                if default is not None:
                    return default
                raise KeyError(key)
            self._deleted_identity_keys.add(key)
            return key
        if default is None:
            return super().pop(key)
        return super().pop(key, default)

    def __delitem__(self, key) -> None:
        if self._is_identity_key(key):
            if key in self._deleted_identity_keys:
                raise KeyError(key)
            self._deleted_identity_keys.add(key)
        else:
            super().__delitem__(key)

    def popitem(self):
        if len(super().keys()) == 0:
            raise KeyError(f"{type(self).__name__} has no explicitly stored items to pop")
        return super().popitem()

    def clear(self) -> None:
        return super().clear()

    def __iter__(self):
        return super().__iter__()

    def keys(self):
        return super().keys()

    def items(self):
        return super().items()

    def values(self):
        return super().values()

    def __len__(self) -> int:
        return len(super().keys())

    def __or__(self, other):
        new = self.copy()
        new.update(other)
        return new

    def __ior__(self, other):
        self.update(other)
        return self


class _IntegerKeysView(KeysView):
    _mapping: "EfficientIntegerVocabulary"

    def __init__(self, mapping):
        super().__init__(mapping)

    def __iter__(self):
        for k in range(self._mapping.max_value + 1):
            if k not in self._mapping._deleted_identity_keys:
                yield k
        for k in dict.keys(self._mapping):
            if not self._mapping._is_identity_key(k):
                yield k

    def __len__(self):
        extras = sum(1 for k in dict.keys(self._mapping) if not self._mapping._is_identity_key(k))
        return self._mapping.max_value + 1 - len(self._mapping._deleted_identity_keys) + extras

    def __contains__(self, key):
        if self._mapping._is_identity_key(key):
            return key not in self._mapping._deleted_identity_keys
        return dict.__contains__(self._mapping, key)


class _IntegerItemsView(ItemsView):
    _mapping: "EfficientIntegerVocabulary"

    def __init__(self, mapping):
        super().__init__(mapping)

    def __iter__(self):
        for k in range(self._mapping.max_value + 1):
            if k not in self._mapping._deleted_identity_keys:
                yield k, k
        for k, v in dict.items(self._mapping):
            if not self._mapping._is_identity_key(k):
                yield k, v

    def __len__(self):
        extras = sum(1 for k in dict.keys(self._mapping) if not self._mapping._is_identity_key(k))
        return self._mapping.max_value + 1 - len(self._mapping._deleted_identity_keys) + extras

    def __contains__(self, item):
        try:
            key, value = item
        except (TypeError, ValueError):
            return False
        if self._mapping._is_identity_key(key):
            return key not in self._mapping._deleted_identity_keys and value == key
        return dict.items(self._mapping).__contains__((key, value))


class _IntegerValuesView(ValuesView):
    _mapping: "EfficientIntegerVocabulary"

    def __init__(self, mapping):
        super().__init__(mapping)

    def __iter__(self):
        for k in range(self._mapping.max_value + 1):
            if k not in self._mapping._deleted_identity_keys:
                yield k
        for k, v in dict.items(self._mapping):
            if not self._mapping._is_identity_key(k):
                yield v

    def __len__(self):
        extras = sum(1 for k in dict.keys(self._mapping) if not self._mapping._is_identity_key(k))
        return self._mapping.max_value + 1 - len(self._mapping._deleted_identity_keys) + extras

    def __contains__(self, value):
        for v in self:
            if v == value:
                return True
        return False


class EfficientIntegerVocabulary(_EfficientVocabulary):
    """
    Memory-efficient mapping for token->id where any non-negative int key <= max_value
    is treated as an implicit identity mapping (key -> key) without being stored.
    Only keys outside that range are stored explicitly.
    """

    def __init__(self, max_value: int, initial: Optional[dict] = None, deleted: Optional[Iterable[int]] = None):
        if not isinstance(max_value, int) or max_value < 0:
            raise ValueError("max_value must be a non-negative integer")
        self.max_value = max_value
        super().__init__(deleted=deleted)
        if initial:
            super().update(initial)

    def _is_identity_key(self, key) -> bool:
        return isinstance(key, int) and 0 <= key <= self.max_value

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return _IntegerKeysView(self)

    def items(self):
        return _IntegerItemsView(self)

    def values(self):
        return _IntegerValuesView(self)

    def __len__(self) -> int:
        extras = sum(1 for k in dict.keys(self) if not self._is_identity_key(k))
        return (self.max_value + 1 - len(self._deleted_identity_keys)) + extras

    def copy(self):
        return EfficientIntegerVocabulary(
            self.max_value,
            dict(super().items()),
            deleted=self._deleted_identity_keys.copy()
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.max_value, dict(super().items()), self._deleted_identity_keys.copy())
        )


class BiMap(MutableMapping[K, V], Generic[K, V]):
    """
    Bidirectional map:
    - Forward: behaves like a normal dict: key -> value
    - Reverse: extra helpers for value -> key

    When initialized with an EfficientIntegerVocabulary the huge implicit
    identity range is *not* materialized into the internal dicts. Instead,
    the vocabulary is retained as a lazy backing store and consulted on
    every lookup.  Only the explicitly stored extras (special tokens, etc.)
    are copied into ``_forward`` / ``_inverse``.  This keeps initialisation
    O(extras) rather than O(max_value) for large-bit-width tokenizers.
    """

    def __init__(self, initial: "dict[K, V] | EfficientIntegerVocabulary | None" = None):
        self._forward: dict[K, V] = {}
        self._inverse: dict[V, K] = {}
        # Lazy backing store – set only when constructed from EfficientIntegerVocabulary.
        self._vocab: "EfficientIntegerVocabulary | None" = None

        if initial is None:
            return

        if isinstance(initial, EfficientIntegerVocabulary):
            # Only materialise the small set of explicit extras (special tokens).
            self._vocab = initial
            for k in list(dict.keys(initial)):          # explicit extras only
                if not initial._is_identity_key(k):
                    v = dict.__getitem__(initial, k)
                    if v in self._inverse and self._inverse[v] != k:
                        raise ValueError(f"Value {v!r} already mapped to key {self._inverse[v]!r}")
                    self._forward[k] = v
                    self._inverse[v] = k
        else:
            for k, v in initial.items():
                self[k] = v

    def __getitem__(self, key: K) -> V:
        if key in self._forward:
            return self._forward[key]
        if self._vocab is not None and self._vocab._is_identity_key(key):
            if key not in self._vocab._deleted_identity_keys:
                return key  # type: ignore[return-value]
        raise KeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        # Handle existing key: clean up old inverse entry.
        if key in self:
            old_value = self[key]
            if old_value == value:
                return
            # Remove old inverse mapping from _inverse if it was stored there.
            if old_value in self._inverse:
                del self._inverse[old_value]
            # If old_value was an identity entry we need to delete it from the vocab.
            if (self._vocab is not None
                    and self._vocab._is_identity_key(old_value)
                    and old_value not in self._inverse):
                pass  # identity entries don't live in _inverse; nothing extra to do

        if value in self._inverse and self._inverse[value] != key:
            raise ValueError(f"Value {value!r} already mapped to key {self._inverse[value]!r}")
        # Also check the vocab's identity range for a conflicting reverse mapping.
        if (self._vocab is not None
                and self._vocab._is_identity_key(value)
                and value not in self._vocab._deleted_identity_keys
                and value not in self._inverse):
            # Identity says value->value; a new key->value mapping would conflict unless key==value.
            if key != value:
                raise ValueError(f"Value {value!r} already mapped to key {value!r} (identity entry)")

        self._forward[key] = value
        self._inverse[value] = key

        # Keep vocab's explicit dict in sync so vocab-based lookups stay correct.
        if self._vocab is not None:
            if not self._vocab._is_identity_key(key):
                dict.__setitem__(self._vocab, key, value)

    def __delitem__(self, key: K) -> None:
        if key not in self:
            raise KeyError(key)
        # Remove from _forward if it lives there.
        if key in self._forward:
            value = self._forward.pop(key)
            self._inverse.pop(value, None)
        elif self._vocab is not None and self._vocab._is_identity_key(key):
            # Mark the identity entry as deleted in the vocab.
            self._vocab._deleted_identity_keys.add(key)

    def __iter__(self) -> Iterator[K]:
        if self._vocab is None:
            return iter(self._forward)

        # Yield identity range (minus deletions), then explicit extras.
        def _iter():
            for k in range(self._vocab.max_value + 1):
                if k not in self._vocab._deleted_identity_keys:
                    yield k
            for k in self._forward:
                yield k

        return _iter()

    def __len__(self) -> int:
        if self._vocab is None:
            return len(self._forward)
        identity_count = self._vocab.max_value + 1 - len(self._vocab._deleted_identity_keys)
        total = identity_count + len(self._forward)
        import sys
        # len() must fit in a C ssize_t; clamp for huge lazy vocabs.
        return min(total, sys.maxsize)

    def __contains__(self, key: object) -> bool:
        if key in self._forward:
            return True
        if self._vocab is not None and self._vocab._is_identity_key(key):
            return key not in self._vocab._deleted_identity_keys
        return False

    def __getstate__(self):
        state: dict = {"_forward": self._forward}
        if self._vocab is not None:
            state["_vocab_max_value"] = self._vocab.max_value
            state["_vocab_deleted"] = self._vocab._deleted_identity_keys.copy()
        return state

    def __setstate__(self, state):
        self._forward = state["_forward"]
        if "_vocab_max_value" in state:
            self._vocab = EfficientIntegerVocabulary(state["_vocab_max_value"])
            self._vocab._deleted_identity_keys = state["_vocab_deleted"]
            # Re-populate the vocab's explicit dict from _forward extras.
            for k, v in self._forward.items():
                dict.__setitem__(self._vocab, k, v)
        else:
            self._vocab = None
        self._inverse = {}
        # Rebuild inverse from _forward extras.
        for k, v in self._forward.items():
            self._inverse[v] = k
        # For vocab-backed maps the identity values equal their keys so no
        # additional inverse entries are needed there.

    def get_key(self, value: V, default: "K | None" = None) -> "K | None":
        """Reverse lookup: value -> key, returns default if not found."""
        if value in self._inverse:
            return self._inverse[value]
        # For vocab-backed maps identity values map to themselves.
        if (self._vocab is not None
                and self._vocab._is_identity_key(value)
                and value not in self._vocab._deleted_identity_keys):
            return value  # type: ignore[return-value]
        return default

    def contains_value(self, value: V) -> bool:
        """Check if value is present in the mapping."""
        if value in self._inverse:
            return True
        if (self._vocab is not None
                and self._vocab._is_identity_key(value)
                and value not in self._vocab._deleted_identity_keys):
            return True
        return False

    @property
    def forward(self) -> dict[K, V]:
        """Get the forward mapping as a standard dictionary."""
        return self._forward.copy()

    @property
    def inverse(self) -> dict[V, K]:
        """Read-only view of the reverse mapping (value -> key, extras only)."""
        return self._inverse.copy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._forward!r})"
