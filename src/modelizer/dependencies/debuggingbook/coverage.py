# Extension to DebuggingBook's coverage tracing utilities.
# Author: Tural Mammadov

import sys
import inspect
from contextlib import AbstractContextManager
from typing import Any, Callable, Optional, Tuple, Union, List, Type


ExcludeRule = Union[
    Type[BaseException],
    Tuple[Type[BaseException], Optional[Union[str, "re.Pattern"]]],
    Callable[[Type[BaseException], BaseException], bool],
]


class _ExceptionView:
    __slots__ = ("original",)

    def __init__(self, exc: BaseException):
        self.original = exc

    def __str__(self) -> str:
        # Normalize to first argument if present, which unquotes KeyError messages.
        try:
            args = getattr(self.original, "args", ())
            if args and len(args) == 1:
                msg = args[0]
                return msg if isinstance(msg, str) else str(msg)
        except Exception:
            pass
        return str(self.original)

    def __repr__(self) -> str:
        return self.__str__()


class CoverageTracer(AbstractContextManager):
    def __init__(self,
                 obj: Type[Any],
                 include_dunder: bool = True,
                 exclude: Optional[List[ExcludeRule]] = None,
                 enlarge_scope: bool = False) -> None:
        """
        :param obj: An object (typically an instance of a class) whose methods will be traced.
        :param include_dunder: Whether to include "dunder" methods (methods with names starting and ending with double underscores) in the coverage tracking.
        :param exclude: A list of exclusion rules for exceptions.
        :param enlarge_scope: If True, tracks calls to external methods invoked within the target object's methods, as well as exceptions raised in those external calls.
        """

        assert isinstance(include_dunder, bool), "include_dunder must be a boolean"
        assert isinstance(enlarge_scope, bool), "enlarge_scope must be a boolean"

        self._codes = {}
        for name, fn in inspect.getmembers(obj, inspect.isfunction):
            if include_dunder or not name.startswith("__"):
                self._codes[fn.__code__] = f"{obj.__name__}.{name}"

        self.executed_methods = set()
        self.all_methods = sorted(self._codes.values())

        self._exclude = list(exclude or [])
        self._exceptions = {}
        self._prev = None

        self._enlarge_scope = enlarge_scope
        self._executed_external = set()
        self._scope_depth = 0

    def _should_exclude(self, exc_type: Type[BaseException], exc_value: BaseException) -> bool:
        message = str(exc_value)

        for rule in self._exclude:
            # rule = Exception subclass
            if isinstance(rule, type) and issubclass(rule, BaseException):
                if issubclass(exc_type, rule):
                    return True

            # rule = (Exception subclass, message|regex|None)
            elif isinstance(rule, tuple) and len(rule) == 2:
                exc_cls, msg_pat = rule
                if isinstance(exc_cls, type) and issubclass(exc_cls, BaseException):
                    if issubclass(exc_type, exc_cls):
                        # None => wildcard: exclude any message for this type
                        if msg_pat is None:
                            return True
                        # exact match
                        if isinstance(msg_pat, str):
                            if message == msg_pat:
                                return True
                        else:
                            # regex-like: has .search()
                            search = getattr(msg_pat, "search", None)
                            if callable(search) and search(message):
                                return True

            # rule = callable(exc_type, exc_value) -> bool
            elif callable(rule):
                try:
                    # Pass a normalized exception view to make message checks easy (e.g., KeyError).
                    if rule(exc_type, _ExceptionView(exc_value)):
                        return True
                except Exception:
                    pass

        return False

    @staticmethod
    def _name_for_frame(frame) -> str:
        co = frame.f_code
        module = frame.f_globals.get("__name__", "<unknown>")
        func = co.co_name
        try:
            if "self" in frame.f_locals:
                return f"{module}.{frame.f_locals['self'].__class__.__name__}.{func}"
            if "cls" in frame.f_locals and isinstance(frame.f_locals["cls"], type):
                return f"{module}.{frame.f_locals['cls'].__name__}.{func}"
        except Exception:
            pass
        return f"{module}.{func}"

    def _tracer(self, frame, event, arg):
        if event == "call":
            co = frame.f_code
            if co in self._codes:
                self.executed_methods.add(self._codes[co])
                if self._enlarge_scope:
                    self._scope_depth += 1
            elif self._enlarge_scope and self._scope_depth > 0:
                self._executed_external.add(self._name_for_frame(frame))
                self._scope_depth += 1

        elif event == "return":
            if self._enlarge_scope and self._scope_depth > 0:
                self._scope_depth -= 1

        elif event == "exception":
            exc_type, exc_value, tb = arg

            if self._should_exclude(exc_type, exc_value):
                return self._tracer

            origin_tb = getattr(exc_value, "__traceback__", None) or tb
            if origin_tb is not None:
                while origin_tb.tb_next is not None:
                    origin_tb = origin_tb.tb_next

                origin_co = origin_tb.tb_frame.f_code
                filename = origin_co.co_filename
                lineno = origin_tb.tb_lineno

                # Only record exceptions whose origin is inside target codes.
                if origin_co in self._codes:
                    key = (exc_type, str(exc_value), filename, lineno)
                    if key not in self._exceptions:
                        self._exceptions[key] = {
                            "type": exc_type.__name__,
                            "qualified_type": f"{exc_type.__module__}.{exc_type.__name__}",
                            "message": str(exc_value),
                            "method": self._codes.get(origin_co),
                            "location": f"{filename}:{lineno}",
                        }

        return self._tracer

    def __enter__(self):
        self._prev = sys.gettrace()
        sys.settrace(self._tracer)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.settrace(self._prev)

    def reset(self):
        self.executed_methods.clear()
        self._exceptions.clear()
        self._prev = None
        self._scope_depth = 0
        self._executed_external.clear()

    def covered_names(self):
        return set(self.executed_methods)

    def external_covered_names(self):
        return set(self._executed_external)

    def max_possible(self):
        return len(self.all_methods)

    def unique_exceptions(self):
        return list(self._exceptions.values())

    def unique_exceptions_count(self):
        return len(self._exceptions)

    def summary(self):
        covered = len(self.executed_methods)
        total = len(self.all_methods)
        percent = (covered / total * 100.0) if total else 100.0
        summary = {
            "covered_count": covered,
            "total_possible": total,
            "percent": percent,
            "executed_methods": sorted(self.executed_methods),
            "all_methods": list(self.all_methods),
            "unique_exceptions_count": self.unique_exceptions_count(),
            "unique_exceptions": self.unique_exceptions(),
        }

        if self._enlarge_scope:
            summary["executed_external_count"] = len(self._executed_external)
            summary["executed_external"] = sorted(self._executed_external)

        return summary


class FullCoverageTracer(CoverageTracer):
    def __init__(self,
                 obj: Type[Any],
                 include_dunder: bool = True,
                 exclude: Optional[List[ExcludeRule]] = None,
                 exhaustive: bool = False,
                 enlarge_scope: bool = False) -> None:
        """
        Extends CoverageTracer with:
        - Per-line coverage for methods of the target class.
        - Chronological trace of events (`call`, `line`, `return`, `exception`) inside target class methods.
        :param obj: An object (typically an instance of a class) whose methods will be traced.
        :param include_dunder: Whether to include "dunder" methods (methods with names starting and ending with double underscores) in the coverage tracking.
        :param exclude: A list of exclusion rules for exceptions.
        :param exhaustive: If True, captures instance and class attribute state when changes occur at each executed line within target methods.
        :param enlarge_scope: If True, tracks calls to external methods invoked within the target object's methods, as well as exceptions raised in those external calls.
        """

        super().__init__(obj, include_dunder=include_dunder, exclude=exclude, enlarge_scope=enlarge_scope)
        self._obj: Type[Any] = obj
        self._exhaustive = bool(exhaustive)

        self._line_hits_by_co = {}
        self._line_hits_external_by_co = {}
        self._trace_events = []
        self._seq = 0

        self._last_snapshots = {}
        self._attr_changes = []

        # Stable names for external code objects (avoid mutating code objects)
        self._co_names = {}

    def executed_lines(self) -> dict:
        """returns mapping: method name -> sorted list of executed lines (target only)"""
        out = {}
        for co, lines in self._line_hits_by_co.items():
            name = self._codes.get(co)
            if name:
                out[name] = sorted(lines)
        return out

    def executed_lines_all(self) -> dict:
        """returns mapping: name -> sorted list of executed lines (target + external when enlarged)"""
        out = self.executed_lines()
        for co, lines in self._line_hits_external_by_co.items():
            name = self._co_names.get(co) or f"{co.co_filename}:{co.co_firstlineno}:{co.co_name}"
            out[name] = sorted(lines)
        return out

    def full_trace(self):
        """returns chronological list of event dicts"""
        return list(self._trace_events)

    def attribute_changes(self):
        """returns chronological list of attribute change records"""
        return list(self._attr_changes)

    def reset(self):
        super().reset()
        self._line_hits_by_co.clear()
        self._line_hits_external_by_co.clear()
        self._trace_events.clear()
        self._attr_changes.clear()
        self._last_snapshots.clear()
        self._seq = 0
        self._co_names.clear()

    def _tracer(self, frame, event, arg):
        co = frame.f_code
        method_name = self._codes.get(co)
        pre_depth = self._scope_depth  # depth before base handling changes it

        # Decide if current event is within tracing interest
        in_target = method_name is not None
        in_scope = in_target or (self._enlarge_scope and pre_depth > 0)

        # Chronological trace for calls/lines/returns
        if event in ("call", "line", "return") and in_scope:
            self._seq += 1
            name = method_name if in_target else self._name_for_frame(frame)
            if not in_target:
                self._co_names[co] = name
            self._trace_events.append({
                "seq": self._seq,
                "event": event,
                "in_target": bool(in_target),
                "method": name,
                "function": co.co_name,
                "filename": co.co_filename,
                "lineno": frame.f_lineno if event != "call" else co.co_firstlineno,
            })

        # Per-line coverage (target + external when enlarged)
        if event == "line" and in_scope:
            lineno = frame.f_lineno
            if in_target:
                self._line_hits_by_co.setdefault(co, set()).add(lineno)
            else:
                self._line_hits_external_by_co.setdefault(co, set()).add(lineno)
                self._co_names.setdefault(co, self._name_for_frame(frame))

            # Exhaustive attribute change capture only for the target object
            if in_target and self._exhaustive:
                self_obj = frame.f_locals.get("self")
                if isinstance(self_obj, self._obj):
                    self._record_attr_changes(
                        obj=self_obj,
                        is_class_obj=False,
                        filename=co.co_filename,
                        lineno=lineno,
                        method_name=method_name
                    )

                # Class attribute changes \(`cls`\ in classmethods\)
                cls_obj = frame.f_locals.get("cls")
                if cls_obj is self._obj:
                    self._record_attr_changes(
                        obj=cls_obj,
                        is_class_obj=True,
                        filename=co.co_filename,
                        lineno=lineno,
                        method_name=method_name
                    )

        # Exception trace event with origin binding
        if event == "exception" and in_scope:
            exc_type, exc_value, tb = arg
            origin_tb = getattr(exc_value, "__traceback__", None) or tb
            if origin_tb is not None:
                while origin_tb.tb_next is not None:
                    origin_tb = origin_tb.tb_next
                origin_co = origin_tb.tb_frame.f_code
                origin_in_target = origin_co in self._codes
                name = self._codes.get(origin_co) if origin_in_target else self._name_for_frame(origin_tb.tb_frame)
                if not origin_in_target:
                    self._co_names.setdefault(origin_co, name)

                self._seq += 1
                self._trace_events.append({
                    "seq": self._seq,
                    "event": "exception",
                    "in_target": bool(origin_in_target),
                    "method": name,
                    "function": origin_co.co_name,
                    "filename": origin_co.co_filename,
                    "lineno": origin_tb.tb_lineno,
                    "type": exc_type.__name__,
                    "qualified_type": f"{exc_type.__module__}.{exc_type.__name__}",
                    "message": str(exc_value),
                })

        # Delegate to base to update coverage/scope depth (base no longer records external-origin exceptions)
        super()._tracer(frame, event, arg)
        return self._tracer

    def _record_attr_changes(self, obj, is_class_obj: bool, filename: str, lineno: int, method_name: str):
        oid = id(obj)
        current = self._snapshot_attrs(obj, is_class_obj=is_class_obj)
        prev = self._last_snapshots.get(oid)

        # Determine changes; treat the first snapshot as a change
        changes = {}

        if prev is None:
            # Initial state
            for k, v in current.items():
                changes[k] = {"old": None, "new": v}
        else:
            # Modified or added
            for k, v in current.items():
                if k not in prev or prev[k] != v:
                    changes[k] = {"old": prev.get(k), "new": v}
            # Deleted
            for k in prev.keys() - current.keys():
                changes[k] = {"old": prev[k], "new": None}

        if changes:
            self._seq += 1
            rec = {
                "seq": self._seq,
                "event": "attr_change",
                "is_class": bool(is_class_obj),
                "object_id": oid,
                "object_class": obj.__name__ if is_class_obj else obj.__class__.__name__,
                "method": method_name,
                "filename": filename,
                "lineno": lineno,
                "changes": changes,
                "state": dict(current),  # full current state snapshot
            }
            self._attr_changes.append(rec)

        self._last_snapshots[oid] = current

    @staticmethod
    def _snapshot_attrs(obj, is_class_obj: bool):
        """
        Return a snapshot mapping attr name -> safe repr(value).
        - For instances: include __dict__ and __slots__ values.
        - For classes: include non-callable entries from __dict__.
        """
        snap = {}

        def safe_repr(z):
            try:
                return repr(z)
            except Exception:
                try:
                    return f"<unrepr:{type(z).__name__} id={id(z)}>"
                except Exception:
                    return "<unrepr:?>"

        if is_class_obj:
            for k, v in getattr(obj, "__dict__", {}).items():
                # skip callables and dunder to reduce noise
                if callable(v):
                    continue
                snap[k] = safe_repr(v)
            return snap

        # instance: __dict__
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for k, v in d.items():
                snap[k] = safe_repr(v)

        # instance: __slots__ across MRO
        for cls in obj.__class__.__mro__:
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots or ():
                # ignore Python-internal slots commonly present
                if slot in ("__weakref__", "__dict__"):
                    continue
                try:
                    if hasattr(obj, slot):
                        v = getattr(obj, slot)
                        snap[slot] = safe_repr(v)
                except Exception:
                    # accessing a slot could raise; ignore safely
                    pass

        return snap
