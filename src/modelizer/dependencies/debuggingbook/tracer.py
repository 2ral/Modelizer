# Based on the `DebuggingBook` implementation by Andreas Zeller et al. - https://github.com/uds-se/debuggingbook/

import sys
import inspect
import warnings
import traceback

from types import FrameType, TracebackType, FunctionType
from typing import Any, Optional, Callable, Type, TextIO, cast


Location = tuple[Callable, int]


class StackInspector:
    """Provide functions to inspect the stack"""
    _generated_function_cache: dict[tuple[str, int], Callable] = {}

    def caller_frame(self) -> FrameType:
        """Return the frame of the caller."""
        # Walk up the call tree until we leave the current class
        frame = cast(FrameType, inspect.currentframe())
        while self.our_frame(frame):
            frame = cast(FrameType, frame.f_back)
        return frame

    def our_frame(self, frame: FrameType) -> bool:
        """Return true if `frame` is in the current (inspecting) class."""
        return isinstance(frame.f_locals.get('self'), self.__class__)

    def caller_globals(self) -> dict[str, Any]:
        """Return the globals() environment of the caller."""
        return self.caller_frame().f_globals

    def caller_locals(self) -> dict[str, Any]:
        """Return the locals() environment of the caller."""
        return self.caller_frame().f_locals

    def caller_location(self) -> Location:
        """Return the location (func, lineno) of the caller."""
        return self.caller_function(), self.caller_frame().f_lineno

    def search_frame(self, name: str, frame: Optional[FrameType] = None) -> tuple[Optional[FrameType], Optional[Callable]]:
        """Return a pair (`frame`, `item`) in which the function `name` is defined as `item`."""
        if frame is None:
            frame = self.caller_frame()
        while frame:
            item = None
            if name in frame.f_globals:
                item = frame.f_globals[name]
            if name in frame.f_locals:
                item = frame.f_locals[name]
            if item and callable(item):
                return frame, item
            frame = cast(FrameType, frame.f_back)
        return None, None

    def search_func(self, name: str, frame: Optional[FrameType] = None) -> Callable | None:
        """Search in callers for a definition of the function `name`"""
        frame, func = self.search_frame(name, frame)
        return func

    def create_function(self, frame: FrameType) -> Callable:
        """Create function for given frame"""
        name = frame.f_code.co_name
        cache_key = (name, frame.f_lineno)
        if cache_key in self._generated_function_cache:
            return self._generated_function_cache[cache_key]
        try:
            # Create a new function from given code
            generated_function = cast(Callable, FunctionType(frame.f_code, globals=frame.f_globals, name=name))
        except TypeError:
            # Unsuitable code for creating a function -> Last resort: Return some function
            generated_function = self.unknown

        except Exception as exc:
            # Any other exception
            warnings.warn(f"Couldn't create function for {name} ({type(exc).__name__}: {exc})")
            generated_function = self.unknown

        self._generated_function_cache[cache_key] = generated_function
        return generated_function

    def caller_function(self) -> Callable:
        """Return the calling function"""
        frame = self.caller_frame()
        name = frame.f_code.co_name
        func = self.search_func(name)
        if func:
            return func

        if not name.startswith('<'):
            warnings.warn(f"Couldn't find {name} in caller")

        return self.create_function(frame)

    def unknown(self) -> None:
        """Placeholder for unknown functions"""
        pass

    def is_internal_error(self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType) -> bool:
        """Return True if an exception was raised from `StackInspector` or a subclass."""
        if not exc_tp:
            return False
        for frame, lineno in traceback.walk_tb(exc_traceback):
            if self.our_frame(frame):
                return True
        return False


########################################################################################################################
#                                                 Tracer Class                                                         #
########################################################################################################################
class Tracer(StackInspector):
    """A class for tracing a piece of code. Use as `with Tracer(): block()`"""

    def __init__(self, *, file: TextIO = sys.stdout, tracing_mode="variables") -> None:
        """Trace a block of code and writing to log
        :param file: file to write logs to. Defaults to `sys.stdout`
        :param tracing_mode: one of `variables`, `code`, `debugger`. Defaults to `variables`
        """
        self.file = file
        tracing_mode = tracing_mode.lower()
        self.original_trace_function: Optional[Callable] = None
        match tracing_mode:
            case 'code':
                self.traceit = self.trace_code
            case 'debugger':
                self.traceit = self.print_debugger_status
            case 'delta':
                self.traceit = self.track_changed_vars
            case 'variables':
                self.traceit = self.trace_variables
            case _:
                raise ValueError("Unknown tracing mode")
        self.last_vars: dict[str, Any] = {}

    def __enter__(self) -> Any:
        """Called at begin of `with` block. Turn tracing on."""
        self.original_trace_function = sys.gettrace()
        sys.settrace(self._traceit)
        # This extra line also enables tracing for the current block
        # inspect.currentframe().f_back.f_trace = self._traceit
        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType) -> bool | None:
        """Called at end of `with` block. Turn tracing off. Return `None` if ok, not `None` if internal error."""
        sys.settrace(self.original_trace_function)
        # Note: we must return a non-True value here, such that we re-raise all exceptions
        # False -> internal error / None -> all fine
        return False if self.is_internal_error(exc_tp, exc_value, exc_traceback) else None

    def trace_variables(self, frame: FrameType, event: str, arg: Any) -> None:
        self.log(event, frame.f_lineno, frame.f_code.co_name, frame.f_locals, arg)

    def trace_code(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function. To be overridden in subclasses."""
        match event:
            case 'call':
                self.log(f"Calling {frame.f_code.co_name}()")
            case 'line':
                module = inspect.getmodule(frame.f_code)
                source = inspect.getsource(frame.f_code) if module is None else inspect.getsource(module)
                current_line = source.split('\n')[frame.f_lineno - 1]
                self.log(frame.f_lineno, current_line)
            case 'return':
                self.log(f"{frame.f_code.co_name}() returns {repr(arg)}")
            case _:
                pass

    def _traceit(self, frame: FrameType, event: str, arg: Any) -> Callable | None:
        """Internal tracing function."""
        if not self.our_frame(frame):
            self.traceit(frame, event, arg)
        return self._traceit

    def log(self, *objects: Any, sep: str = ' ', end: str = '\n', flush: bool = True) -> None:
        """ Like `print()`, but always sending to `file` given at initialization, and flushing by default."""
        print(*objects, sep=sep, end=end, file=self.file, flush=flush)

    def track_changed_vars(self, new_vars: dict[str, Any]) -> dict[str, Any]:
        """Track changed variables, based on `new_vars` observed."""
        changed = {}
        for var_name, var_value in new_vars.items():
            if var_name not in self.last_vars or self.last_vars[var_name] != var_value:
                changed[var_name] = var_value
        self.last_vars = new_vars.copy()
        return changed

    def print_debugger_status(self, frame: FrameType, event: str, arg: Any) -> None:
        """Show current source line and changed vars"""
        changes = self.track_changed_vars(frame.f_locals)
        changes_s = ", ".join([var + " = " + repr(changes[var]) for var in changes])

        if event == 'call':
            self.log("Calling " + frame.f_code.co_name + '(' + changes_s + ')')
        elif changes:
            self.log(' ' * 40, '#', changes_s)
        elif event == 'line':
            try:
                module = inspect.getmodule(frame.f_code)
                source = inspect.getsource(frame.f_code) if module is None else inspect.getsource(module)
                current_line = source.split('\n')[frame.f_lineno - 1]
            except OSError as err:
                self.log(f"{err.__class__.__name__}: {err}")
                current_line = ""
            self.log(repr(frame.f_lineno) + ' ' + current_line)
        elif event == 'return':
            self.log(frame.f_code.co_name + '()' + " returns " + repr(arg))
            self.last_vars = {}  # Delete 'last' variables


########################################################################################################################
#                                              Conditional Tracing                                                     #
########################################################################################################################
class ConditionalTracer(Tracer):
    def __init__(self, *, condition: Optional[str] = None, file: TextIO = sys.stdout) -> None:
        """Constructor. Trace all events for which `condition` (a Python expr) holds."""
        if condition is None:
            condition = 'False'
        assert isinstance(condition, str)
        self.condition: str = condition
        self.last_report: Optional[bool] = None
        super().__init__(file=file)

    @staticmethod
    def eval_in_context(expr: str, frame: FrameType) -> bool | None:
        frame.f_locals['function'] = frame.f_code.co_name
        frame.f_locals['line'] = frame.f_lineno
        try:
            cond = eval(expr, None, frame.f_locals)
        except NameError:  # (yet) undefined variable
            cond = None
        return cond

    def do_report(self, frame: FrameType, event: str, arg: Any) -> bool | None:
        return self.eval_in_context(self.condition, frame)

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        report = self.do_report(frame, event, arg)
        if report != self.last_report:
            if report:
                self.log("...")
            self.last_report = report
        if report:
            self.print_debugger_status(frame, event, arg)


########################################################################################################################
#                                              Event Tracer                                                            #
########################################################################################################################
class EventTracer(ConditionalTracer):
    """Log when a given event expression changes its value"""

    def __init__(self, *, condition: Optional[str] = None, events: list[str] | tuple[str] = (), file: TextIO = sys.stdout):
        """Constructor. `events` is a list of expressions to watch."""
        # Allow events to be either list or tuple instead of asserting only for list
        assert isinstance(events, (list, tuple))  # updated to accept both list and tuple
        self.events = events
        self.last_event_values: dict[str, Any] = {}
        super().__init__(file=file, condition=condition)

    def events_changed(self, events: list[str] | tuple[str], frame: FrameType) -> bool:
        """Return True if any of the observed `events` has changed"""
        change = False
        for event in events:
            value = self.eval_in_context(event, frame)
            if event not in self.last_event_values or value != self.last_event_values[event]:
                self.last_event_values[event] = value
                change = True
        return change

    def do_report(self, frame: FrameType, event: str, arg: Any) -> bool:
        """Return True if a line should be shown"""
        return self.eval_in_context(self.condition, frame) or self.events_changed(self.events, frame)


TRACER = Tracer()
TRACER_CODE = "TRACER.print_debugger_status(inspect.currentframe(), 'line', None); "


def insert_tracer(function: Callable,
                  tracer_code: str = TRACER_CODE,
                  breakpoints: list[int] | tuple[int] = (),
                  same_origin: bool = True) -> Callable:
    """Return a variant of `function` with tracing code `tracer_code` inserted at each line given by `breakpoints`."""

    source_lines, starting_line_number = inspect.getsourcelines(function)

    for given_line in sorted(breakpoints, reverse=True):
        # Set new source line
        relative_line = given_line - starting_line_number + 1
        inject_line = source_lines[relative_line - 1]
        indent = len(inject_line) - len(inject_line.lstrip())
        source_lines[relative_line - 1] = ' ' * indent + tracer_code + inject_line.lstrip()

    # Rename function
    new_function_name = function.__name__ + "_traced"
    source_lines[0] = source_lines[0].replace(function.__name__, new_function_name)
    new_def = "".join(source_lines)

    # We keep the original source and filename to ease debugging
    prefix = '\n' * starting_line_number    # Get line number right
    new_function_code = compile(prefix + new_def, function.__code__.co_filename, 'exec')
    if same_origin:
        exec(new_function_code, function.__globals__)
        new_function = function.__globals__[new_function_name]
    else:
        exec(new_function_code)
        new_function = eval(new_function_name)
    return new_function
