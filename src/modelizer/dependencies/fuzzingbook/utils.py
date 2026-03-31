import re

from copy import deepcopy
from time import perf_counter
from types import TracebackType
from functools import lru_cache
from collections import defaultdict
from sys import stderr as sys_stderr
from copy import deepcopy as copy_deepcopy
from typing import Any, Type, Dict, List, Tuple, Union, Optional, Callable, TypeAlias


########################################################################################################################
#                                                   Timer                                                              #
########################################################################################################################
class Timer:
    def __init__(self) -> None:
        self.start_time = perf_counter()
        self.end_time = None

    def __enter__(self) -> Any:
        self.start_time = perf_counter()
        self.end_time = None
        return self

    def __exit__(self, exc_type: Type, exc_value: BaseException, tb: TracebackType) -> None:
        self.end_time = perf_counter()

    def elapsed_time(self) -> float:
        if self.end_time is None:
            return perf_counter() - self.start_time
        else:
            return self.end_time - self.start_time  # type: ignore


def timeit(func: Callable, args: Optional[tuple | list] = (), kwargs: Optional[dict] = None) -> float:
    """
    Time the execution of a function with given arguments.
    :param func: Function to be timed
    :param args: Arguments to be passed to the function
    :param kwargs: Keyword arguments to be passed to the function
    :return: Elapsed time in seconds as a float
    """
    if kwargs is None:
        kwargs = dict()
    if not isinstance(args, tuple | list):
        raise TypeError(f"args must be a tuple or list, got {type(args)}")
    with Timer() as timer:
        func(*args, **kwargs)
    return timer.elapsed_time()


########################################################################################################################
#                                                   Type Aliases                                                       #
########################################################################################################################
Option = Dict[str, Any]
Expansion = Union[str, Tuple[str, Option]]
Grammar = Dict[str, List[Expansion]]
ExpansionKey: TypeAlias = str
DerivationTree = Tuple[str, Optional[List["DerivationTree"]]]
CanonicalGrammar: TypeAlias = dict[str, list[list[str]]]

"""Derivation Tree Representation: (SYMBOL, [CHILDREN])
The tree is recursively constructed, hence each node in the tree, is a derivation tree itself.
possible values for [CHILDREN]:
    I. `None` -> Placeholder for future expansion, meaning that the derivation tree, at current node 
                 is not expanded yet and SYMBOL is a nonterminal.(current node is a candidate for expansion)
    II. []    -> Indicates that the current node is fully expanded and SYMBOL is a terminal.
    III. List[DerivationTree] -> List of children nodes, each representing a derivation tree.
"""

########################################################################################################################
#                                                   Constants                                                          #
########################################################################################################################

SEED = 42
EPSILON = ''
START_SYMBOL = "<start>"
MIN_NONTERMINALS = 0
MAX_NONTERMINALS = 10
RE_NONTERMINAL = re.compile(r'(<[^<> ]*>)')
RE_PARENTHESIZED_EXPR = re.compile(r'\([^()]*\)[?+*]')
RE_EXTENDED_NONTERMINAL = re.compile(r'(<[^<> ]*>[?+*])')


########################################################################################################################
#                                       Grammar Fuzzing Utility Functions                                              #
########################################################################################################################
def is_valid_probabilistic_grammar(grammar: Grammar, start_symbol: str = START_SYMBOL) -> bool:
    if not is_valid_grammar(grammar, start_symbol):
        return False

    for nonterminal in grammar:
        expansions = grammar[nonterminal]
        _ = exp_probabilities(expansions, nonterminal)

    return True


def is_derivation_tree(obj) -> bool:
    if not isinstance(obj, tuple) or len(obj) != 2 or not isinstance(obj[0], str):
        return False

    stack = [obj[1]]  # start with the children list (or None)

    while stack:
        current = stack.pop()
        if current is None:
            continue
        if not isinstance(current, list):
            return False
        for child in current:
            if not isinstance(child, tuple) or len(child) != 2 or not isinstance(child[0], str):
                return False
            stack.append(child[1])
    return True


def count_nonterminals(tree: DerivationTree) -> int:
    """
    Return the total number of non-terminal nodes in the given derivation tree.
    A node is counted if its symbol matches a non-terminal (e.g., `<...>`),
    regardless of whether it is expanded (children list) or not (children is None).
    """
    count = 0
    stack: list[DerivationTree] = [tree]
    while stack:
        symbol, children = stack.pop()
        if is_nonterminal(symbol):
            count += 1
        if children:
            stack.extend(children)
    return count


def exp_string(expansion: Expansion) -> str:
    """Extract the string to be expanded from `expansion`"""
    return expansion if isinstance(expansion, str) else expansion[0]


def exp_option(expansion: Expansion, option) -> Any:
    """Extract `option` attribute of an expansion. Return `None` If `option` is not defined."""
    return exp_options(expansion).get(option, None)


def exp_options(expansion: Expansion) -> Option:
    """Extract all specified options of an expansion and return in a Dictionary"""
    return {} if isinstance(expansion, str) else expansion[1]


def exp_probability(expansion: Expansion) -> float | None:
    """Extract probability attribute of an expansion, None if probability is not defined"""
    return exp_option(expansion, 'prob')


def exp_probabilities(expansions: list[Expansion], nonterminal: str = "<symbol>") -> dict[str, float]:
    """Given a list of expansions from a single grammar rule, Extract and Normalize
       probabilities of each expansion and return in the form of a mapping SYMBOL_NAME: PROBABILITY"""
    probabilities = [exp_probability(exp) for exp in expansions]
    normalized_probabilities = prob_distribution(probabilities, nonterminal)

    prob_mapping: dict[str, float] = {}

    for i in range(len(expansions)):
        expansion = exp_string(expansions[i])
        prob_mapping[expansion] = normalized_probabilities[i]

    return prob_mapping


def prob_distribution(probabilities: list[float | None], nonterminal: str = "<symbol>") -> list[float]:
    """Normalize the list of probabilities and Replace None values with calculated probabilities"""
    epsilon = 0.00001
    number_of_nones = probabilities.count(None)  # Number of unspecified probabilities
    specified_probabilities_sum = sum(p for p in probabilities if p is not None)  # Sum of specified probabilities

    assert 0.0 <= specified_probabilities_sum <= 1.0, f"{nonterminal}: Sum of specified probabilities must be between 0.0 and 1.0"

    none_value = (1.0 - specified_probabilities_sum) / number_of_nones if number_of_nones > 0 else 0.0
    normalized_probabilities = [none_value if p is None else p for p in probabilities]

    assert all(0.0 <= p <= 1.0 for p in
               normalized_probabilities), f"{nonterminal}: Each symbol probability must be between 0.0 and 1.0"
    assert abs(
        sum(normalized_probabilities) - 1.0) < epsilon, f"{nonterminal}: Sum of normalized probabilities is not equal to 1.0"

    return normalized_probabilities


def convert_and_validate_ebnf_grammar(ebnf_grammar: Grammar) -> Grammar:
    converted_grammar = convert_ebnf_operators(convert_ebnf_parentheses(ebnf_grammar))
    if "{'prob':" in str(converted_grammar):
        assert is_valid_probabilistic_grammar(converted_grammar), "Invalid Probabilistic grammar"
    else:
        assert is_valid_grammar(converted_grammar), "Invalid Grammar"
    return converted_grammar


def convert_ebnf_operators(ebnf_grammar: Grammar) -> Grammar:
    """Convert a grammar in extended BNF to BNF"""
    grammar = extend_grammar(ebnf_grammar)
    for nonterminal in ebnf_grammar:
        expansions = ebnf_grammar[nonterminal]

        for i in range(len(expansions)):
            expansion = expansions[i]
            extended_symbols = extended_nonterminals(expansion)

            for extended_symbol in extended_symbols:
                operator = extended_symbol[-1:]
                original_symbol = extended_symbol[:-1]
                assert original_symbol in ebnf_grammar, f"{original_symbol} is not defined in grammar"

                new_sym = new_symbol(grammar, original_symbol)

                exp = grammar[nonterminal][i]
                opts_ = None
                if isinstance(exp, tuple):
                    (exp, opts_) = exp
                assert isinstance(exp, str)

                new_exp = exp.replace(extended_symbol, new_sym, 1)
                grammar[nonterminal][i] = (new_exp, opts_) if opts_ else new_exp

                if operator == '?':
                    grammar[new_sym] = ["", original_symbol]
                elif operator == '*':
                    grammar[new_sym] = ["", original_symbol + new_sym]
                elif operator == '+':
                    grammar[new_sym] = [original_symbol, original_symbol + new_sym]

    return grammar


def new_symbol(grammar: Grammar, symbol_name: str = "<symbol>") -> str:
    """Return a new symbol for `grammar` based on `symbol_name`"""
    if symbol_name not in grammar:
        return symbol_name

    count = 1
    while True:
        tentative_symbol_name = symbol_name[:-1] + "-" + repr(count) + ">"
        if tentative_symbol_name not in grammar:
            return tentative_symbol_name
        count += 1


def extended_nonterminals(expansion: Expansion) -> list[str]:
    # In later chapters, we allow expansions to be tuples,
    # with the expansion being the first element
    if isinstance(expansion, tuple):
        expansion = expansion[0]

    return re.findall(RE_EXTENDED_NONTERMINAL, expansion)


def parenthesized_expressions(expansion: Expansion) -> list[str]:
    # we allow expansions to be tuples, with the expansion being the first element
    if isinstance(expansion, tuple):
        expansion = expansion[0]

    return re.findall(RE_PARENTHESIZED_EXPR, expansion)


def convert_ebnf_parentheses(ebnf_grammar: Grammar) -> Grammar:
    """Convert a grammar in extended BNF to BNF"""
    grammar = extend_grammar(ebnf_grammar)
    for nonterminal in ebnf_grammar:
        expansions = ebnf_grammar[nonterminal]

        for i in range(len(expansions)):
            expansion = expansions[i]
            if not isinstance(expansion, str):
                expansion = expansion[0]

            while True:
                parenthesized_exprs = parenthesized_expressions(expansion)

                if len(parenthesized_exprs) == 0:
                    break

                for expr in parenthesized_exprs:
                    operator = expr[-1:]
                    contents = expr[1:-2]
                    new_sym = new_symbol(grammar)
                    exp = grammar[nonterminal][i]
                    opts_ = None
                    if isinstance(exp, tuple):
                        (exp, opts_) = exp
                    assert isinstance(exp, str)
                    expansion = exp.replace(expr, new_sym + operator, 1)
                    grammar[nonterminal][i] = (expansion, opts_) if opts_ else expansion
                    grammar[new_sym] = [contents]

    return grammar


def extend_grammar(grammar: Grammar, extension=None) -> Grammar:
    """Create a copy of `grammar`, updated with `extension`."""
    extension = extension if extension is not None else {}
    new_grammar = copy_deepcopy(grammar)
    new_grammar.update(extension)
    return new_grammar


def opts(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def all_terminals(tree: DerivationTree) -> str:
    symbol, children = tree
    if children is None:
        # This is a nonterminal symbol not expanded yet
        return symbol
    elif len(children) == 0:
        # This is a terminal symbol
        return symbol

    # This is an expanded symbol:
    # Concatenate all terminal symbols from all children
    return ''.join(all_terminals(c) for c in children)


def nonterminals(expansion: Expansion):
    # we allow expansions to be tuples, with the expansion being the first element
    if isinstance(expansion, tuple):
        expansion = expansion[0]

    return RE_NONTERMINAL.findall(expansion)


@lru_cache(maxsize=8192)
def _cached_tree_to_string(tree_tuple):
    """Cached version of tree_to_string for immutable tree tuples"""
    symbol, children_tuple = tree_tuple
    if children_tuple:
        return ''.join(_cached_tree_to_string(c) for c in children_tuple)
    else:
        return '' if is_nonterminal(symbol) else symbol


def tree_to_string(tree: DerivationTree) -> str:
    """Optimized tree_to_string using caching"""
    def make_hashable(node):
        symbol, children = node
        if children is None:
            return symbol, None
        return symbol, tuple(make_hashable(c) for c in children)

    return _cached_tree_to_string(make_hashable(tree))


def is_nonterminal(symbol: str):
    return RE_NONTERMINAL.match(symbol)


def has_nonterminals(expansion: Expansion) -> bool:
    return bool(RE_NONTERMINAL.search(expansion))


def rewrite_mixed_grammar_rules(grammar: Grammar) -> dict[str, list[str]]:
    """
    For each rule that mixes terminal-only and non-terminal-containing expansions,
    move the terminal-only expansions into a fresh helper non-terminal.
    Returns a new updated grammar.
    """
    new_grammar: Grammar = deepcopy(grammar)

    # Iterate over snapshot of original rules only
    for symbol, expansions in list(new_grammar.items()):
        term_only: List[Union[str, Tuple[str, dict[str, Any]]]] = []
        with_nonterm: List[Union[str, Tuple[str, dict[str, Any]]]] = []

        for exp in expansions:
            exp_str = exp_string(exp)
            (with_nonterm if has_nonterminals(exp_str) else term_only).append(exp)

        if term_only and with_nonterm:
            # Generate unique helper symbol
            base = symbol[1:-1]
            idx = 1
            while True:
                helper_sym = f"<{base}_opt_{idx}>"
                if helper_sym not in new_grammar:
                    break
                idx += 1

            # Rebuild original rule preserving order, inserting helper once
            reordered: List[Union[str, Tuple[str, dict[str, Any]]]] = []
            helper_inserted = False
            for exp in expansions:
                if exp in term_only:
                    if not helper_inserted:
                        reordered.append(helper_sym)
                        helper_inserted = True
                else:
                    reordered.append(exp)

            new_grammar[symbol] = reordered
            new_grammar[helper_sym] = term_only

    return new_grammar


def find_root_node(grammar: Grammar) -> str:
    """
    Detect and return the root node (<start>) non-terminal of a CFG.
    Raises ValueError if no unique root can be determined.
    """

    used = set()
    defined = set(grammar.keys())

    for expansions in grammar.values():
        for expansion in expansions:
            if isinstance(expansion, tuple):
                expansion = expansion[0]
            for nt in RE_NONTERMINAL.findall(expansion):
                used.add(nt)

    candidates = defined - used
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ValueError(f"Cannot determine unique root. Candidates: {sorted(candidates)}")


def get_nonterminal_rules(grammar: dict[str, list[str]]) -> Grammar:
    """Return a dict of only those rules having at least one expansion that contains a non-terminal expansion."""
    return {sym: exps for sym, exps in grammar.items() if any(has_nonterminals(e) for e in exps)}


def get_terminal_rules(grammar: dict[str, list[str]]) -> Grammar:
    """Return rules whose every expansion is terminal-only symbols. Empty string '' (epsilon) counts as terminal."""
    return {sym: exps for sym, exps in grammar.items() if all(not has_nonterminals(e) for e in exps)}


def opts_used(grammar: Grammar) -> set[str]:
    used_opts = set()
    for symbol in grammar:
        for expansion in grammar[symbol]:
            used_opts |= set(exp_options(expansion).keys())
    return used_opts


def def_used_nonterminals(grammar: Grammar, start_symbol: str = START_SYMBOL) -> tuple[Optional[set[str]], Optional[set[str]]]:
    """Return a pair (`defined_nonterminals`, `used_nonterminals`) in `grammar`. In case of error, return (`None`, `None`)."""
    defined_nonterminals = set()
    used_nonterminals = {start_symbol}

    for defined_nonterminal in grammar:
        defined_nonterminals.add(defined_nonterminal)
        expansions = grammar[defined_nonterminal]
        if not isinstance(expansions, list):
            print(f"{repr(defined_nonterminal)}: expansion is not a list", file=sys_stderr)
            return None, None
        elif len(expansions) == 0:
            print(f"{repr(defined_nonterminal)}: expansion list empty", file=sys_stderr)
            return None, None

        for expansion in expansions:
            if isinstance(expansion, tuple):
                expansion = expansion[0]
            elif not isinstance(expansion, str):
                print(f"{repr(defined_nonterminal)}: {repr(expansion)}: not a string", file=sys_stderr)
                return None, None

            for used_nonterminal in nonterminals(expansion):
                used_nonterminals.add(used_nonterminal)

    return defined_nonterminals, used_nonterminals


def reachable_nonterminals(grammar: Grammar, start_symbol: str = START_SYMBOL) -> set[str]:
    reachable = set()

    def _find_reachable_nonterminals(grammar_, symbol):
        nonlocal reachable
        reachable.add(symbol)
        for expansion in grammar_.get(symbol, []):
            for nonterminal in nonterminals(expansion):
                if nonterminal not in reachable:
                    _find_reachable_nonterminals(grammar_, nonterminal)

    _find_reachable_nonterminals(grammar, start_symbol)
    return reachable


def unreachable_nonterminals(grammar: Grammar, start_symbol=START_SYMBOL) -> set[str]:
    return grammar.keys() - reachable_nonterminals(grammar, start_symbol)


def is_valid_grammar(grammar: Grammar, start_symbol: str = START_SYMBOL, supported_opts=None) -> bool:
    """
    Check if the given `grammar` is valid.
    :param grammar: the grammar to check for validity
    :param start_symbol: optional start symbol (default: `<start>`)
    :param supported_opts: options supported (default: none)
    return: True if the grammar is valid, False otherwise
    """

    if supported_opts is None:
        supported_opts = set()

    defined_nonterminals, used_nonterminals = def_used_nonterminals(grammar, start_symbol)
    if defined_nonterminals is None or used_nonterminals is None:
        return False
    elif START_SYMBOL in grammar:
        # Do not complain about '<start>' being not used, even if start_symbol is different
        used_nonterminals.add(START_SYMBOL)

    for unused_nonterminal in defined_nonterminals - used_nonterminals:
        print(f"{repr(unused_nonterminal)}: defined, but not used. Consider applying trim_grammar() on the grammar", file=sys_stderr)

    for undefined_nonterminal in used_nonterminals - defined_nonterminals:
        print(f"{repr(undefined_nonterminal)}: used, but not defined", file=sys_stderr)

    # Symbols must be reachable either from <start> or given start symbol
    unreachable = unreachable_nonterminals(grammar, start_symbol)
    msg_start_symbol = start_symbol

    if START_SYMBOL in grammar:
        unreachable = unreachable - reachable_nonterminals(grammar, START_SYMBOL)
        if start_symbol != START_SYMBOL:
            msg_start_symbol += " or " + START_SYMBOL

    for unreachable_nonterminal in unreachable:
        print(f"{repr(unreachable_nonterminal)}: unreachable from {msg_start_symbol}. Consider applying trim_grammar() on the grammar", file=sys_stderr)

    if len(supported_opts) > 0:
        used_but_not_supported_opts = opts_used(grammar).difference(supported_opts)
        for opt in used_but_not_supported_opts:
            print(f"warning: option {repr(opt)} is not supported", file=sys_stderr)

    return start_symbol in grammar and used_nonterminals == defined_nonterminals and len(unreachable) == 0


def expansion_to_children(expansion: Expansion) -> list[DerivationTree]:
    """Given an expansion, return a list of nodes, per symbol in the expansion"""

    expansion: str = exp_string(expansion)
    assert isinstance(expansion, str)

    if expansion == "":  # Special case: epsilon expansion
        return [("", [])]

    symbols = re.split(RE_NONTERMINAL, expansion)
    return [(symbol, None) if is_nonterminal(symbol) else (symbol, []) for symbol in symbols if len(symbol) > 0]


def expansion_to_symbol_list(expansion: Expansion) -> list[str]:
    """Break down a single `expansion` into a list of symbols"""
    if expansion == '':
        return ['']
    return [symbol for symbol in re.split(RE_NONTERMINAL, expansion) if symbol != '']


def expansion_key(nonterminal: str, expansion: Expansion) -> ExpansionKey:
    """Compute a string key for the given expansion in the format of: `NONTERMINAL -> EXPRESSION`."""
    assert is_nonterminal(nonterminal), f"{nonterminal} is not a nonterminal symbol"
    expansion = exp_string(expansion)

    assert isinstance(expansion, str)
    return f"{nonterminal} -> {expansion}"


########################################################################################################################
#                                                   Parsing Utils                                                      #
########################################################################################################################
def single_char_tokens(grammar):
    return {
        key: [
            [token if token in grammar else char
             for token in rule
             for char in (token if token not in grammar else [token])]
            for rule in grammar[key]
        ]
        for key in grammar
    }


@lru_cache(maxsize=8192)
def _split_nonterminal(s):
    return tuple(re.split(RE_NONTERMINAL, s))


def canonical(grammar):
    return {
        k: [
            [tok for tok in _split_nonterminal(exp[0] if isinstance(exp, tuple) else exp) if tok]
            for exp in alternatives
        ]
        for k, alternatives in grammar.items()
    }


def non_canonical(grammar):
    return {k: [''.join(symbols) for symbols in expansions] for k, expansions in grammar.items()}


class Item:
    __slots__ = ('name', 'expr', 'dot')

    def __init__(self, name, expr, dot):
        self.name = name
        self.expr = expr
        self.dot = dot

    def finished(self):
        return self.dot >= len(self.expr)

    def advance(self):
        return Item(self.name, self.expr, self.dot + 1)

    def at_dot(self):
        return self.expr[self.dot] if self.dot < len(self.expr) else None


class State(Item):
    __slots__ = ('s_col', 'e_col')

    def __init__(self, name, expr, dot, s_col, e_col=None):
        super().__init__(name, expr, dot)
        self.s_col = s_col
        self.e_col = e_col

    def __str__(self):
        s = self.s_col.index if self.s_col else -1
        e = self.e_col.index if self.e_col else -1
        expr = ' '.join(str(p) for p in [*self.expr[:self.dot], '|', *self.expr[self.dot:]])
        return f"{self.name} := {expr} ({s},{e})"

    def __hash__(self):
        return hash((self.name, self.expr, self.dot, self.s_col.index))

    def __eq__(self, other):
        return (
            isinstance(other, State) and
            (self.name, self.expr, self.dot, self.s_col.index) ==
            (other.name, other.expr, other.dot, other.s_col.index)
        )

    def advance(self):
        return State(self.name, self.expr, self.dot + 1, self.s_col)


class Column:
    __slots__ = ('index', 'letter', 'states', '_unique', 'waiting')

    def __init__(self, index, letter):
        self.index = index
        self.letter = letter
        self.states = []
        self._unique = {}
        self.waiting = defaultdict(list)

    def add(self, state):
        if state in self._unique:
            return self._unique[state]
        self._unique[state] = state
        self.states.append(state)
        state.e_col = self
        nxt = state.at_dot()
        if nxt is not None:
            self.waiting[nxt].append(state)
        return state

    def __str__(self):
        return "%s chart[%d]\n%s" % (
            self.letter,
            self.index,
            "\n".join(str(s) for s in self.states if s.finished())
        )


def fixpoint(f):
    def helper(arg):
        while True:
            arg2 = f(arg)
            if arg2 == arg:
                return arg
            arg = arg2
    return helper


def rules(grammar):
    return [(k, choice) for k, choices in grammar.items() for choice in choices]


def nullable_expr(expr, nullables):
    return all(token in nullables for token in expr)


def nullable(grammar):
    prods = rules(grammar)

    @fixpoint
    def nullable_(nulls):
        new_nulls = set(nulls)
        for A, expr in prods:
            if nullable_expr(expr, nulls):
                new_nulls.add(A)
        return new_nulls

    return nullable_({EPSILON})


def extract_terminals_from_derivation_tree(tree: DerivationTree | list[DerivationTree]) -> list[str]:
    """
    Extract terminal substrings from a derivation tree.

    A node is considered a concrete terminal fragment if:
    - its children list is empty (children == []), and
    - its symbol is not a non-terminal (does not match `<...>`).

    Nodes with children is None are unexpanded nonterminals and are ignored.
    """
    terminals: list[str] = []
    stack: list[DerivationTree] = [tree] if is_derivation_tree(tree) else tree.copy()

    while stack:
        symbol, children = stack.pop()

        if is_nonterminal(symbol):
            if children:
                for child in reversed(children):
                    stack.append(child)
            continue

        if children is None:
            continue

        if len(children) == 0:
            if symbol != "":
                terminals.append(symbol)
            continue

        for child in reversed(children):
            stack.append(child)

    return terminals
