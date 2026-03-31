import random

from typing import (
    List,
    Literal,
    Optional,
    Callable,
)

from enum import Enum
from abc import ABC, abstractmethod

import modelizer.dependencies.fuzzingbook.utils as utils


class ExpandNodeStrategy(Enum):
    """Indicates the Node expansion strategy in the derivation tree of a `GrammarFuzzer`."""
    RANDOM = "RANDOM"
    MIN_COST = "MIN_COST"
    MAX_COST = "MAX_COST"


class Fuzzer(ABC):
    """Base class for fuzzers."""

    @abstractmethod
    def fuzz(self) -> str:
        """
        This method should be overridden by subclasses to produce a fuzzed string.
        :return: A fuzzed string.
        """
        raise NotImplementedError


def delete_random_character(s: str) -> str:
    """Returns s with a random character deleted"""
    if s == "":
        return s

    pos = random.randint(0, len(s) - 1)
    return s[:pos] + s[pos + 1:]


def insert_random_character(s: str) -> str:
    """Returns s with a random character inserted"""
    pos = random.randint(0, len(s))
    return s[:pos] + chr(random.randrange(32, 127)) + s[pos:]


def flip_random_character(s: str):
    """Returns s with a random bit flipped in a random position"""
    if s == "":
        return s

    pos = random.randint(0, len(s) - 1)
    bit = 1 << random.randint(0, 6)
    return s[:pos] + chr(ord(s[pos]) ^ bit) + s[pos + 1:]


class MutationFuzzer(Fuzzer):
    """Base class for mutational fuzzing"""

    def __init__(self, seed: List[str],
                 min_mutations: int = 1,
                 max_mutations: int = 10):
        """Constructor.
        `seed` - a list of (input) strings to mutate.
        `min_mutations` - the minimum number of mutations to apply.
        `max_mutations` - the maximum number of mutations to apply.
        """
        self.seed = list(seed)
        self.seed_index = 0
        self.min_mutations = min_mutations
        self.max_mutations = max_mutations
        self.mutators = [
            delete_random_character,
            insert_random_character,
            flip_random_character,
        ]

    def reset(self) -> None:
        """Set population to initial seed.
        To be overloaded in subclasses."""
        self.seed_index = 0

    def fuzz(self) -> str:
        if self.seed_index < len(self.seed):
            candidate = self.seed[self.seed_index]
            self.seed_index += 1
        else:
            candidate = random.choice(self.seed)
            for i in range(random.randint(self.min_mutations, self.max_mutations)):
                candidate = random.choice(self.mutators)(candidate)
        return candidate


class GrammarFuzzer(Fuzzer):
    """Produce strings from `grammar` using derivation trees, starting with `start_symbol`. If `min_nonterminals`
       or `max_nonterminals` is given, use them as limits for the number of nonterminals produced."""

    def __init__(self,
                 grammar: utils.Grammar,
                 start_symbol: str = utils.START_SYMBOL,
                 min_nonterminals: int = utils.MIN_NONTERMINALS,
                 max_nonterminals: int = utils.MAX_NONTERMINALS,
                 seed=utils.SEED, **_):
        super().__init__()
        assert utils.is_valid_grammar(grammar, start_symbol=start_symbol)
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.min_nonterminals = min_nonterminals
        self.max_nonterminals = max_nonterminals

        self.derivation_tree = None
        self.expand_node: Callable = self.expand_node_randomly
        """Function pointer to the method used for expanding a node in the derivation tree.
           Changes dynamically upon assigning the value of `expand_node_strategy`"""
        self._expand_node_strategy = ExpandNodeStrategy.RANDOM
        """The strategy used to expand a node in the derivation tree."""

        self._random = random.Random(seed)

    @property
    def expand_node_strategy(self) -> ExpandNodeStrategy:
        return self._expand_node_strategy

    @expand_node_strategy.setter
    def expand_node_strategy(self, strategy: Literal['RANDOM', 'MIN_COST', 'MAX_COST']) -> None:
        self._expand_node_strategy = strategy
        match strategy:
            case 'RANDOM':
                self.expand_node = self.expand_node_randomly
            case 'MIN_COST' | 'MAX_COST':
                self.expand_node = self.expand_node_by_cost
            case _:
                raise ValueError(f"Unsupported strategy: {strategy}")

    def fuzz(self) -> str:
        """Produce a string from the grammar."""
        # Initialize the derivation tree with only the start symbol
        self.derivation_tree: utils.DerivationTree = (self.start_symbol, None)

        # Expand the derivation tree until all non-terminals are expanded
        self.derivation_tree = self.expand_tree(self.derivation_tree)
        return utils.all_terminals(self.derivation_tree)

    def expand_tree(self, tree: utils.DerivationTree, **kwargs) -> utils.DerivationTree:
        """Expand `tree` in 3 phases until all expansions are complete."""
        # Phase 1: Node Inflation, Expand all non-terminals using maximum cost strategy until
        # we have at least `min_nonterminals` nonterminals in the derivation tree
        self.expand_node_strategy = 'MAX_COST'
        tree = self.expand_tree_with_limit(tree, self.min_nonterminals, **kwargs)

        # Phase 2: Expand all non-terminals randomly until we have `max_nonterminals` nonterminals
        self.expand_node_strategy = 'RANDOM'
        tree = self.expand_tree_with_limit(tree, self.max_nonterminals, **kwargs)

        # Phase 3: Expand all non-terminals with minimum cost until no more expansion is possible
        self.expand_node_strategy = 'MIN_COST'
        tree = self.expand_tree_with_limit(tree, None, **kwargs)

        assert self.possible_expansions(tree) == 0
        return tree

    def expand_tree_with_limit(self, tree: utils.DerivationTree, limit: Optional[int], **kwargs) -> utils.DerivationTree:
        """Expand the tree until the number of possible expansions reaches `limit`.
           If `limit` was None, expand until no more expansions are possible."""

        while True:
            exp_count = self.possible_expansions(tree)
            if exp_count == 0 or (limit is not None and exp_count >= limit):
                break
            tree = self.expand_tree_once(tree, **kwargs)
        return tree

    def expand_tree_once(self, tree: utils.DerivationTree, **kwargs) -> utils.DerivationTree:
        """Perform one expansion iteration on the given tree. Mutates the `tree` argument."""
        assert self.any_possible_expansions(tree)  # There must be at least one possible expansion

        _, children = tree
        if children is None:  # The tree is expandable at the current node.
            return self.expand_node(tree)

        # Find all expandable children
        expandable_children_indices = [i for (i, c) in enumerate(children) if self.any_possible_expansions(c)]

        # Randomly pick one of the expandable children and expand it
        child_index = self._random.choice(expandable_children_indices)
        children[child_index] = self.expand_tree_once(children[child_index])
        return tree

    def expand_node_by_cost(self, node: utils.DerivationTree) -> utils.DerivationTree:
        symbol, children = node
        assert children is None, f"node {repr(node)}: already expanded"  # `node` must not be expanded yet
        cost_criterion: Callable = min if self.expand_node_strategy == 'MIN_COST' else max

        # Fetch all possible expansions of `symbol` from grammar
        expansions: list[utils.Expansion] = self.grammar[symbol]
        expansions_costs = [self.expansion_cost(exp, {symbol}) for exp in expansions]
        chosen_cost = cost_criterion(expansions_costs)

        # Find all expansions having the chosen cost
        expansions_with_chosen_cost = [exp for (exp, cost) in zip(expansions, expansions_costs) if cost == chosen_cost]

        chosen_expansion = self.choose_node_expansion(node, expansions_with_chosen_cost)
        children = utils.expansion_to_children(chosen_expansion)  # children of the current node after expanding

        return symbol, children

    def expand_node_randomly(self, node: utils.DerivationTree) -> utils.DerivationTree:
        """Expand the current `node` of the derivation tree by selecting one its expansions randomly."""
        symbol, children = node
        assert children is None, f"node {repr(node)}: already expanded"  # `node` must not be expanded yet

        # Fetch all possible expansions of `symbol` from grammar
        expansions: list[utils.Expansion] = self.grammar[symbol]
        chosen_expansion = self.choose_node_expansion(node, expansions)
        children = utils.expansion_to_children(chosen_expansion)  # children of the current node after expanding

        return symbol, children

    def choose_node_expansion(self, node: utils.DerivationTree, expansion_alternatives: list[utils.Expansion]) -> utils.Expansion:
        """Select one expansion from `expansion_alternatives` randomly.
           `expansion_alternatives`: a subset of all possible expansions for `node`
           Overload this method for custom selection strategy."""
        return self._random.choices(expansion_alternatives)[0]

    def possible_expansions(self, tree: utils.DerivationTree) -> int:
        """Count the number of possible expansions in the tree."""
        _, children = tree
        return 1 if children is None else sum(self.possible_expansions(c) for c in children)

    def any_possible_expansions(self, tree: utils.DerivationTree) -> bool:
        """Check if there is any possible expansion in the tree."""
        _, children = tree
        return True if children is None else any(self.any_possible_expansions(c) for c in children)

    def symbol_cost(self, symbol: str, seen: set[str] = None) -> int | float:
        """The minimum cost of expanding `symbol` among all of its possible expansions."""
        if seen is None:
            seen = set()
        expansions: list[utils.Expansion] = self.grammar[symbol]
        return min(self.expansion_cost(e, seen | {symbol}) for e in expansions)

    def expansion_cost(self, expansion: utils.Expansion, seen: set[str] = None) -> int | float:
        """Sum of all symbol_costs in `expansion`. Returns infinity if the nonterminal is
           encountered during traversal. (indicating a potential infinite loop)"""
        if seen is None:
            seen = set()
        symbols = utils.nonterminals(expansion)
        if len(symbols) == 0:
            return 1  # no symbol
        elif any(s in seen for s in symbols):
            return float('inf')
        else:
            # the value of an expansion is the sum of all expandable variables inside + 1
            return sum(self.symbol_cost(s, seen) for s in symbols) + 1


class ProbabilisticGrammarFuzzer(GrammarFuzzer):
    def __init__(self,
                 grammar: utils.Grammar,
                 start_symbol: str = utils.START_SYMBOL,
                 min_nonterminals: int = utils.MIN_NONTERMINALS,
                 max_nonterminals: int = utils.MAX_NONTERMINALS,
                 seed=utils.SEED):
        assert utils.is_valid_probabilistic_grammar(grammar, start_symbol=start_symbol)
        super().__init__(grammar, start_symbol, min_nonterminals, max_nonterminals, seed)

    def choose_node_expansion(self, node: utils.DerivationTree, expansion_alternatives: list[utils.Expansion]) -> utils.Expansion:
        symbol, _ = node

        # 1. Calculate the probabilities of all possible expansions
        all_expansions = self.grammar[symbol]
        probabilities = utils.exp_probabilities(all_expansions)

        # 2. Select the corresponding probabilities of the `expansion_alternatives`
        expansion_alternatives_weights = [probabilities[utils.exp_string(exp)] for exp in expansion_alternatives]

        # 3. Pick an expansion alternative according to the weights (probabilities)
        if sum(expansion_alternatives_weights) == 0:
            return random.choices(expansion_alternatives)[0]
        else:
            return random.choices(expansion_alternatives, weights=expansion_alternatives_weights)[0]


class ForcingProbabilisticGrammarFuzzer(GrammarFuzzer):
    def __init__(
            self,
            grammar: utils.Grammar,
            start_symbol: str = utils.START_SYMBOL,
            min_nonterminals: int = utils.MIN_NONTERMINALS,
            max_nonterminals: int = utils.MAX_NONTERMINALS,
            seed=utils.SEED,
            generation_attempts: int = 10,
            budget_conditioning: bool = False,
    ):
        """
        Create a probabilistic grammar fuzzer that always samples expansions according
        to rule probabilities. It grows the derivation up to the frontier budget
        (``max_nonterminals``) and then finishes expansion, both probabilistically.

        When ``budget_conditioning`` is enabled, the growth phase prunes expansions at leaves
        whose frontier delta would exceed the remaining budget and renormalizes the remaining
        probabilities before sampling. Otherwise, growth still samples probabilistically and
        stops as soon as the current frontier reaches the budget. The final output is accepted
        only if the number of nonterminals is within ``[min_nonterminals, max_nonterminals]``,
        up to ``generation_attempts`` retries.

        :param grammar: A valid probabilistic grammar. Each rule's expansions may carry a
                        ``'prob'`` weight; unspecified probabilities are normalized so the
                        rule's weights sum to 1.0.
        :param start_symbol: Start nonterminal of the grammar.
        :param min_nonterminals: Minimum allowed number of nonterminals in the final derivation.
                                 Enforced via rejection in ``fuzz()``.
        :param max_nonterminals: Target upper bound on the frontier (number of possible expansions)
                                 during the growth phase.
        :param seed: Seed for the internal random number generator.
        :param generation_attempts: Maximum number of attempts to produce a result within the
                                    ``[min_nonterminals, max_nonterminals]`` bounds.
        :param budget_conditioning: If ``True``, prune infeasible expansions during growth and
                                    renormalize probabilities before sampling; if ``False``,
                                    sample unconditionally and stop when the budget is reached.
        :raises AssertionError: If the provided grammar is not a valid probabilistic grammar.
        """
        assert utils.is_valid_probabilistic_grammar(grammar)
        super().__init__(grammar, start_symbol, min_nonterminals, max_nonterminals, seed)
        self.__generation_attempts__ = generation_attempts
        self.__budget_conditioning__ = budget_conditioning
        self.__fuzzer_invocations__ = 0
        self.__generation_rounds__ = 0
        self.__fuzzer_failures__ = 0

    @property
    def generation_attempts(self) -> int:
        return self.__generation_attempts__

    @generation_attempts.setter
    def generation_attempts(self, attempts: int) -> None:
        if not (isinstance(attempts, int) and attempts > 0):
            raise ValueError("generation_attempts must be a positive integer")
        else:
            self.__generation_attempts__ = attempts

    @property
    def budget_conditioning(self) -> bool:
        return self.__budget_conditioning__

    @budget_conditioning.setter
    def budget_conditioning(self, conditioning: bool) -> None:
        if not isinstance(conditioning, bool):
            raise ValueError("budget_conditioning must be a boolean value")
        else:
            self.__budget_conditioning__ = conditioning

    @property
    def failure_rate(self) -> float:
        if self.__fuzzer_invocations__ == 0:
            return 0.0
        return self.__fuzzer_failures__ / self.__fuzzer_invocations__

    @property
    def average_generation_rounds(self) -> float:
        if self.__fuzzer_invocations__ == 0:
            return 0.0
        return self.__generation_rounds__ / self.__fuzzer_invocations__

    def reset(self):
        self.__fuzzer_invocations__ = 0
        self.__generation_rounds__ = 0
        self.__fuzzer_failures__ = 0

    def fuzz(self) -> str:
        self.__fuzzer_invocations__ += 1
        self.derivation_tree = (self.start_symbol, None)
        for _ in range(self.__generation_attempts__):
            self.__generation_rounds__ += 1
            tree = self.expand_tree(self.derivation_tree)
            if self.min_nonterminals <= utils.count_nonterminals(tree) <= self.max_nonterminals:
                self.derivation_tree = tree
                return utils.all_terminals(self.derivation_tree)
        self.__fuzzer_failures__ += 1
        return ""

    def expand_tree(self, tree: utils.DerivationTree, **kwargs) -> utils.DerivationTree:
        # Phase 1: grow to maximum budget
        if self.__budget_conditioning__:
            tree = self.expand_tree_with_limit_bounded(tree, self.max_nonterminals, **kwargs)
        else:
            tree = self.expand_tree_with_limit(tree, self.max_nonterminals, **kwargs)
        # Phase 2: finish expansion
        tree = self.expand_tree_with_limit(tree, None, **kwargs)
        assert self.possible_expansions(tree) == 0
        return tree

    def expand_tree_with_limit(self, tree: utils.DerivationTree, limit: Optional[int], **kwargs) -> utils.DerivationTree:
        while True:
            exp_count = self.possible_expansions(tree)
            if exp_count == 0 or (limit is not None and exp_count >= limit):
                break
            tree = self.expand_tree_once(tree, **kwargs)
        return tree

    def expand_tree_with_limit_bounded(self, tree: utils.DerivationTree, limit: Optional[int], **kwargs) -> utils.DerivationTree:
        # Budget-conditioned probabilistic growth (prune infeasible, renormalize)
        while True:
            exp_count = self.possible_expansions(tree)
            if exp_count == 0:
                break
            if limit is not None and exp_count >= limit:
                break
            remaining = None if limit is None else (limit - exp_count)
            tree, expanded = self.expand_tree_once_bounded(tree, remaining, **kwargs)
            if not expanded:
                # No feasible expansion exists under remaining budget
                break
        return tree

    def expand_tree_once(self, tree: utils.DerivationTree, **kwargs) -> utils.DerivationTree:
        # One unbounded probabilistic step
        symbol, children = tree
        if children is None:
            return self.expand_node_randomly(tree)  # uses weighted chooser below
        expandable_children_indices = [i for (i, c) in enumerate(children) if self.any_possible_expansions(c)]
        child_index = self._random.choice(expandable_children_indices)
        children[child_index] = self.expand_tree_once(children[child_index], **kwargs)
        return symbol, children

    def expand_tree_once_bounded(self, tree: utils.DerivationTree, remaining: Optional[int], **kwargs) -> tuple[utils.DerivationTree, bool]:
        # One bounded probabilistic step: prune infeasible at leaf, renormalize weights
        symbol, children = tree
        if children is None:
            expansions: list[utils.Expansion] = self.grammar[symbol]
            prob_map = utils.exp_probabilities(expansions)
            feasible: list[utils.Expansion] = []
            weights: list[float] = []
            for e in expansions:
                if remaining is None or len(utils.nonterminals(e)) - 1 <= remaining:
                    feasible.append(e)
                    weights.append(prob_map[utils.exp_string(e)])
            if not feasible:
                return tree, False
            # choices re-normalizes weights internally
            chosen = self._random.choices(feasible, weights=weights, k=1)[0]
            new_children = utils.expansion_to_children(chosen)
            return (symbol, new_children), True

        ids = [i for (i, c) in enumerate(children) if self.any_possible_expansions(c)]
        if not ids:
            return tree, False
        self._random.shuffle(ids)
        for i in ids:
            new_child, ok = self.expand_tree_once_bounded(children[i], remaining)
            if ok:
                children[i] = new_child
                return tree, True
        return tree, False

    def choose_node_expansion(self, node: utils.DerivationTree, expansion_alternatives: list[utils.Expansion]) -> utils.Expansion:
        # Weighted chooser used by unbounded steps (both phases)
        symbol, _ = node
        all_expansions = self.grammar[symbol]
        probabilities = utils.exp_probabilities(all_expansions)
        weights = [probabilities[utils.exp_string(exp)] for exp in expansion_alternatives]
        total = sum(weights)
        if total == 0.0:
            return self._random.choices(expansion_alternatives, k=1)[0]
        return self._random.choices(expansion_alternatives, weights=weights, k=1)[0]


class GrammarCoverageFuzzer(GrammarFuzzer):
    """Prioritize expansions that would yield a new coverage in a shallower depth (Adaptive Breadth-First Lookahead)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.covered_expansions: set[utils.ExpansionKey] = set()
        """Each grammar expansion rule is stored with its key (in the format: SYMBOL -> EXPANSION) when covered."""
        self.symbol_max_expansion_coverage: dict[str, dict[int, set[utils.ExpansionKey]]] = {}
        """Precomputation of Maximum expansion coverage for each symbol in the grammar per each possible depth"""

        self.symbol_max_expansion_coverage: dict[str, dict[int, set[utils.ExpansionKey]]] = {
            symbol: {depth: self.max_expansion_coverage(symbol, depth) for depth in range(len(self.grammar) + 1)}
            for symbol in self.grammar
        }

        self._total_coverable_expansions: int = len(self.max_expansion_coverage())
        self._coverage_cycles_completed: int = 0

    def reset_coverage(self) -> None:
        """Clear coverage info"""
        self.covered_expansions.clear()
        self._coverage_cycles_completed = 0

    def _expansion_coverage(self, symbol: str, max_depth: int, seen_nonterminals: set[str], seen_expansions: set[utils.ExpansionKey]) -> tuple[set[str], set[utils.ExpansionKey]]:
        """Helper method for computing maximum expansion coverage for a given symbol and depth. Do not call directly."""
        if max_depth <= 0:
            return seen_nonterminals, seen_expansions
        seen_nonterminals.add(symbol)

        for expansion in self.grammar[symbol]:
            seen_expansions.add(utils.expansion_key(symbol, expansion))
            for nonterminal in utils.nonterminals(expansion):
                if nonterminal not in seen_nonterminals:
                    seen_nonterminals, seen_expansions = self._expansion_coverage(nonterminal, max_depth - 1, seen_nonterminals, seen_expansions)

        return seen_nonterminals, seen_expansions

    def max_expansion_coverage(self, symbol: Optional[str] = None, max_depth: Optional[int] = None) -> set[utils.ExpansionKey]:
        """Compute all reachable expansions from `symbol` by the given maximum depth (default: grammar depth)"""
        if symbol is None:
            symbol = self.start_symbol

        if max_depth is None:
            max_depth = len(self.grammar)

        assert symbol in self.grammar, f"Invalid nonterminal symbol: {symbol}"
        assert 0 <= max_depth, f"Invalid max_depth: {max_depth}"

        # Check if the coverage is already computed
        if symbol in self.symbol_max_expansion_coverage and max_depth in self.symbol_max_expansion_coverage[symbol]:
            return self.symbol_max_expansion_coverage[symbol][max_depth]

        # Traverse the grammar from `symbol` and store the key of all covered expansions
        seen_nonterminals, seen_expansions = self._expansion_coverage(symbol, max_depth, set(), set())

        assert len(seen_nonterminals) <= len(self.grammar), "Visited invalid number of nonterminals"
        assert len(seen_expansions) <= sum(len(self.grammar[s]) for s in seen_nonterminals), "Visited invalid number of expansions"

        return seen_expansions

    def add_coverage(self, non_terminal: str, expansion: utils.Expansion) -> None:
        """Add the key of the expansion to the covered expansions"""
        key = utils.expansion_key(non_terminal, expansion)
        if key in self.covered_expansions:
            return
        self.covered_expansions.add(key)

        if 0 < self._total_coverable_expansions <= len(self.covered_expansions):
            self._coverage_cycles_completed += 1
            self.covered_expansions.clear()

    def missing_expansion_coverage(self) -> set[utils.ExpansionKey]:
        """Return expansions, not covered yet"""
        return self.max_expansion_coverage() - self.covered_expansions

    def new_coverage(self, symbol: str, expansion: utils.Expansion, max_depth: int) -> set[utils.ExpansionKey]:
        new_cov: set[utils.ExpansionKey] = {utils.expansion_key(symbol, expansion)}

        remaining = max_depth - 1
        if remaining < 0:
            return new_cov - self.covered_expansions

        for child_symbol, _ in utils.expansion_to_children(expansion):
            if child_symbol in self.grammar:  # nonterminal
                new_cov |= self.max_expansion_coverage(child_symbol, remaining)

        return new_cov - self.covered_expansions

    def new_coverages(self, symbol: str, expansion_alternatives: list[utils.Expansion]) -> list[set[utils.ExpansionKey]] | None:
        """Return maximum new coverages that would be obtained by expanding `symbol` each expansion at minimum depth"""
        for max_depth in range(len(self.grammar) + 1):
            new_coverages = [self.new_coverage(symbol, expansion, max_depth) for expansion in expansion_alternatives]
            if any(len(cov) > 0 for cov in new_coverages):
                return new_coverages

        return None  # All covered

    def choose_node_expansion(self, node: utils.DerivationTree, expansion_alternatives: list[utils.Expansion]) -> utils.Expansion:
        """Choose an expansion of `node` among `expansion_alternatives` that yields the highest additional coverage.
        If all expansions are covered, fall back to the superclass method."""
        symbol, _ = node
        new_coverages = self.new_coverages(symbol, expansion_alternatives)

        if new_coverages is None:  # All expansions covered - select randomly (parent method)
            chosen_expansion = super().choose_node_expansion(node, expansion_alternatives)
            self.add_coverage(node[0], chosen_expansion)
            return chosen_expansion

        # Find a subset of expansions with the maximum new coverage
        max_coverage = max(len(coverage) for coverage in new_coverages)
        expansions_with_max_coverage = [
            expansion for (i, expansion) in enumerate(expansion_alternatives) if len(new_coverages[i]) == max_coverage
        ]
        chosen_expansion = super().choose_node_expansion(node, expansions_with_max_coverage)
        self.add_coverage(node[0], chosen_expansion)
        return chosen_expansion

    def coverage_proportion(self) -> float:
        """
        Return cumulative coverage as a floating point number:
        - 0.8 if 80% of the grammar is covered in the current cycle
        - 2.5 if fully covered 2 times, plus 50% in the current cycle
        """
        total = self._total_coverable_expansions
        if total <= 0:
            return float(self._coverage_cycles_completed)
        return self._coverage_cycles_completed + (len(self.covered_expansions) / total)


class KPathGrammarFuzzer(GrammarFuzzer):
    """
    Fuzzer that targets coverage of nonterminal k‑paths in a grammar.

    A k‑path is any suffix (length 1..k) of the ancestor→descendant chain of nonterminal
    symbols ending at a node. For every expanded nonterminal node, this fuzzer marks as
    covered all suffixes of its ancestor chain (including itself) up to length ``k``.
    Terminals are excluded from paths. Coverage is computed over all distinct k‑paths
    reachable from the start symbol.

    Behavior:
      - When ``guidance`` is True, expansion choices are biased toward alternatives that
        yield the largest number of previously unseen k‑paths; ties are broken randomly,
        or (optionally) by preferring expansions whose nonterminal children can continue.
      - When ``guidance`` is False, selection falls back to the base strategy.
      - ``coverage_mode='cumulative'`` reports the fraction of distinct k‑paths covered in
        the current run; ``coverage_mode='exact'`` counts completed full‑coverage cycles
        and the fraction in the ongoing cycle.

    :param grammar: Context‑free grammar mapping nonterminal symbols to expansions.
                    Must be valid for the base fuzzer.
    :param start_symbol: Start nonterminal of the grammar.
    :param min_nonterminals: Minimum allowed number of nonterminals in the final derivation.
                             Enforced by the base expansion logic.
    :param max_nonterminals: Upper bound on the derivation frontier during growth.
                             Enforced by the base expansion logic.
    :param seed: Seed for the internal pseudo‑random generator used for selection.
    :param k: Positive length of paths to cover (k ≥ 1). ``k=1`` reduces to symbol coverage.
    :param guidance: If True, enable coverage‑guided expansion; if False, defer to the base
                     chooser (e.g., random or cost‑based, depending on strategy).
    :param coverage_mode: Coverage accounting mode; ``'cumulative'`` returns only the current
                          cycle proportion, while ``'exact'`` returns completed cycles plus
                          the current cycle proportion.
    :param prioritize_longest: If True, when multiple expansions provide equal new k‑path
                               gain, prefer those whose nonterminal children can continue
                               (i.e., have at least one expansion that introduces a nonterminal).

    :raises ValueError: If ``k`` is not positive, or if ``coverage_mode`` is not one of ``'cumulative'`` or ``'exact'``.

    Notes:
      - Use ``coverage_proportion()`` to query cumulative or exact coverage as configured.
      - Paths only include nonterminals present in ``grammar``; terminals are ignored.
    """

    def __init__(self,
                 grammar: utils.Grammar,
                 start_symbol: str = utils.START_SYMBOL,
                 min_nonterminals: int = utils.MIN_NONTERMINALS,
                 max_nonterminals: int = utils.MAX_NONTERMINALS,
                 seed=utils.SEED,
                 k: int = 2,
                 guidance: bool = True,
                 coverage_mode: str = "exact",
                 prioritize_longest: bool = False,
                 **kwargs):
        super().__init__(grammar=grammar, start_symbol=start_symbol, seed=seed,
                         min_nonterminals=min_nonterminals, max_nonterminals=max_nonterminals, **kwargs)
        if k <= 0:
            raise ValueError("k must be positive")
        if coverage_mode not in ("cumulative", "exact"):
            raise ValueError("coverage_mode must be 'cumulative' or 'exact'")
        self._k = k
        self.guidance = guidance
        self.coverage_mode = coverage_mode
        self.prioritize_longest = prioritize_longest
        self.covered_paths: set[tuple[str, ...]] = set()
        self._all_possible_paths: set[tuple[str, ...]] | None = None
        self._coverage_cycles = 0
        self._parent_chain_for_guidance = []

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int):
        assert isinstance(value, int) and value > 0, (f"k must be a positive integer, "
                                                      f"got type={type(value)} value={value} instead.")
        if value != self._k:
            self._k = value
            self.covered_paths.clear()
            self._all_possible_paths = None

    def reset_coverage(self):
        self.covered_paths.clear()
        self._coverage_cycles = 0

    def _compute_all_possible_paths(self) -> set[tuple[str, ...]]:
        if self._all_possible_paths is not None:
            return self._all_possible_paths

        all_paths: set[tuple[str, ...]] = set()
        seen_states: set[tuple[str, ...]] = set()
        queue = [[self.start_symbol]]
        seen_states.add((self.start_symbol,))

        while queue:
            chain = queue.pop(0)
            for L in range(1, min(self._k, len(chain)) + 1):
                all_paths.add(tuple(chain[-L:]))

            last_sym = chain[-1]
            if last_sym not in self.grammar:
                continue

            for exp in self.grammar[last_sym]:
                for child in utils.nonterminals(exp):
                    if child not in self.grammar:
                        continue
                    new_chain = chain + [child]
                    state = tuple(new_chain[-self._k:])
                    if state in seen_states:
                        continue
                    seen_states.add(state)
                    queue.append(new_chain)

        self._all_possible_paths = all_paths
        return all_paths

    def coverage_proportion(self) -> float:
        if self._k <= 0:
            return 0.0
        total = len(self._compute_all_possible_paths())
        if total == 0:
            return 0.0
        if self.coverage_mode == "cumulative":
            return len(self.covered_paths) / total
        else:
            return self._coverage_cycles + (len(self.covered_paths) / total)

    def fuzz(self) -> str:
        self.derivation_tree = (self.start_symbol, None)
        self._record_paths_for_node(self.derivation_tree, [])
        self.derivation_tree = self.expand_tree(self.derivation_tree, chain=[self.start_symbol])
        return utils.all_terminals(self.derivation_tree)

    def expand_tree_once(self, tree: utils.DerivationTree, **kwargs) -> utils.DerivationTree:
        symbol, children = tree
        chain = kwargs.get('chain', [self.start_symbol])

        if children is None:  # Expand this node
            # Supply parent chain (which includes current node) to guidance
            self._parent_chain_for_guidance = chain
            expanded = self.expand_node(tree)
            _, new_children = expanded
            if new_children:
                for child in new_children:
                    # record child's paths with correct parent chain
                    self._record_paths_for_node(child, chain)
            return expanded

        # Choose a random expandable child
        expandable = [i for i, c in enumerate(children) if self.any_possible_expansions(c)]
        if not expandable:
            return tree
        idx = self._random.choice(expandable)  # was random.choice
        child_sym, _ = children[idx]
        child_chain = chain + [child_sym]
        children[idx] = self.expand_tree_once(children[idx], chain=child_chain)
        return tree

    def _record_paths_for_node(self, node: utils.DerivationTree, ancestor_chain: list[str]):
        sym, _ = node
        if not utils.is_nonterminal(sym):
            return

        current_chain = ancestor_chain + [sym]
        added: set[tuple[str, ...]] = set()
        for L in range(1, min(self._k, len(current_chain)) + 1):
            path = tuple(current_chain[-L:])
            if path not in self.covered_paths:
                added.add(path)

        self.covered_paths.update(added)

        if self.coverage_mode == "exact":
            all_paths = self._compute_all_possible_paths()
            if len(self.covered_paths) >= len(all_paths):
                self._coverage_cycles += 1
                self.covered_paths.clear()
                self.covered_paths.update(added)

    def _compute_new_paths(self, parent_chain: list[str], expansion: utils.Expansion) -> set[tuple[str, ...]]:
        new_paths: set[tuple[str, ...]] = set()
        for child_sym in utils.nonterminals(expansion):
            if child_sym not in self.grammar:
                continue
            child_chain = parent_chain + [child_sym]
            for L in range(1, min(self._k, len(child_chain)) + 1):
                path = tuple(child_chain[-L:])
                if path not in self.covered_paths:
                    new_paths.add(path)
        return new_paths

    def _expansion_continuation_score(self, expansion: utils.Expansion) -> int:
        max_cont = 0
        for child_sym in utils.nonterminals(expansion):
            if child_sym not in self.grammar:
                continue
            cont = 1 if any(utils.nonterminals(e) for e in self.grammar.get(child_sym, [])) else 0
            if cont > max_cont:
                max_cont = cont
        return max_cont

    def choose_node_expansion(self, node: utils.DerivationTree, alternatives: list[utils.Expansion]) -> utils.Expansion:
        if not self.guidance or not alternatives:
            return super().choose_node_expansion(node, alternatives)
        scores = [len(self._compute_new_paths(self._parent_chain_for_guidance, exp)) for exp in alternatives]
        if not scores or max(scores) == 0:
            return super().choose_node_expansion(node, alternatives)

        max_score = max(scores)
        best = [exp for exp, s in zip(alternatives, scores) if s == max_score]

        if self.prioritize_longest and len(best) > 1:
            cont_scores = [self._expansion_continuation_score(exp) for exp in best]
            best_cont = max(cont_scores)
            best = [exp for exp, cs in zip(best, cont_scores) if cs == best_cont]

        return self._random.choice(best)

    def _build_ancestor_chain(self, node: utils.DerivationTree) -> list[str]:
        if not hasattr(self, 'derivation_tree') or self.derivation_tree is None:
            return []
        chain: list[utils.DerivationTree] = []
        self._find_node_path(self.derivation_tree, node, chain)
        return [sym for sym, _ in chain if utils.is_nonterminal(sym)]

    def _find_node_path(self, current: utils.DerivationTree, target: utils.DerivationTree, path: list[utils.DerivationTree]) -> bool:
        if current is target:
            path.append(current)
            return True

        sym, children = current
        if children is None:
            return False

        path.append(current)
        for child in children:
            if self._find_node_path(child, target, path):
                return True
        path.pop()
        return False
