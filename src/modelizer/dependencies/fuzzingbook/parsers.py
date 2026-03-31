import modelizer.dependencies.fuzzingbook.utils as utils

from abc import ABC, abstractmethod

from collections import deque
from typing import Iterator, Generator


class Parser(ABC):
    """Base class for parsing."""
    def __init__(self, grammar: utils.Grammar, start_symbol=utils.START_SYMBOL, canonical_grammar=False, coalesce=True, tokens=None, log=False):
        if tokens is None:
            tokens = set()

        self._grammar = grammar
        self._start_symbol = start_symbol
        self.log = log
        self.coalesce_tokens = coalesce
        self.tokens = tokens

        # Cache for tree pruning
        self._prune_cache = {}

        if canonical_grammar:
            self.cgrammar = utils.single_char_tokens(grammar)
            self._grammar = utils.non_canonical(grammar)
        else:
            self._grammar = dict(grammar)
            self.cgrammar = utils.single_char_tokens(utils.canonical(grammar))

        if len(grammar.get(self._start_symbol, [])) != 1:
            self.cgrammar['<>'] = [[self._start_symbol]]

    @property
    def grammar(self):
        return self._grammar

    @abstractmethod
    def parse_prefix(self, text):
        """Returns the pair (cursor, forest), which contains the index until which
         parsing was completed successfully, and the parse forest until that index."""
        raise NotImplementedError

    def parse(self, text):
        """Parse `text` using the grammar. Return a list of derivation trees if the parse was successful."""
        cursor, forest = self.parse_prefix(text)
        if cursor < len(text):
            raise SyntaxError("at " + repr(text[cursor:]))
        return [self.prune_tree(t) for t in forest]

    def prune_tree(self, tree):
        """Optimized tree pruning with caching and iterative approach"""
        # Create cache key
        cache_key = id(tree)
        if cache_key in self._prune_cache:
            return self._prune_cache[cache_key]

        name, children = tree

        if name == '<>':
            result = self.prune_tree(children[0])
            self._prune_cache[cache_key] = result
            return result

        if name in self.tokens:
            result = name, [(utils.tree_to_string(tree), [])]
            self._prune_cache[cache_key] = result
            return result

        new_root = (name, [])
        queue = deque([(tree, new_root)])

        while queue:
            (orig_name, orig_children), new_node = queue.popleft()
            if self.coalesce_tokens:
                orig_children = self.coalesce(orig_children)
            for cn, cc in orig_children:
                if cn == '<>':
                    cn, cc = cc[0]
                if cn in self.tokens:
                    new_node[1].append((cn, [(utils.tree_to_string((cn, cc)), [])]))
                else:
                    new_child = (cn, [])
                    new_node[1].append(new_child)
                    queue.append(((cn, cc), new_child))

        self._prune_cache[cache_key] = new_root
        return new_root

    def coalesce(self, children):
        """Optimized coalesce using list comprehension and string builder"""
        if not children:
            return []

        result = []
        current_str_parts = []

        for cn, cc in children:
            if cn not in self._grammar:
                current_str_parts.append(cn)
            else:
                if current_str_parts:
                    result.append((''.join(current_str_parts), []))
                    current_str_parts.clear()
                result.append((cn, cc))

        if current_str_parts:
            result.append((''.join(current_str_parts), []))

        return result

    def parse_on(self, text, start_symbol):
        old = self._start_symbol
        try:
            self._start_symbol = start_symbol
            yield from self.parse(text)
        finally:
            self._start_symbol = old

    @staticmethod
    def derivation_tree_to_pattern(tree: utils.DerivationTree) -> str:
        """Convert a derivation tree to a pattern showing the structure with non-terminals replacing terminal values."""
        node_results = {}

        def process_node(node):
            node_id = id(node)
            if node_id in node_results:
                return node_results[node_id]

            symbol, children = node

            if not children:
                result = symbol
            elif utils.is_nonterminal(symbol) and len(children) == 1 and not children[0][1]:
                result = symbol
            else:
                result = ''.join(process_node(child) for child in children)

            node_results[node_id] = result
            return result

        return process_node(tree)

    def input_to_pattern(self, text: str) -> str | None:
        """
        Convert an input string to a pattern by parsing it and then converting the resulting tree to a pattern.
        :param text: Input string to be converted to a pattern
        :return: Pattern string representing the structure of the input
        """
        assert text is not None and len(text) > 0, "Input text must not be empty and must not be None"
        try:
            tree = self.parse(text)
        except SyntaxError:
            return text

        if isinstance(tree, Iterator):
            while True:
                try:
                    return self.derivation_tree_to_pattern(next(tree))
                except SyntaxError:
                    continue
                except StopIteration:
                    return text
        else:
            return self.derivation_tree_to_pattern(tree)  # type: ignore

    def rewrite_mixed_grammar_rules(self) -> utils.Grammar:
        return utils.rewrite_mixed_grammar_rules(self._grammar)

    def get_nonterminal_rules(self) -> utils.Grammar:
        return utils.get_nonterminal_rules(self._grammar)

    def get_terminal_rules(self) -> utils.Grammar:
        return utils.get_terminal_rules(self._grammar)

    def extract_terminals_from_input(self, text: str) -> list[str]:
        """Extract terminal symbols from the input text based on the grammar."""
        try:
            trees = self.parse(text)
        except SyntaxError:
            terminals = []
        else:
            terminals = utils.extract_terminals_from_derivation_tree(trees)
        return terminals


class EarleyParser(Parser):
    """Earley Parser. This parser can parse any context-free grammar."""
    def __init__(self, grammar: utils.Grammar, placeholder_mapping=None, **kwargs):
        super().__init__(grammar, **kwargs)
        self.epsilon = utils.nullable(self.cgrammar)
        self.table = None
        self._furthest = -1
        self._completed_at = {}
        self.placeholder_mapping = placeholder_mapping

        # Caches for optimization
        self._parse_paths_cache = {}
        self._forest_cache = {}

    def chart_parse(self, words, start):
        chart = [utils.Column(i, tok) for i, tok in enumerate([None, *words])]
        chart[0].add(utils.State(start, tuple(self.cgrammar[start][0]), 0, chart[0]))
        return self.fill_chart(chart)

    def fill_chart(self, chart):
        furthest = -1
        comp_at = {}

        for col in chart:
            i = 0
            states = col.states

            while i < len(states):
                st = states[i]
                i += 1

                if st.finished():
                    # Batch process waiting states
                    waiting_states = st.s_col.waiting.get(st.name, ())
                    for p in waiting_states:
                        col.add(p.advance())
                    if st.name == self._start_symbol:
                        if col.index > furthest:
                            furthest = col.index
                        comp_at.setdefault(col.index, []).append(st)
                else:
                    sym = st.at_dot()
                    if sym in self.cgrammar:
                        # Add all alternatives at once
                        for alt in self.cgrammar[sym]:
                            col.add(utils.State(sym, tuple(alt), 0, col))
                        if sym in self.epsilon:
                            col.add(st.advance())
                    else:
                        nc_i = col.index + 1
                        if nc_i < len(chart):
                            nc = chart[nc_i]
                            if nc.letter == sym:
                                nc.add(st.advance())

            if self.log:
                print(col, "\n")

        self._furthest = furthest
        self._completed_at = comp_at
        return chart

    def parse_prefix(self, text):
        words = text if isinstance(text, list) else list(text)
        self.table = self.chart_parse(words, self._start_symbol)
        if self._furthest >= 0:
            return self._furthest, self._completed_at[self._furthest]
        return -1, []

    def parse(self, text):
        cursor, states = self.parse_prefix(text)
        start = next((s for s in states if s.finished()), None)

        if cursor < len(text) or not start:
            raise SyntaxError("at " + repr(text[cursor:]))

        forest = self.parse_forest(self.table, start)
        for tree in self.extract_trees(forest):
            yield self.prune_tree(tree)

    def parse_paths(self, named_expr, chart, frm, til):
        cache_key = (tuple(named_expr), frm, til)
        if cache_key in self._parse_paths_cache:
            return self._parse_paths_cache[cache_key]

        def paths(state, start, k, e):
            if not e:
                return [[(state, k)]] if start == frm else []
            return [[(state, k)] + r for r in self.parse_paths(e, chart, frm, start)]

        *expr, var = named_expr
        if var not in self.cgrammar:
            starts = [(var, til - len(var), 't')] if til > 0 and chart[til].letter == var else []
        else:
            starts = [(s, s.s_col.index, 'n') for s in chart[til].states if s.finished() and s.name == var]

        result = [p for s, start, k in starts for p in paths(s, start, k, expr)]
        self._parse_paths_cache[cache_key] = result
        return result

    def forest(self, s, kind, chart):
        return self.parse_forest(chart, s) if kind == 'n' else (s, [])

    def parse_forest(self, chart, state):
        cache_key = (id(chart), id(state))
        if cache_key in self._forest_cache:
            return self._forest_cache[cache_key]

        pathexprs = self.parse_paths(state.expr, chart, state.s_col.index, state.e_col.index) if state.expr else []
        result = state.name, [[(v, k, chart) for v, k in reversed(p)] for p in pathexprs]
        self._forest_cache[cache_key] = result
        return result

    def extract_trees(self, forest_node) -> Generator[utils.DerivationTree, None, None]:
        name, paths = forest_node
        if not paths:
            yield name, []
            return

        for path in paths:
            tree_generators = [self.extract_trees(self.forest(*p)) for p in path]

            def generate_combinations(generators, current_combination):
                if not generators:
                    yield name, tuple(current_combination)
                    return

                first_gen = generators[0]
                rest_gens = generators[1:]

                for tree in first_gen:
                    yield from generate_combinations(rest_gens, current_combination + [tree])

            yield from generate_combinations(tree_generators, [])

    def abstract_tree(self, tree: utils.DerivationTree) -> utils.DerivationTree:
        if self.placeholder_mapping is None:
            return tree

        sym, children = tree
        if sym in self.placeholder_mapping:
            return self.placeholder_mapping[sym], []

        new_root = (sym, [])
        queue = deque([(tree, new_root)])

        while queue:
            (orig_sym, orig_children), new_node = queue.popleft()
            if orig_children:
                for cs, cc in orig_children:
                    if cs in self.placeholder_mapping:
                        new_node[1].append((self.placeholder_mapping[cs], []))
                    else:
                        new_child = (cs, [])
                        new_node[1].append(new_child)
                        if cc:
                            queue.append(((cs, cc), new_child))
        return new_root

    def abstract(self, text: str) -> str:  # type: ignore
        """
        Abstract the input text by replacing specified non-terminal nodes with placeholders.
        :param text: Input string to get abstracted
        :return: Abstracted string with non-terminals replaced with placeholders
        """
        if self.placeholder_mapping is None:
            return text

        while True:
            try:
                tree = next(self.parse(text))
            except SyntaxError:
                continue
            except StopIteration:
                return text
            else:
                modified = self.abstract_tree(tree)
                return utils.tree_to_string(modified)

    def abstract_mapped(self, text: str) -> tuple[str, dict[str, str]]:
        if self.placeholder_mapping is None:
            return text, {}

        tree = None
        while True:
            try:
                tree = next(self.parse(text))
            except SyntaxError:
                continue
            except StopIteration:
                return text, {}
            else:
                break
        if tree is None:
            raise ValueError("Failed to parse the input text.")
        else:
            token_mapping: dict[str, dict[str, str]] = {}
            counters: dict[str, int] = {}
            mapping: dict[str, str] = {}
            result_parts = []

            def traverse(node: tuple[str, list]) -> None:
                sym, children = node
                if sym in self.placeholder_mapping:
                    token_type = self.placeholder_mapping[sym]
                    real_val = utils.tree_to_string(node)
                    if token_type not in token_mapping:
                        token_mapping[token_type] = {}
                        counters[token_type] = 1
                    if real_val in token_mapping[token_type]:
                        result_parts.append(token_mapping[token_type][real_val])
                    else:
                        placeholder = f"{token_type}_{counters[token_type]}"
                        counters[token_type] += 1
                        token_mapping[token_type][real_val] = placeholder
                        mapping[placeholder] = real_val
                        result_parts.append(placeholder)
                else:
                    if children is None or children == []:
                        if not utils.is_nonterminal(sym):
                            result_parts.append(sym)
                    else:
                        for child in children:
                            traverse(child)

            traverse(tree)
            return ''.join(result_parts), mapping
