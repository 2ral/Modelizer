from modelizer.generators.abstract import (
    GeneratorInterface,
    BaseSubject,
    Optional,
    Logger,
    configs,
)

from modelizer.dependencies.fuzzingbook import (
    Grammar,
    GrammarFuzzer as __GrammarFuzzer__,
    ProbabilisticGrammarFuzzer as __ProbabilisticGrammarFuzzer__,
    GrammarCoverageFuzzer as __GrammarCoverageFuzzer__,
    KPathGrammarFuzzer as __KPathGrammarFuzzer__,
    convert_and_validate_ebnf_grammar
)


class GrammarFuzzerGenerator(GeneratorInterface):
    """A class for generating strings from a given grammar."""
    def __init__(self,
                 grammar: Grammar,
                 source: str,
                 target: str,
                 subject: BaseSubject,
                 *,
                 fuzzer_type: str = "random",
                 min_nonterminals: int = 0,
                 max_nonterminals: int = 10,
                 seed: int = configs.SEED,
                 logger: Optional[Logger] = None, **_):
        """
        Constructor for the GrammarFuzzer class.
        :param grammar: the dictionary of recursive string generation rules to synthesize inputs.
        Supported formats are EBNF and BNF.
        :param source: the source type name.
        :param target: the target type name.
        :param subject: the instance of the BaseSubject subclass.
        :param fuzzer_type: the type of fuzzer to use. Options are "coverage", "kpath", and "probabilistic".
            - "coverage": uses GrammarCoverageFuzzer to maximize grammar coverage.
            - "kpath": uses KPathGrammarFuzzer to ensure each production rule is used at least k times.
            - "probabilistic": uses ProbabilisticGrammarFuzzer to generate strings based on specified probabilities in the grammar.
            - "random": uses GrammarFuzzer to generate strings randomly without specific coverage or probabilities.
            Default is "random".
        :param min_nonterminals: the minimum number of nonterminals expansions in the grammar to be performed by the fuzzer.
        :param max_nonterminals: the maximum number of nonterminals expansions in the grammar to be performed by the fuzzer.
        :param seed: the seed to initialize the random number generator for reproducibility.
        :param logger: the optional logger for logging data generation process.
        """
        super().__init__(source, target, subject, seed, logger)
        self._grammar = convert_and_validate_ebnf_grammar(grammar)
        if "{'prob':" in str(grammar) and fuzzer_type != "probabilistic":
            self._logger.warning("The provided grammar contains probabilities, switching fuzzer_type to 'probabilistic'.")
            fuzzer_type = "probabilistic"
        match fuzzer_type:
            case "coverage":
                fuzzer = __GrammarCoverageFuzzer__
            case "kpath":
                fuzzer = __KPathGrammarFuzzer__
            case "probabilistic":
                fuzzer = __ProbabilisticGrammarFuzzer__
            case "random":
                fuzzer = __GrammarFuzzer__
            case _:
                self._logger.warning(f"Unknown fuzzer_type '{fuzzer_type}', defaulting to 'random'.")
                fuzzer = __GrammarFuzzer__
        self._fuzzer = fuzzer(self._grammar, min_nonterminals=min_nonterminals,  max_nonterminals=max_nonterminals, seed=seed)

    def generate(self) -> str:
        return self._fuzzer.fuzz()
