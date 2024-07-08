from pathlib import Path
from modelizer.subjects.latexify import __python2latex__
from modelizer.generators.fuzzer import AbstractFuzzer
from modelizer.tokenizer.expression import PythonExpressionTokenizer, LatexExpressionTokenizer
from modelizer.generators.grammars import PYTHON_EXPRESSION_GRAMMAR, PYTHON_EXPRESSION_MARKERS


class PythonExpressionFuzzer(AbstractFuzzer):
    def __init__(self, root_dir: Path):
        super(PythonExpressionFuzzer, self).__init__("Python Expression", root_dir,
                                                     PYTHON_EXPRESSION_GRAMMAR,
                                                     markers=PYTHON_EXPRESSION_MARKERS,
                                                     columns=["expression", "latex"])

    @staticmethod
    def __convert__(subjects: list[str]) -> list[tuple]:
        results = []
        for s in subjects:
            latex = r"{}".format(__python2latex__(f"def f():\n\treturn {s}")).replace("()", "", 1)
            latex = latex.replace(r"\mathrm{f} = ", "", 1)
            latex = r"{}".format(latex)
            results.append((s, latex))
        return results

    @staticmethod
    def __tokenize__(subjects: list[tuple]) -> list[tuple]:
        expression_tokenizer = PythonExpressionTokenizer()
        latex_tokenizer = LatexExpressionTokenizer()
        return [(expression_tokenizer.feed(expression), latex_tokenizer.feed(latex)) for expression, latex in subjects]
