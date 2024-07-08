from modelizer.tokenizer.markup import MarkdownTokenizer, HTMLTokenizer
from modelizer.tokenizer.query import SQLTokenizer, KQLTokenizer
from modelizer.tokenizer.mathml import MathMLTokenizer, ASCIIMathTokenizer, LatexFormulaTokenizer, SVGTokenizer
from modelizer.tokenizer.expression import PythonExpressionTokenizer, LatexExpressionTokenizer

TOKENIZERS_MAPPING = {
    "markdown": MarkdownTokenizer,
    "html": HTMLTokenizer,
    "sql": SQLTokenizer,
    "kql": KQLTokenizer,
    "svg": SVGTokenizer,
    "mathml": MathMLTokenizer,
    "ascii_math": ASCIIMathTokenizer,
    "latex_mathml": LatexFormulaTokenizer,
    "expression": PythonExpressionTokenizer,
    "latex_expression": LatexExpressionTokenizer,
}
