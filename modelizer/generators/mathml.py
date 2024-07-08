from pathlib import Path
from datetime import datetime
from multiprocessing import current_process
from xml.etree.ElementTree import ParseError

from modelizer.generators.fuzzer import AbstractFuzzer
from modelizer.generators.utils import PlaceholderProcessor
from modelizer.subjects.py_asciimath import MathML2Tex, Tex2ASCIIMath, fix_latex_formula
from modelizer.generators.grammars import MATHML_GRAMMAR, MATHML_REFINED_GRAMMAR, MATHML_MARKERS
from modelizer.tokenizer.mathml import MathMLTokenizer, LatexFormulaTokenizer, ASCIIMathTokenizer, SVGTokenizer


class MathMLFuzzer(AbstractFuzzer):
    def __init__(self, root_dir: Path):
        super(MathMLFuzzer, self).__init__("MathML", root_dir, MATHML_GRAMMAR, MATHML_MARKERS, ["mathml", "latex", "ascii_math"])

    @staticmethod
    def __convert__(subjects: list[str]) -> list[tuple]:
        # Parsers and Converters
        mathml_converter = MathML2Tex()
        latex_converter = Tex2ASCIIMath(log=False, inplace=True)
        processor = PlaceholderProcessor(MATHML_MARKERS)

        # LaTeX Header and Doctype
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
        doctype = '<!DOCTYPE math PUBLIC "-//W3C//DTD MathML 2.0//EN" "http://www.w3.org/Math/DTD/mathml3/mathml3.dtd">'

        mathml_rules = [(">", "> "), ("<", " <"), ("  ", " ")]

        result = []
        for mathml_formula in subjects:
            # Replace Placeholders with random values
            for before_rule, after_rule in mathml_rules:
                mathml_formula = mathml_formula.replace(before_rule, after_rule)
            mathml_formula = mathml_formula[1:-1]
            formula, formula_mapping = processor.replace_placeholders(mathml_formula)
            # MathML to LaTeX
            xml_file = f'{xml_header}\n{doctype}\n{formula}'
            try:
                latex = mathml_converter.translate(xml_file, network=False, from_file=False)[2:-1]
            except:
                continue
            latex = fix_latex_formula(latex)
            # LaTeX to ASCII Math
            try:
                ascii_math = latex_converter.translate(latex, from_file=False, pprint=False).replace(" ", "")
            except:
                continue
            # Recover Placeholders
            latex = processor.recover_placeholders(latex, formula_mapping)
            ascii_math = processor.recover_placeholders(ascii_math, formula_mapping)
            # Append to Result
            result.append((mathml_formula, latex, ascii_math))
        return result

    @staticmethod
    def __tokenize__(subjects: list[tuple]) -> list[tuple]:
        mathml_tokenizer = MathMLTokenizer()
        latex_tokenizer = LatexFormulaTokenizer()
        ascii_math_tokenizer = ASCIIMathTokenizer()
        return [(mathml_tokenizer.feed(mathml), latex_tokenizer.feed(latex), ascii_math_tokenizer.feed(ascii_math))
                for mathml, latex, ascii_math in subjects]
