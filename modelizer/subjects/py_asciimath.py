from py_asciimath.translator.translator import MathML2Tex, Tex2ASCIIMath


def fix_latex_formula(latex: str):
    latex = r"{}".format(latex)
    latex.replace(r"\left", "").replace(r"\right", "")
    latex = r"{}".format(latex)
    latex = latex.replace(r"\_", "_")
    latex = latex.replace(" ", "")
    return latex


def mathml2latex(mathml: str):
    converter = MathML2Tex()
    xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
    doctype = '<!DOCTYPE math PUBLIC "-//W3C//DTD MathML 2.0//EN" "http://www.w3.org/Math/DTD/mathml3/mathml3.dtd">'
    mathml = f'{xml_header}\n{doctype}\n{mathml}'
    latex = converter.translate(mathml, network=False, from_file=False)[2:-1]
    latex = fix_latex_formula(latex)
    return latex


def latex2ascii_math(latex: str):
    converter = Tex2ASCIIMath(log=False, inplace=True)
    return converter.translate(latex, from_file=False, pprint=False).replace(" ", "")
