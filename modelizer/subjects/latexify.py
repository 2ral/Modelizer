# Temp extension to latexify to allow for converting python code representation as string to latex
# https://github.com/google/latexify_py/issues/108
# Can be deprecated with latexify 0.3.0 and replaced with built-in implementation
import ast
import textwrap
from latexify import codegen, transformers, exceptions


def __python2latex__(
        code_string: str,
        *,
        identifiers: dict[str, str] | None = None,
        reduce_assignments: bool = False,
        use_math_symbols: bool = False,
        use_raw_function_name: bool = False,
        use_signature: bool = True,
        use_set_symbols: bool = False,
) -> str:
    # Obtains the source AST.
    tree = ast.parse(textwrap.dedent(code_string))
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        raise exceptions.LatexifySyntaxError("Not a function.")

    # Applies AST transformations.
    if identifiers is not None:
        tree = transformers.IdentifierReplacer(identifiers).visit(tree)
    if reduce_assignments:
        tree = transformers.AssignmentReducer().visit(tree)

    # Generates LaTeX.
    return codegen.FunctionCodegen(
        use_math_symbols=use_math_symbols,
        use_raw_function_name=use_raw_function_name,
        use_signature=use_signature,
        use_set_symbols=use_set_symbols,
    ).visit(tree)


def python2latex(expression: str) -> str:
    latex = r"{}".format(__python2latex__(f"def f():\n\treturn {expression}")).replace("()", "", 1)
    latex = latex.replace(r"\mathrm{f} = ", "", 1).replace(" ", "")
    return r"{}".format(latex)
