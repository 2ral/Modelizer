from pathlib import Path
from re import compile as re_compile
from modelizer.tokenizer.generic import WhitespaceTokenizer


class FormulaTokenizer(WhitespaceTokenizer):
    def __init__(self, special_tokens: dict[str, str], exclusion_tokens: tuple = ()):
        super(FormulaTokenizer, self).__init__(placeholders=("NUMBER", "IDENTIFIER"))
        self.__special_tokens__ = special_tokens
        self.__exclusions__ = exclusion_tokens
        self.__special_tokens_inverse__ = {v: k for k, v in self.__special_tokens__.items()}

    def feed(self, data: str):
        data = [t.strip() for t in data.split(self.__delimiter__) if t.strip()]
        if self.__mask_mapping__:
            for i, v in enumerate(data):
                if v.isdigit() or v.replace(".", "", 1).isdigit() or v.replace("..", "", 1).isdigit():
                    tag = self.__mask_token__(v, self.__placeholders__[0])  # Number
                elif v in self.__special_tokens__:
                    tag = self.__mask_token__(self.__special_tokens__[v], self.__placeholders__[1])  # Identifier
                elif v.isalnum() and v not in self.__exclusions__:
                    tag = self.__mask_token__(v, self.__placeholders__[1])  # Identifier
                else:
                    continue
                data[i] = tag
        self.buffer = data

    def mapped_reconstruct(self, tokens: list[str], mapping: dict[str, str]) -> str:
        for k, v in mapping.items():
            if "IDENTIFIER_" in k and v in self.__special_tokens_inverse__:
                mapping[k] = self.__special_tokens_inverse__[v]
        return super().mapped_reconstruct(tokens, mapping)


class MathMLTokenizer(FormulaTokenizer):
    def __init__(self):
        special_tokens = {
            "λ": "lambda_token",
            "δ": "delta_token",
            "ε": "epsilon_token",
            "ψ": "psi_token",
            "Φ": "phi_token",
            "σ": "sigma_token",
            "τ": "tau_token",
            "Δ": "delta_token2",
            "θ": "theta_token",
            "α": "alpha_token",
            "β": "beta_token",
            "ϕ": "phi_token2",
            "γ": "gamma_token",
            "Λ": "lambda_token2",
            "μ": "mu_token",
            "π": "pi_token",
            "ρ": "rho_token",
            "Γ": "gamma_token2",
            "η": "eta_token",
            "ξ": "xi_token",
            "Θ": "theta_token2",
            "ζ": "zeta_token",
            "∞": "infinity_token",
            "⋯": "cdots_token",
            "interval": "interval_token",
        }

        self.__char_mapping__ = {
            "≠": "&ne", "≈": "&ap", "∼": "&sim", "≅": "&cong", "∝": "&propto", "≙": "&wedgeq", "<": "&lt", "≤": "&leq",
            "≪": "&ll", ">": "&gt", "≥": "&geq", "≫": "&gg", "·": "&middot", "×": "&times", "∘": "&compfn", "÷": "&div",
            "∖": "&setminus", "⊕": "&oplus", "∩": "&cap", "∪": "&cup", "⊂": "&subset", "⊃": "&supset", "∈": "&isin",
            "∉": "&notin", "∧": "&wedge", "∨": "&vee", "¬": "&not", "→": "&rightarrow", "⇒": "&Rightarrow", "⇔": "&iff",
            "↦": "&mapsto", "∢": "&angsph", "∑": "&sum", "∫": "&int", "∃": "&exist", "∀": "&forall"
        }
        self.__char_mapping__.update({c: f"&#{hex(ord(c))[1:]};" for c in {'×', '+', '-', '/', '≠', '≈', '∼', '≅', '∝',
                                                                           '≙', '<', '≤', '≪', '>', '≥', '≫', 'E', '∑',
                                                                           '∏', '∫', '∂', '∇', '⪰', '¬', '×', '∘', '÷',
                                                                           '∖', '⊕', '∩', '∪', '⊂', '⊃', '∈', '∉', '∧',
                                                                           '∨', '→', '⇒', '⇔', '↦', '∢', '⇔', '|', '→',
                                                                           '↦', '∘', '⨯', '‖', '⋅', '⊗', '∩', '≈', '*',
                                                                           '⇆', ',', '∈', ':', '.', '(', ')', '='}})
        self.__char_mapping_inverse__ = {v: k for k, v in self.__char_mapping__.items()}
        self.filter_rules = {"<mo>": re_compile(r"<mo rspace=\"\d+\.?\d?pt\"\s?>")}
        super(MathMLTokenizer, self).__init__(special_tokens)

    def feed(self, data: str):
        for tag, rule in self.filter_rules.items():
            candidates = rule.findall(data)
            for candidate in candidates:
                data = data.replace(candidate, tag)
        for k, v in self.__char_mapping__.items():
            data = data.replace(k, f"{self.__delimiter__}{v}{self.__delimiter__}")
        super().feed(data)

    def reconstruct(self, tokens: list[str]) -> str:
        reconstructed = self.__delimiter__.join(tokens) if "<math" in tokens[0] else f'<math> {self.__delimiter__.join(tokens)} </math>'
        for k, v in self.__char_mapping_inverse__.items():
            reconstructed = reconstructed.replace(k, v)
        return reconstructed


class LatexFormulaTokenizer(FormulaTokenizer):
    def __init__(self):
        self.keywords = [
            '=', '+', '-', '/', '{', '}', '(', ')', '[', ']', '<', '>', ',', ':', '*', '^', '|',
            '\\otimes', '\\sum', '\\cdots', '\\oplus', '\\int', '\\rfloor', '\\le', '\\nabla', '\\right',
            '\\cap', '\\mathrm', '\\times', '\\succeq', '\\in', '\\gef', '\\infty', '\\underset', '\\circ',
            '\\leftrightarrows', '\\mapsto', '\\left', '\\ast', '\\partial', '\\frac', '\\cdot', '\\lfloor',
            'left', '\\to', '\\Leftrightarrow', '\\sqrt', '\\mathbb', "\\notin", "\\subset", "\\cup", "\\cong",
            "\\neg", "\\ne", "\\stackrel", "\\wedge", "\\ge", "\\prod", "\\mathmr", "\\approx", "\\forall",
            "\\propto", "\\exists", "\\setminus", "\\div", "\\supset",

            "lim", "coth", "cot", "sech", "arcsin", "csc", "and", "tanh", "arctanh", "arccsc", "arccos", "arccot", "sinh",
            "cos", "sin", "cot", "ln", "coth", "arccosh", "arcsinh", "arccosh", "sec", "arcsec", "tan", "arccsch", "cosh",
            "csch", "arccoth", "E", "arcsech", "arctan"
        ]

        self.recovery_rules = {
            "cdot s": "cdots", "in fty": "infty", "in t": "int", "le ft": "left", "left right": "leftright", "ne g": "neg",
            "ID E NTIFI E R": "IDENTIFIER", "NUMB E R": "NUMBER", "s in": "sin", "arcs in": "arcsin",
            "sin h": "sinh", "s in h": "sinh", "cos h": "cosh", "tan h": "tanh", "cot h": "coth", "sec h": "sech", "csc h": "csch",
            "arc sin": "arcsin", "arc cos": "arccos", "arc tan": "arctan", "arc cot": "arccot", "arc sec": "arcsec", "arc csc": "arccsc",
            "arc sin h": "arcsinh", "arc cos h": "arccosh", "arc tan h": "arctanh", "arc cot h": "arccoth", "arc sec h": "arcsech", "arc csc h": "arccsch",
            "arc sinh": "arcsinh", "arc cosh": "arccosh", "arc tanh": "arctanh", "arc coth": "arccoth", "arc sech": "arcsech", "arc csch": "arccsch",
            "arc  sinh": "arcsinh", "arc  cosh": "arccosh", "arc  tanh": "arctanh", "arc  coth": "arccoth", "arc  sech": "arcsech", "arc  csch": "arccsch",
            "arcsin h": "arcsinh", "arccos h": "arccosh", "arctan h": "arctanh", "arccot h": "arccoth", "arcsec h": "arcsech", "arccsc h": "arccsch",
        }

        special_tokens = {
            "\\lambda": "lambda_token",
            "\\delta": "delta_token",
            "\\epsilon": "epsilon_token",
            "\\psi": "psi_token",
            "\\Phi": "phi_token",
            "\\sigma": "sigma_token",
            "\\tau": "tau_token",
            "\\theta": "theta_token",
            "\\Theta": "theta_token2",
            "\\Delta": "delta_token2",
            "\\alpha": "alpha_token",
            "\\beta": "beta_token",
            "\\varphi": "phi_token2",
            "\\gamma": "gamma_token",
            "\\Lambda": "lambda_token2",
            "\\mu": "mu_token",
            "\\pi": "pi_token",
            "\\rho": "rho_token",
            "\\Gamma": "gamma_token2",
            "\\eta": "eta_token",
            "\\xi": "xi_token",
            "\\zeta": "zeta_token",
            "\\infty": "infinity_token",
            "\\cdots": "cdots_token",
            "interval": "interval_token",
        }

        self.__backslash_filter__ = re_compile(r"[a-zA-Z0-9]\\")
        self.keywords.extend(special_tokens.keys())
        super(LatexFormulaTokenizer, self).__init__(special_tokens)

    def feed(self, data: str):
        for k in self.keywords:
            data = data.replace(k, f"{self.__delimiter__}{k}{self.__delimiter__}")
        data = data.replace(self.__delimiter__ * 2, self.__delimiter__)
        while any([k in data for k in self.recovery_rules]):
            for k, v in [(k, v) for k, v in self.recovery_rules.items() if k in data]:
                data = data.replace(k, v)
        for case in self.__backslash_filter__.findall(data):
            data = data.replace(case, f"{case[0]}{self.__delimiter__}\\")
        data = data.replace("\\ \\", "\\\\")
        super().feed(data)

    def reconstruct(self, tokens: list[str]) -> str:
        return "".join(tokens)


class ASCIIMathTokenizer(FormulaTokenizer):
    def __init__(self):
        exclusions = (
            'xx', 'delta', 'sum', 'int', 'epsilon', 'del', 'frac', 'hArr', 'harrs', 'cdots', 'ox', 'text',
            'rho', 'nn', 'grad', 'interval', 'in', 'sqrt', 'lim', 'underset', 'root', 'sub', 'stackrel',
            'not', 'prod', 'and', 'AA', 'prop', 'ln', 'EE', 'uu', 'sup',
        )

        to_split = (
            'E',
            "sin", "cos", "tan", "cot", "sec", "csc",
            "sinh", "cosh", "tanh", "coth", "sech", "csch",
            "arcsin", "arccos", "arctan", "arccot", "arcsec", "arccsc",
            "arcsinh", "arccosh", "arctanh", "arccoth", "arcsech", "arccsch",
        )

        special_tokens = {
            "lambda": "lambda_token",
            "delta": "delta_token",
            "epsilon": "epsilon_token",
            "psi": "psi_token",
            "Phi": "phi_token",
            "sigma": "sigma_token",
            "tau": "tau_token",
            "theta": "theta_token",
            "Delta": "delta_token2",
            "alpha": "alpha_token",
            "beta": "beta_token",
            "varphi": "phi_token2",
            "gamma": "gamma_token",
            "Lambda": "lambda_token2",
            "mu": "mu_token",
            "pi": "pi_token",
            "rho": "rho_token",
            "Gamma": "gamma_token2",
            "eta": "eta_token",
            "xi": "xi_token",
            "Theta": "theta_token2",
            "zeta": "zeta_token",
            "oo": "infinity_token",
            "cdots": "cdots_token",
            "interval": "interval_token",
        }

        self.keywords = ["(", ")", "{", "}", "[", "]", "+", "-", "*", "/", "^^", "^", "=", "@", "->", ",", ":", "//", "<", ">",
                         '~', '~=', '~=', '-:', "\\\\", "|", "!", "~~", "|->",]

        self.recovery_rules = {
            "del ta": "delta", "e psi lon": "epsilon", "th eta": "theta", "Th eta": "Theta", "z eta": "zeta",
            "r oo t": "root", "r o o t": "root", "in t": "int", "intext": "in  text", "int ext": "in text",
            "int erval": "interval", "in terval": "interval", "b eta": "beta", "E E": "EE", "^ ^": "^^",
            "| ->": "|->", "| - >": "|->", "|- >": "|->", '~ =': '~=', '~ ~': '~~', '- :': '-:', "- >": "->",
            "ID E NTIFI E R": "IDENTIFIER", "NUMB E R": "NUMBER", "s in": "sin", "arcs in": "arcsin",
            "sin h": "sinh", "s in h": "sinh", "cos h": "cosh", "tan h": "tanh", "cot h": "coth", "sec h": "sech", "csc h": "csch",
            "arc sin": "arcsin", "arc cos": "arccos", "arc tan": "arctan", "arc cot": "arccot", "arc sec": "arcsec", "arc csc": "arccsc",
            "arc sin h": "arcsinh", "arc cos h": "arccosh", "arc tan h": "arctanh", "arc cot h": "arccoth", "arc sec h": "arcsech", "arc csc h": "arccsch",
            "arc sinh": "arcsinh", "arc cosh": "arccosh", "arc tanh": "arctanh", "arc coth": "arccoth", "arc sech": "arcsech", "arc csch": "arccsch",
            "arc  sinh": "arcsinh", "arc  cosh": "arccosh", "arc  tanh": "arctanh", "arc  coth": "arccoth", "arc  sech": "arcsech", "arc  csch": "arccsch",
            "arcsin h": "arcsinh", "arccos h": "arccosh", "arctan h": "arctanh", "arccot h": "arccoth", "arcsec h": "arcsech", "arccsc h": "arccsch",
        }
        self.keywords.extend(exclusions)
        self.keywords.extend(to_split)
        self.keywords.extend(special_tokens.keys())

        super(ASCIIMathTokenizer, self).__init__(special_tokens, exclusions)

    def feed(self, data: str):
        for op in self.keywords:
            data = data.replace(op, f" {op} ")
        data = data.replace(self.__delimiter__ * 2, self.__delimiter__)
        while any([k in data for k in self.recovery_rules]):
            for k, v in [(k, v) for k, v in self.recovery_rules.items() if k in data]:
                data = data.replace(k, v)
        super().feed(data)

    def reconstruct(self, tokens: list[str]) -> str:
        result = "".join(tokens)
        return result


class SVGTokenizer(WhitespaceTokenizer):
    def __init__(self):
        super(SVGTokenizer, self).__init__(delimiter="__")

    def feed(self, data):
        if "<svg" not in data:
            data = Path(data).read_text()
        data = data.replace("><", f">{self.__delimiter__}<")
        self.buffer = data.split(self.__delimiter__)

    def reconstruct(self, tokens: list[str]) -> str:
        return "".join(tokens)
