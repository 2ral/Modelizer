from modelizer.tokenizer.generic import WhitespaceTokenizer


class PythonExpressionTokenizer(WhitespaceTokenizer):
    def __init__(self):
        super(PythonExpressionTokenizer, self).__init__(placeholders=("FLOAT", "INTEGER"))

    def feed(self, data: str):
        data = data.replace(" ** 1 ", " **1 ")
        while "))" in data:
            data = data.replace("))", ") )")
        data = [t.strip() for t in data.split(self.__delimiter__) if t.strip()]
        if self.__mask_mapping__:
            operands = ['+', '%', '[-', '/', '**', '//', ')', ']', '))', '[', '-', '*', ',']
            for i, v in enumerate(data):
                if v in operands or "math." in v:
                    continue
                elif '.' in v and v.replace('.', '', 1).isdigit():
                    tag = self.__mask_token__(v, self.__placeholders__[0])
                elif v.isdigit():
                    tag = self.__mask_token__(v, self.__placeholders__[1])
                else:
                    continue
                data[i] = tag
        self.buffer = data

    def reconstruct(self, tokens: list[str]) -> str:
        result = super().reconstruct(tokens)
        result = result.replace(" **1 ", " ** 1 ")
        while ") )" in result:
            result = result.replace(") )", "))")
        return result


class LatexExpressionTokenizer(WhitespaceTokenizer):
    def __init__(self):
        super(LatexExpressionTokenizer, self).__init__(placeholders=("FLOAT", "INTEGER"))
        self.__keywords__ = ["[", "|", "}", "`", ",", "{", "^", "]", "-", "!", "+", ")", "(", "'", "\\space", "\\lfloor", "\\rfloor"]

    def feed(self, data: str):
        data = data.replace("\\", f"{self.__delimiter__}\\")
        data = data.replace("^", f"{self.__delimiter__}^")
        data = data.replace("-", f"-{self.__delimiter__}")
        for kw in self.__keywords__:
            data = data.replace(kw, f"{self.__delimiter__}{kw}{self.__delimiter__}")
        for rule in self.__filter_rules__.values():
            placeholders = rule.findall(data)
            placeholders.sort(reverse=True)
            for pl in placeholders:
                data = data.replace(pl, f"{self.__delimiter__}{pl}{self.__delimiter__}".replace("_", "_!_!_"))
        data = data.replace("_!_!_", "_")
        data = data.replace("\log_ { 10 }", "\log_{10}").replace("\log_ { 2 }", "\log_{2}")
        data = [t.strip() for t in data.split(self.__delimiter__) if t.strip()]
        if self.__mask_mapping__:
            for i, v in enumerate(data):
                if '.' in v and v.replace('.', '', 1).isdigit():
                    tag = self.__mask_token__(v, self.__placeholders__[0])
                elif v.isdigit():
                    tag = self.__mask_token__(v, self.__placeholders__[1])
                else:
                    continue
                data[i] = tag
        self.buffer = data

    def reconstruct(self, tokens: list[str]) -> str:
        return "".join(tokens)
