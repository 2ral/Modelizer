import sqlparse
from shlex import shlex as shlex_shlex
from modelizer.tokenizer.generic import WhitespaceTokenizer, MappingPolicy


class QueryTokenizer(WhitespaceTokenizer):
    def __init__(self, delimiter: str = " ", placeholders: tuple = ()):
        super(QueryTokenizer, self).__init__(delimiter=delimiter, placeholders=placeholders)
        self.__leftover_mapping__ = {}
        self.__arithmetic_keys__ = ("+", "-", "/", "*", "%")
        self.temp_value_counter = 0
        self.arithmetic_index = -1
        self.arithmetic_buffer = []
        self.arithmetic_mapping = {}

    def mapped_feed(self, data: str) -> tuple[list[str], dict[str, str]]:
        self.__leftover_mapping__.clear()
        tokens, mapping = super().mapped_feed(data)
        for k, v in self.__leftover_mapping__.items():
            if k in mapping:
                del mapping[k]
            for i, t in enumerate(tokens):
                if t == k:
                    tokens[i] = v
                elif k in t:
                    tokens[i] = tokens[i].replace(k, v)
        return tokens, mapping

    def __mask_token__(self, token: str, placeholder: str) -> str:
        match self.__mapping_policy__:
            case MappingPolicy.SIMPLIFIED:
                tag = placeholder
                self.__mappings__[placeholder][f"{placeholder}_{len(self.__mappings__[placeholder]) + 1}"] = token
            case MappingPolicy.OPTIMIZING:
                if token in self.__inverse_mapping__:
                    tag = self.__inverse_mapping__[token]
                    if placeholder not in tag and "SELECT" not in tag:
                        old_tag = tag
                        tag = f"{placeholder}_{len(self.__mappings__[placeholder]) + 1}"
                        self.__mappings__[placeholder][tag] = token
                        self.__inverse_mapping__[token] = tag
                        self.__leftover_mapping__[old_tag] = tag
                else:
                    tag = f"{placeholder}_{len(self.__mappings__[placeholder]) + 1}"
                    self.__mappings__[placeholder][tag] = token
                    self.__inverse_mapping__[token] = tag
            case _:
                tag = f"{placeholder}_{len(self.__mappings__[placeholder]) + 1}"
                self.__mappings__[placeholder][tag] = token
                self.__inverse_mapping__[token] = tag
        return tag

    def __flush_arithmetic_buffer__(self):
        if len(self.arithmetic_buffer):
            key = f"TEMP_VALUE_{self.temp_value_counter}" if self.temp_value_counter > 0 else self.arithmetic_index
            value_data = " ".join(self.arithmetic_buffer) if len(self.arithmetic_buffer) > 1 else self.arithmetic_buffer[0]
            self.arithmetic_mapping[key] = self.__mask_token__(value_data, self.__placeholders__[3])  # Value
            self.arithmetic_buffer.clear()
            self.arithmetic_index = -1


class SQLTokenizer(QueryTokenizer):
    def __init__(self):
        super(SQLTokenizer, self).__init__(placeholders=("TABLE", "ALIAS", "COLUMN", "VALUE"))
        self.__table_keywords__ = ("FROM", "JOIN", "INTO", "UPDATE")
        self.__ignored_keywords__ = ("AVG", "SUM", "MIN", "MAX", "COUNT")
        self.__exclusions__ = ("type", "date", "week", "location", "result", "year", "position", "character", "role",
                               "events", "time", "catalog", "class", "number", "host", "aggregate")

    def feed(self, data):
        data = self.parse(data) if self.__mask_mapping__ else data
        for kw in self.__ignored_keywords__:
            data = data.replace(kw, f"{kw}{self.__delimiter__}")
        data = data.replace("..", "__DOUBLE_DOT_PLACEHOLDER__").replace('.', ' . ').replace("__DOUBLE_DOT_PLACEHOLDER__", "..")
        data = data.replace(",", f" ,").replace(";", " ;")
        data = data.replace('(', '( ').replace(')', ' )')
        self.buffer = [t.strip() for t in data.split(self.__delimiter__) if len(t.strip())]

    def parse(self, data: str):
        last_keyword = None

        def __parse__(t):
            nonlocal last_keyword

            for i in range(len(t)):
                if self.arithmetic_index != -1 and not (t[i].ttype in [sqlparse.tokens.Literal.Number.Integer, sqlparse.tokens.Literal.Number.Float, sqlparse.tokens.Whitespace] or t[i].value in self.__arithmetic_keys__):
                    self.__flush_arithmetic_buffer__()
                if isinstance(t[i], sqlparse.sql.TokenList):
                    t[i].tokens = __parse__(t[i].tokens)
                else:
                    if t[i].ttype in [sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DML] and t[i].value.lower() not in self.__exclusions__:
                        t[i].value = t[i].value.upper()
                        last_keyword = t[i].value
                    elif t[i].ttype in [sqlparse.tokens.Name, sqlparse.tokens.Wildcard] or t[i].value.lower() in self.__exclusions__:
                        if last_keyword in self.__table_keywords__:
                            last_keyword = None
                            t[i].value = self.__mask_token__(t[i].value, self.__placeholders__[0])  # Table
                        elif last_keyword == "AS":
                            last_keyword = None
                            t[i].value = self.__mask_token__(t[i].value, self.__placeholders__[1])  # Alias
                        elif i + 1 < len(t) and t[i + 1].value == ".":
                            t[i].value = self.__mask_token__(t[i].value, self.__placeholders__[0])  # Table
                        elif t[i].value in self.__ignored_keywords__:
                            continue
                        else:
                            t[i].value = self.__mask_token__(t[i].value, self.__placeholders__[2])  # Column
                    elif t[i].ttype == sqlparse.tokens.Literal.String.Symbol or "'" in t[i].value:
                        t[i].value = self.__mask_token__(t[i].value, self.__placeholders__[3])  # Value
                    elif t[i].ttype in [sqlparse.tokens.Literal.Number.Integer, sqlparse.tokens.Literal.Number.Float] or t[i].value in self.__arithmetic_keys__:
                        self.arithmetic_buffer.append(t[i].value)
                        if self.arithmetic_index == -1:
                            self.arithmetic_index = i
                            self.temp_value_counter += 1
                            t[i].value = f"TEMP_VALUE_{self.temp_value_counter}"
                        else:
                            t[i].value = ""
            return t

        parsed_str = str(" ".join([str(sub_q) for sub_q in __parse__(sqlparse.parse(data))]))
        self.__flush_arithmetic_buffer__()
        if len(self.arithmetic_mapping):
            for k, v in self.arithmetic_mapping.items():
                parsed_str = parsed_str.replace(k, v)
            self.arithmetic_mapping.clear()
        return parsed_str

    def reconstruct(self, tokens: list[str]) -> str:
        result = self.__delimiter__.join(tokens)
        result = result.replace(" . ", ".").replace(" ,", ",").replace(" ;", ";")
        result = result.replace("( ", "(").replace(" )", ")")
        for kw in self.__ignored_keywords__:
            result = result.replace(f"{kw}{self.__delimiter__}", kw)
        return result

    def mapped_reconstruct(self, tokens: list[str], mapping: dict[str, str]) -> str:
        match self.__mapping_policy__:
            case MappingPolicy.SIMPLIFIED:
                result = self.limited_reconstruct(tokens, mapping)
            case _:
                result = self.reconstruct(tokens)
                for t in mapping:
                    result = result.replace(t, mapping[t])
        return result


class KQLTokenizer(QueryTokenizer):
    def __init__(self):
        super(KQLTokenizer, self).__init__(placeholders=("TABLE", "ALIAS", "COLUMN", "VALUE"))
        self.__keywords__ = ["project", "distinct", "any", "by", "join", "$left.", "$right.", "where", "or", "and",
                             "between", '!between', "in", "!in", "<", ">", "<=", ">=", "==", "!=",
                             "extend", "min", "max", "count", "sum", "contains", "avg"]
        self.__ignored__ = ["order", ",", "|", "(", ")", "..", "asc", "desc", "on", "summarize", "not",
                            "kind=left", "kind=right", "kind=outer", "kind=inner", "="]
        self.__column_clause__ = ["distinct", "any", "by", "$left.", "$right.",
                                  "where", "or", "and", "extend", "min", "max", "count", "sum", "avg"]
        self.__value_clause__ = ["between", '!between', "in", "!in", "<", ">", "<=", ">=", "==", "!=", "contains"]

    def __mask_token__(self, token: str, placeholder: str) -> str:
        placeholder = "TABLE" if len(token) and token[-1] == "." else placeholder
        return super(KQLTokenizer, self).__mask_token__(token, placeholder)

    def feed(self, data: str):
        def __strip_and_recover__(token: str) -> str:
            token = token.strip()
            if token not in self.__ignored__:
                for bracket_char in ["(", ")"]:
                    if bracket_char in token:
                        token = token.replace(f" {bracket_char} ", bracket_char)
            return token
        data = data.replace('(', ' ( ').replace(')', ' ) ')
        result = []
        for row in data.split('\n'):
            lexer = shlex_shlex(row)
            lexer.quotes = "'"
            lexer.whitespace_split = True
            for t in list(lexer):
                if '.' in t and all([c not in t for c in ['..', "'", '"', ':', '%']]) and not any(char.isdigit() for char in t) and len(t.split(".", 1)) == 2:
                    t_left, t_right = t.split('.', 1)
                    if t_left:
                        result.append(__strip_and_recover__(t_left))
                    result.append('.')
                    if t_right:
                        result.append(__strip_and_recover__(t_right))
                elif ',' in t and "'" not in t:
                    t_left, t_right = t.split(',', 1)
                    if t_left:
                        result.append(__strip_and_recover__(t_left))
                    result.append(',')
                    if t_right:
                        result.append(__strip_and_recover__(t_right))
                else:
                    t = __strip_and_recover__(t)
                    if t:
                        result.append(t)
        result = [t for t in result if len(t)]

        if self.__mask_mapping__:
            last_keyword = None
            result[0] = self.__mask_token__(result[0], self.__placeholders__[0])  # Table
            result_slice = result[1:]
            for i, v in enumerate(result_slice):
                if v in self.__keywords__:
                    last_keyword = v
                    self.__flush_arithmetic_buffer__()
                    continue
                elif v in self.__ignored__:
                    self.__flush_arithmetic_buffer__()
                    continue
                elif v == "*":
                    tag = self.__mask_token__(v, self.__placeholders__[2])  # Column
                elif last_keyword == "join":
                    tag = self.__mask_token__(v, self.__placeholders__[0])  # Table
                elif last_keyword == "project":
                    if v[-1] == '.':
                        tag = self.__mask_token__(v[:-1], self.__placeholders__[0]) + '.'  # Table
                    else:
                        next_char_id = i + 1
                        if next_char_id < len(result_slice) and result_slice[next_char_id] == "=":
                            p = self.__placeholders__[1]  # Alias
                        else:
                            p = self.__placeholders__[2]  # Column
                        tag = self.__mask_token__(v, p)
                elif last_keyword in self.__column_clause__:
                    if v[-1] == '.':
                        tag = self.__mask_token__(v[:-1], self.__placeholders__[0]) + '.'  # Table
                    else:
                        tag = self.__mask_token__(v, self.__placeholders__[2])  # Column
                elif last_keyword in self.__value_clause__:
                    if v in self.__arithmetic_keys__ or v.isdigit() or v.replace(".", "").isdigit():
                        self.arithmetic_buffer.append(v)
                        if self.arithmetic_index == -1:
                            self.arithmetic_index = i + 1
                        tag = ""
                    else:
                        tag = self.__mask_token__(v, self.__placeholders__[3])  # Value
                else:
                    self.__flush_arithmetic_buffer__()
                    continue
                result[i + 1] = tag
        self.__flush_arithmetic_buffer__()
        if len(self.arithmetic_mapping):
            for k, v in self.arithmetic_mapping.items():
                result[k] = v
            self.arithmetic_mapping.clear()
        self.buffer = [t for t in result if len(t)]

    def reconstruct(self, tokens: list[str]) -> str:
        result = " ".join([f"\n {t}" if t in ["and", "or"] else t for t in tokens])
        result = result.replace(" ( ", "(").replace(" ) ", ")").replace(" |", "|")
        result = result.replace("|", "\n|").replace(". ", ".")
        result = result.replace(")any", ") any").replace("where(", "where (")
        result = result.replace(")by", ") by").replace("by(", "by (").replace(' .', '.')
        result = result.replace(")between", ") between").replace("between(", "between (")
        result = result.replace("kind=right(", "kind=right (").replace("kind=left(", "kind=left (")
        result = result.replace(")in", ") in").replace("in(", "in (").replace("..", " .. ")
        result = result.replace(")and", ") and").replace("and(", "and (")
        result = result.replace(")or", ") or").replace("or(", "or (")
        result = result.replace("(not", "( not").replace("not(", "not (").replace(")not", ") not")
        result = result.replace("outer(", "outer (").replace("inner(", "inner (")
        result = result.replace(")on", ") on").replace("min (", "min(")
        result = result.replace(" and ", "  and ").replace(" or ", "  or ").replace(" ,", ",")
        result = result.replace("(( ", "((").replace(")) ", "))").replace("((not", "(( not")
        result = result.replace("( ", "(").replace(" )", ")").replace("(not", "( not")
        return result

    def mapped_reconstruct(self, tokens: list[str], mapping: dict[str, str]) -> str:
        match self.__mapping_policy__:
            case MappingPolicy.SIMPLIFIED:
                result = self.limited_reconstruct(tokens, mapping)
            case _:
                for i, v in enumerate(tokens):
                    if v in mapping and "TEXT" not in v:
                        tokens[i] = mapping[v]
                result = self.reconstruct(tokens)
                for t in mapping:
                    result = result.replace(t, mapping[t])
        return result
