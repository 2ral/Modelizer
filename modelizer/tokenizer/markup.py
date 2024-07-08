from pandoc.types import *
from html.parser import HTMLParser
from modelizer.tokenizer.generic import AbstractTokenizer, MappingPolicy


class MarkdownTokenizer(AbstractTokenizer):
    def __init__(self):
        super(MarkdownTokenizer, self).__init__(placeholders=("URL", "TEXT"))
        self.__element_level__ = 0
        self.__current_block__ = None
        self.__string_buffer__ = list()

    def feed(self, data: str):
        data = data.split("\n\n")
        for d in data:
            self.__current_block__ = d
            doc = pandoc.read(self.__current_block__, format="markdown")
            # Element-wise parsing
            for e in doc[1]:
                self.__parse__(e)
            if len(d) and d[-1] == "\n":
                x = self.buffer[-1]
                if x[-1] != "\n":
                    if self.__mask_mapping__ and self.__mapping_policy__ != MappingPolicy.SIMPLIFIED:
                        if "TEXT" not in x:
                            self.buffer.append('\n')
                        elif self.__mappings__["TEXT"][x][-1] != "\n":
                            self.buffer.append('\n')
                    else:
                        self.buffer.append('\n')
            if self.buffer[-1] != "\n\n":
                self.buffer.append("\n\n")
        while self.buffer[-1] == "\n\n":
            del self.buffer[-1]

    def reconstruct(self, tokens: list[str]) -> str:
        result = "".join(tokens)
        result = result.replace("\n   -", "\n-")
        return result

    def __flush_string_buffer__(self):
        if len(self.__string_buffer__):
            r_string = "".join(self.__string_buffer__)
            if self.__element_level__ == 2:
                r_string = r_string.replace("\n", "\n    ")
            self.__string_buffer__.clear()
            self.buffer.append(self.__mask_token__(r_string, self.__placeholders__[1]) if self.__mask_mapping__ else r_string)

    def __parse__(self, e):
        if isinstance(e, Str):
            if self.__mask_mapping__:
                self.__string_buffer__.append(e[0])
            else:
                self.buffer.append(e[0])
        elif isinstance(e, Space):
            if self.__mask_mapping__:
                self.__string_buffer__.append(" ")
            else:
                self.buffer.append(" ")
        elif isinstance(e, SoftBreak):
            if self.__element_level__ >= 3:
                if ">" in self.buffer[-1]:
                    self.buffer[-1] = " "
                else:
                    self.buffer.append('\n')
                    self.buffer.extend(['> ' for _ in range(self.__element_level__ - 2)])
            else:
                self.__string_buffer__.append("\n")
        elif isinstance(e, Quoted):
            quote_char = "'" if isinstance(e[0], SingleQuote) else '"'
            if self.__mask_mapping__:
                self.__string_buffer__.append(quote_char)
            else:
                self.buffer.append(quote_char)
            for inner_e in e[1]:
                self.__parse__(inner_e)
            if self.__mask_mapping__:
                self.__string_buffer__.append(quote_char)
            else:
                self.buffer.append(quote_char)
        else:
            self.__flush_string_buffer__()
            if isinstance(e, Header):
                last_element_level = self.__element_level__
                if self.__element_level__ < 1:
                    self.__element_level__ = 1
                self.buffer.append("#" * e[0] + " ")
                for inner_e in e[2]:
                    self.__parse__(inner_e)
                self.__flush_string_buffer__()
                if self.__element_level__ == 1:
                    self.buffer.append("\n")
                self.__element_level__ = last_element_level
            elif isinstance(e, Para):
                last_element_level = self.__element_level__
                if self.__element_level__ < 1:
                    self.__element_level__ = 1
                for inner_e in e[0]:
                    self.__parse__(inner_e)
                self.__flush_string_buffer__()
                # if self.element_level == 1:
                #     self.buffer.append("\n")
                self.__element_level__ = last_element_level
            elif isinstance(e, BlockQuote):
                last_element_level = self.__element_level__
                if self.__element_level__ >= 3:
                    self.__element_level__ += 1
                    self.buffer.pop()
                else:
                    self.__element_level__ = 3
                for inner_e in e[0]:
                    self.buffer.extend(['> ' for _ in range(self.__element_level__ - 2)])
                    self.__parse__(inner_e)
                    self.__flush_string_buffer__()
                    self.buffer.append("\n>\n")
                if self.__element_level__ >= 3:
                    self.buffer.pop()
                    # if self.element_level == 3:
                    #     self.buffer.append("\n\n")
                self.__element_level__ = last_element_level
            elif isinstance(e, OrderedList):
                last_element_level = self.__element_level__
                if self.__element_level__ < 2:
                    self.__element_level__ = 2
                for inner_e in e[1]:
                    self.buffer.append("1.   ")
                    for inner_inner_e in inner_e:
                        self.__parse__(inner_inner_e)
                    self.__flush_string_buffer__()
                    self.buffer.append("\n")
                # if self.element_level == 2:
                #     self.buffer.append("\n")
                self.__element_level__ = last_element_level
            elif isinstance(e, BulletList):
                last_element_level = self.__element_level__
                if self.__element_level__ < 2:
                    self.__element_level__ = 2
                for inner_e in e[0]:
                    self.buffer.append("-   ")
                    for inner_inner_e in inner_e:
                        self.__parse__(inner_inner_e)
                    self.__flush_string_buffer__()
                    self.buffer.append("\n")
                # if self.element_level == 2:
                #     self.buffer.append("\n")
                self.__element_level__ = last_element_level
            elif isinstance(e, CodeBlock):
                self.buffer.append("    ")
                masked = self.__mask_token__(e[1].replace("\n", "\n    "), self.__placeholders__[1]) if self.__mask_mapping__ else e[1]
                self.buffer.append(masked)
            elif isinstance(e, Code):
                self.buffer.append("`")
                self.buffer.append(self.__mask_token__(e[1], self.__placeholders__[1]) if self.__mask_mapping__ else e[1])
                self.buffer.append("`")
            elif isinstance(e, Emph):
                self.buffer.append("_")
                for inner_e in e[0]:
                    self.__parse__(inner_e)
                self.__flush_string_buffer__()
                self.buffer.append("_")
            elif isinstance(e, Strong):
                self.buffer.append("**")
                for inner_e in e[0]:
                    self.__parse__(inner_e)
                self.__flush_string_buffer__()
                self.buffer.append("**")
            elif isinstance(e, Plain):
                for inner_e in e[0]:
                    self.__parse__(inner_e)
                self.__flush_string_buffer__()
            elif isinstance(e, Link):
                self.buffer.append("[")
                if ("<" in self.__current_block__ and ">" in self.__current_block__ and isinstance(e[1][0], Str)
                        and (e[1][0][0].startswith("http") or e[1][0][0].startswith("URL"))):
                    self.buffer.append(self.__mask_token__(e[1][0][0], self.__placeholders__[0]) if self.__mask_mapping__ else e[1][0][0])
                else:
                    for inner_e in e[1]:
                        self.__parse__(inner_e)
                    self.__flush_string_buffer__()
                self.buffer.append("](")
                self.buffer.append(self.__mask_token__(e[2][0], self.__placeholders__[0]) if self.__mask_mapping__ else e[2][0])
                self.buffer.append(")")
            elif isinstance(e, LineBreak):
                self.buffer.append("  \n")
            elif isinstance(e, HorizontalRule):
                self.buffer.append("———")
            elif isinstance(e, Cite):
                self.__parse__(e[0][0])
                for inner_e in e[1]:
                    self.__parse__(inner_e)
            elif isinstance(e, Citation):
                if self.__mask_mapping__:
                    self.__mask_token__(e[0], self.__placeholders__[1])
            else:
                print(f"Unknown element: {e} | Type: {type(e)}")
                self.buffer.append("<UNKNOWN>")


class HTMLTokenizer(AbstractTokenizer, HTMLParser):
    def __init__(self):
        self.special_token_mapping = {" ": "__SPACE_PLACEHOLDER__", "\n": "__NEWLINE_PLACEHOLDER__",
                                      '“': "__DOUBLE_QUOTE_PLACEHOLDER_LEFT__", '”': "__DOUBLE_QUOTE_PLACEHOLDER_RIGHT__",
                                      "‘": "__SINGLE_QUOTE_PLACEHOLDER_LEFT__", "’": "__SINGLE_QUOTE_PLACEHOLDER_RIGHT__"}
        self.special_token_inverse_mapping = {v: k for k, v in self.special_token_mapping.items()}
        super(AbstractTokenizer, self).__init__()
        super(HTMLParser, self).__init__()
        super(HTMLTokenizer, self).__init__(placeholders=("URL", "TEXT"))

    def feed(self, data: str):
        HTMLParser.feed(self, data)

    def reconstruct(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def handle_starttag(self, tag, attrs):
        match tag:
            case "br":
                self.buffer.append("<br />")
            case "hr":
                parsed_attrs = [f'{attr[0]}="{attr[1]}"' for attr in attrs if attr[0] != "id"]
                if len(parsed_attrs):
                    self.buffer.append("<hr ")
                    self.buffer.extend(parsed_attrs)
                    self.buffer.append("/>")
                else:
                    self.buffer.append("<hr />")
            case "a":
                parsed_attrs = list()
                for attr in attrs:
                    if attr[0] == "href":
                        mask = self.__mask_token__(attr[1], self.__placeholders__[0]) if self.__mask_mapping__ else attr[1]
                        parsed_attrs.append(f'{attr[0]}="{mask}"')
                    elif attr[0] != "id":
                        parsed_attrs.append(f'{attr[0]}="{attr[1]}"')
                if len(parsed_attrs):
                    self.buffer.append(f"<a ")
                    self.buffer.extend(parsed_attrs)
                    self.buffer.append(">")
                else:
                    self.buffer.append(f"<a>")
            case "span":
                parsed_attrs = list()
                for attr in attrs:
                    if attr[0] == "data-cites":
                        mask = self.__mask_token__(attr[1], self.__placeholders__[1]) if self.__mask_mapping__ else attr[1]
                        parsed_attrs.append(f'{attr[0]}="{mask}"')
                    elif attr[0] != "id":
                        parsed_attrs.append(f'{attr[0]}="{attr[1]}"')
                if len(parsed_attrs):
                    self.buffer.append(f"<span ")
                    self.buffer.extend(parsed_attrs)
                    self.buffer.append(">")
                else:
                    self.buffer.append(f"<span>")

            case _:
                parsed_attrs = [f'{attr[0]}="{attr[1]}"' for attr in attrs if attr[0] != "id"]
                if len(parsed_attrs):
                    self.buffer.append(f"<{tag} ")
                    self.buffer.extend(parsed_attrs)
                    self.buffer.append(">")
                else:
                    self.buffer.append(f"<{tag}>")

    def handle_endtag(self, tag: str):
        if tag not in ["br", "hr"]:
            self.buffer.append(f"</{tag}>")

    def handle_data(self, data: str):
        if self.__mask_mapping__:
            self.buffer.append(self.__mask_token__(data, self.__placeholders__[1]) if data != "\n" else data)
        else:
            for k, v in self.special_token_mapping.items():
                data = data.replace(k, f" {v} ")
            self.buffer.extend([self.special_token_inverse_mapping[d] if d in self.special_token_inverse_mapping else d for d in data.split()])
