from string import ascii_letters

from fuzzingbook.Grammars import Grammar, opts, convert_ebnf_grammar, is_valid_grammar
from fuzzingbook.ProbabilisticGrammarFuzzer import is_valid_probabilistic_grammar


def convert_and_validate(grammar: dict):
    converted_grammar = convert_ebnf_grammar(grammar)
    if "{'prob':" in str(converted_grammar):
        assert is_valid_probabilistic_grammar(converted_grammar), "Invalid Probabilistic grammar"
    assert is_valid_grammar(converted_grammar), "Invalid Grammar"
    return converted_grammar


def __refine_sql_grammar__(sql_grammar: Grammar) -> Grammar:
    refined_grammar = sql_grammar.copy()
    refined_grammar.update({
        "<table_identifier>": ["<identifier>"],
        "<column_identifier>": ["<identifier>"],
        "<alias_identifier>": ["<identifier>"],
        "<identifier>": ["<letter><alphanumeric>"],
        "<alphanumeric>": ["<letter>", "<digit>"],
        "<value>": ["<text_value>", "<number_value>"],
        "<number_value>": ["<integer_value>", "<float_value>"],
        "<text_value>": ["\'<character>*\'"],
        "<integer_value>": ["<digit>+"],
        "<float_value>": ["<digit>+.<digit>+"],
        "<character>": ["<letter>", "<digit>", "<symbol>"],
        "<letter>": letters,
        "<digit>": digits,
        "<symbol>": ["+", "-", "*", "/", "=", "!=", "<", ">", "<=", ">=", ",", ".", ";", ":", "(", ")", "[", "]", "{", "}"],
    })
    return refined_grammar


digits = [str(d) for d in range(10)]
letters = [str(d) for d in ascii_letters]

__MARKDOWN_EBNF_GRAMMAR__: Grammar = {
    "<start>":
        ["<blocks>"],

    "<blocks>":
        [
            "<element>",
            ("<element>\n\n<blocks>", opts(prob=0.6)),
        ],

    "<element>":
        [
            "<heading>",
            "<paragraph_block>",
            ("<code_block>", opts(prob=0.05)),
            ("<blockquote>", opts(prob=0.1)),
            ("<horizontal_rules>", opts(prob=0.1)),
            ("\n1.   <paragraph_text><ordered_list_extra>?", opts(prob=0.1)),
            ("\n-   <paragraph_text><unordered_list_extra>?", opts(prob=0.1)),
        ],

    "<heading>":
        [
            "# <paragraph_text>",
            "## <paragraph_text>",
            "### <paragraph_text>",
            "#### <paragraph_text>",
            "##### <paragraph_text>",
            "###### <paragraph_text>",
            "<paragraph_text>\n<double_equal_sign>+",
            "<paragraph_text>\n<double_dash_sign>+",
        ],

    "<horizontal_rules>":
        ["<triple_dash_sign>+",],

    "<blockquote>":
        [
            "<greater_space_signs><paragraph_text>",
            "<greater_space_signs><paragraph_text>\n<greater_space_signs>\n<greater_space_signs><paragraph_text>",
            "<blockquote>\n<greater_space_signs>\n<nested_blockquote_bracket><paragraph_text>",
        ],

    "<nested_blockquote_bracket>":
        [
            "<greater_space_signs>",
            "<greater_space_signs><nested_blockquote_bracket>",
        ],

    "<ordered_list_extra>":
        [
            "\n1.   <paragraph_text><ordered_list_extra>*",
            # "\n1.   <paragraph_text><nested_ordered_list_extra>*",
        ],
    # "<nested_ordered_list_extra>":
    #     ["\n1.   <paragraph_text><nested_ordered_list_extra>*",],

    "<unordered_list_extra>":
        [
            "\n-   <paragraph_text><unordered_list_extra>*",
            # "\n-   <paragraph_text><nested_unordered_list_extra>*",
        ],
    # "<nested_unordered_list_extra>":
    #     ["\n-   <paragraph_text><nested_unordered_list_extra>*", ],

    "<code_block>":
        ["`<text>`", "```<text>```", "    <text>"],

    "<paragraph_block>":
        [
            "<paragraph_text>\n<paragraph_block>",
            ("<paragraph_text>\n", opts(prob=0.6)),
        ],

    "<paragraph_text>":
        [
            "<formatted_text>",
            ("<paragraph_text> <paragraph_text>", opts(prob=0.3)),
            ("[<formatted_text>](<url>)", opts(prob=0.2)),  # Link
            ("<code_block>", opts(prob=0.1)),
        ],
    "<formatted_text>":
        [
            ("<text>", opts(prob=0.3)),
            "<bold_text>",
            "<italic_text>",
            "<bold_italic_text>",
            ("<quoted_text>", opts(prob=0.1)),
        ],
    "<bold_text>":
        [
            "<double_asterisk><text><double_asterisk>",
            "<double_underscore><text><double_underscore>",
        ],
    "<italic_text>":
        [
            "<asterisk><text><asterisk>",
            "<underscore><text><underscore>",
        ],
    "<bold_italic_text>":
        [
            "<triple_asterisk><text><triple_asterisk>",
            "<triple_underscore><text><triple_underscore>",
            "<double_underscore><asterisk><text><asterisk><double_underscore>",
            "<double_asterisk><underscore><text><underscore><double_asterisk>",
        ],
    "<quoted_text>":
        [
            "\'<text>\'",
            "\"<text>\"",
        ],

    # Placeholders
    "<text>":
        ["TEXT"],
    "<url>":
        ["URL"],

    # Terminals
    "<greater_space_signs>":
        ["> "],
    "<asterisk>":
        ["*"],
    "<double_asterisk>":
        ["**"],
    "<triple_asterisk>":
        ["***"],
    "<underscore>":
        ["_"],
    "<double_underscore>":
        ["__"],
    "<triple_underscore>":
        ["___"],
    "<double_equal_sign>":
        ["=="],
    "<double_dash_sign>":
        ["——"],
    "<triple_dash_sign>":
        ["———"],
}

MARKDOWN_MARKERS = [__MARKDOWN_EBNF_GRAMMAR__["<text>"], __MARKDOWN_EBNF_GRAMMAR__["<url>"]]

__LINQ_EBNF_GRAMMAR__: Grammar = {
    "<start>":
        ["<query>"],

    "<query>":
        ["<from-clause> <query-clause>* <query-conclusion> <query-extension>?"],

    "<query-clause>":
        [
            "<from-clause>",
            "where <where_clause_extension>",
            "join <identifier> in <source> on <source>.<identifier> equals <source>.<identifier><group-join-extension>?",
            ("orderby <identifier> <ordering-direction>?", opts(prob=0.125)),
            ("orderby <source>.<identifier> <ordering-direction>?<orderby-clause-extension>*)", opts(prob=0.125)),
        ],

    "<from-clause>": ["from <identifier> in <source>"],

    "<query-conclusion>":
        [
            "select <identifier>",
            "group <identifier> by <identifier>.<identifier><group-join-extension>?",
        ],
    "<group-join-extension>":
        [" into <identifier>"],

    "<query-extension>":
        ["into <identifier> <query>",],

    "<where_clause_extension>":
        [
            "<boolean_expression>",
            "<boolean_expression> || <boolean_expression>",
            "<boolean_expression> && <boolean_expression>",
        ],
    "<boolean_expression>":
        ["<boolean_expression_left_argument> <comparison_operator> <expression>"],
    "<boolean_expression_left_argument>":
        [
            "<identifier>",
            "<source>.<identifier>",
            "<function>(<identifier>)",
            "<function>(<source>.<identifier>)",
        ],
    "<comparison_operator>":
        ["==", "!=", "<", "<=", ">", ">="],

    "<orderby-clause-extension>":
        [", orderby <identifier>.<identifier> <ordering-direction>?"],
    "<ordering-direction>": ["ascending", "descending"],

    # Placeholders
    "<source>":
        ["SOURCE"],
    "<identifier>":
        ["IDENTIFIER"],
    "<function>":
        ["FUNCTION"],
    "<expression>":
        ["EXPRESSION"],
}

LINQ_MARKERS = [__LINQ_EBNF_GRAMMAR__["<identifier>"], __LINQ_EBNF_GRAMMAR__["<source>"],
                __LINQ_EBNF_GRAMMAR__["<expression>"], __LINQ_EBNF_GRAMMAR__["<function>"]]

__SQL_EBNF_BASE_GRAMMAR__: Grammar = {
    "<start>":
        ["<select>;"],

    "<select>":
        ["SELECT <select_options> FROM <source><alias>?<join_clause>?<where_clause>?<group_by_clause>?<order_by_clause>?"],

    "<distinct>": ["DISTINCT "],

    "<select_options>":
        [
            "<distinct>?<wildcard>",
            "<distinct>?<table_identifier>.<column_identifier><alias>?",
            "<distinct>?<column_identifier><alias>?<select_extension>?",
            "<filter_function>(<column_identifier>)",
        ],
    "<select_extension>":
        [", <column_identifier><alias>?<select_extension>?"],

    "<filter_function>":
        ["COUNT", "SUM", "AVG", "MIN", "MAX",],

    "<alias>":
        [" AS <alias_identifier>"],

    "<join_clause>":
        ["<join_type> JOIN <join_target> ON <join_left_condition>.<column_identifier> = <join_right_condition>.<column_identifier>"],
    "<join_type>":
        [" INNER", " LEFT", " RIGHT", " FULL OUTER"],

    "<source>":
        ["<table_identifier>"],
    "<join_target>":
        ["<table_identifier>"],
    "<join_left_condition>":
        ["<table_identifier>"],
    "<join_right_condition>":
        ["<table_identifier>"],

    "<where_clause>":
        [" WHERE <where_clause_body><where_clause_extension>?"],
    "<where_clause_body>":
        [
            "<not_clause>?<column_identifier> <comparison_operator> <value>",
            "<column_identifier> <not_clause>?IN (<value_list>)",
            "<column_identifier> <not_clause>?BETWEEN <value> AND <value>",
        ],
    "<where_clause_extension>":
        [
            "<logical_operator><where_clause_body><where_clause_extension>?",
            "<logical_operator>(<where_clause_body><where_clause_extension>)",
        ],
    "<not_clause>":
        ["NOT "],
    "<logical_operator>":
        [" AND ", " OR "],

    "<group_by_clause>":
        [" GROUP BY <column_identifier><having_clause>?"],
    "<having_clause>":
        [" HAVING <function>(<column_identifier>) <comparison_operator> <value>"],
    "<function>":
        ["COUNT", "SUM", "AVG", "MIN", "MAX"],

    "<order_by_clause>":
        [" ORDER BY <order_by_criteria>"],
    "<order_by_criteria>":
        [
            "<column_identifier><ordering>?",
            "<column_identifier><ordering>?, <order_by_criteria>",
        ],
    "<ordering>":
        [" ASC", " DESC"],

    "<value_list>":
        [
            "<value>",
            "<value>, <value_list>",
            # "<select>"  # Temporary Simplification
        ],

    "<comparison_operator>":
        ["=", "<left_bracket><right_bracket>", "<left_bracket>", "<right_bracket>", "<left_bracket>=", "<right_bracket>=", "LIKE"],

    "<left_bracket>":
        ["<"],
    "<right_bracket>":
        [">"],
    "<wildcard>":
        ["*"],

    # PLACEHOLDERS
    "<table_identifier>": ["TABLE"],
    "<alias_identifier>": ["ALIAS"],
    "<column_identifier>": ["COLUMN"],
    "<value>": ["VALUE"],
}

SQL_MARKERS = [__SQL_EBNF_BASE_GRAMMAR__["<table_identifier>"], __SQL_EBNF_BASE_GRAMMAR__["<column_identifier>"],
               __SQL_EBNF_BASE_GRAMMAR__["<alias_identifier>"], __SQL_EBNF_BASE_GRAMMAR__["<value>"]]

__SQL_EBNF_EXTENDED_GRAMMAR__ = __SQL_EBNF_BASE_GRAMMAR__.copy()
__SQL_EBNF_EXTENDED_GRAMMAR__.update({
    "<start>": ["<query>;"],
    "<query>": ["<select>", "<update>", "<delete>", "<insert>"],
    "<update>": ["UPDATE <table_identifier> SET <set_clause><where_clause>?"],
    "<delete>": ["DELETE FROM <table_identifier><where_clause>?"],
    "<insert>": ["INSERT INTO <table_identifier> (<column_list>) VALUES ( <value_list> )"],
    "<column_list>": ["<column_identifier>", "<column_identifier>, <column_list>"],
    "<set_clause>": ["<column_identifier> = <value>", "<column_identifier> = <value>, <set_clause>"],
})

__SQL_REFINED_EBNF_BASE_GRAMMAR__ = __refine_sql_grammar__(__SQL_EBNF_BASE_GRAMMAR__)
__SQL_REFINED_EBNF_EXTENDED_GRAMMAR__ = __refine_sql_grammar__(__SQL_EBNF_EXTENDED_GRAMMAR__)

__mathml_mid_operations_alt__ = [
    "<mo>&ne;</mo>",  # ≠
    "<mo>&ap;</mo>",  # ≈
    "<mo>&sim;</mo>",  # ∼
    "<mo>&cong;</mo>",  # ≅
    "<mo>&propto;</mo>",  # ∝
    "<mo>&wedgeq;</mo>",  # ≙
    "<mo>&lt;</mo>",  # <
    "<mo>&leq;</mo>",  # ≤
    "<mo>&ll;</mo>",  # ≪
    "<mo>&gt;</mo>",  # >
    "<mo>&geq;</mo>",  # ≥
    "<mo>&gg;</mo>",  # ≫
    "<mo>&middot;</mo>",  # ·
    "<mo>&times;</mo>",  # ×
    "<mo>&compfn;</mo>",  # ∘
    "<mo>&div;</mo>",  # ÷
    "<mo>&setminus;</mo>",  # ∖
    "<mo>&oplus;</mo>",  # ⊕
    "<mo>&cap;</mo>",  # ∩
    "<mo>&cup;</mo>",  # ∪
    "<mo>&subset;</mo>",  # ⊂
    "<mo>&supset;</mo>",  # ⊃
    "<mo>&isin;</mo>",  # ∈
    "<mo>&notin;</mo>",  # ∉
    "<mo>&wedge;</mo>",  # ∧
    "<mo>&vee;</mo>",  # ∨
    "<mo>&not;</mo>",  # ¬
    "<mo>&rightarrow;</mo>",  # →
    "<mo>&Rightarrow;</mo>",  # ⇒
    "<mo>&iff;</mo>",  # ⇔
    "<mo>&mapsto;</mo>",  # ↦
    "<mo>&angsph;</mo>",  # ∢
]

__mathml_left_operations_alt__ = [
    "<mo>&sum;</mo>",  # ∑
    "<mo>&int;</mo>",  # ∫
    "<mo>&exist;</mo>",  # ∃
    "<mo>&forall;</mo>",  # ∀
]

mathml_main_operations = [f"<mo>&#{hex(ord(op))[1:]};</mo>" for op in {'×', '+', '-', '/'}]

mathml_comparison_operations = [f"<mo>&#{hex(ord(op))[1:]};</mo>" for op in {'≠', '≈', '∼', '≅', '∝', '≙',
                                                                             '<', '≤', '≪', '>', '≥', '≫'}]

mathml_left_operations = [f"<mo>&#{hex(ord(op))[1:]};</mo>" for op in {'E', '∑', '∏', '∫', '∂', '∇', '⪰', '¬'}]
mathml_left_operations.extend(__mathml_left_operations_alt__)

mathml_mid_operations = [f"<mo>&#{hex(ord(op))[1:]};</mo>" for op in {'×', '∘', '÷', '∖',
                                                                      '⊕', '∩', '∪', '⊂', '⊃', '∈', '∉',
                                                                      '∧', '∨', '→', '⇒', '⇔', '↦',
                                                                      '∢', '⇔', '|', '→', '↦', '∘', '⨯', '‖', '⋅',
                                                                      '⊗', '∩', '≈', '*',  '⇆', ',', '∈',   ':',  '.'}]

# removed
mathml_mid_operations.extend(__mathml_mid_operations_alt__)
mathml_factor_brackets = [f"<mo>&#{hex(ord(br[0]))[1:]};</mo><factor><mo>&#{hex(ord(br[1]))[1:]};</mo>" for br in
                          {('[', ']'), ('⌊', '⌋'), ('⌈', '⌉'), ('(', ')'), ('{', '}'), ('⟨', '⟩')}]

__MATHML_EBNF_GRAMMAR__: Grammar = {
    # "<left_operation>": mathml_left_operations,


    "<start>": ["<mathml>"],

    "<mathml>": ["<math><mi><identifier></mi><mo>&#x3d;</mo><mrow><expression></mrow></math>"],

    "<expression>":
        [
            "<mrow><term></mrow><operation><mrow><expression></mrow>",
            "<term>",
        ],

    "<operation>": [
        "<base_operation>",
        "<comparison_operation>",
        "<mid_operation>",
    ],

    "<base_operation>": mathml_main_operations,
    "<comparison_operation>": mathml_comparison_operations,
    "<mid_operation>": mathml_mid_operations,
    "<left_operation>": mathml_left_operations,

    "<term>":
        [
            "<factor>",
            ("<left_operation><factor>", opts(prob=0.2)),
            "<left_parenthesis><mo>&#x2D;</mo><factor><right_parenthesis>",
        ],

    # "<factor_brackets>": mathml_factor_brackets,

    "<factor>":
        [
            ("<base>", opts(prob=0.4)),
            ("<function>", opts(prob=0.4)),
            "<subexpression>",
        ],

    "<base>":
        [
            "<mn><number></mn>",
            "<mi><identifier></mi>",
        ],

    "<subexpression>": ["<mrow><left_parenthesis><expression><right_parenthesis></mrow>"],

    "<function>":
        [
            "<mrow><msqrt><mrow><expression></mrow></msqrt></mrow>",
            "<mrow><mroot><mrow><expression></mrow><mrow><expression></mrow></mroot></mrow>",
            "<mrow><msub><base><base></msub></mrow>",
            "<mrow><msup><base><base></msup></mrow>",
            "<mrow><msubsup><mo>&#x222B;</mo><base><base></msubsup><expression><mi>d</mi><mi><identifier></mi></mrow>",  # integral
            "<mrow><msup><mi>e</mi><mrow><expression></mrow></msup></mrow>",  # exponent
            "<mrow><munderover><mo>&#x2211;</mo><mrow><mi><identifier></mi><mo>&#x3d;</mo><mn> 1 </mn></mrow><mi><identifier></mi></munderover></mrow>",  # sum
            "<mrow><mo stretchy=\"false\">&#x7c;</mo><mi><identifier></mi><mo stretchy=\"false\">&#x7c;</mo></mrow>",  # absolute value
            "<mrow><mo lspace=\"0em\" rspace=\"thinmathspace\">ln</mo><left_parenthesis><expression><right_parenthesis></mrow>",  # logarithm
            "<mrow><fraction></mrow>",
            "<mrow><trigonometric></mrow>",
            "<mrow><inverse_trigonometric></mrow>",
            "<mrow><hyperbolic></mrow>",
            "<mrow><inverse_hyperbolic></mrow>",
            "<mrow><limit></mrow>",
        ],


    "<fraction>":
        [
            "<mfrac><mrow><expression></mrow><mrow><expression></mrow></mfrac>",
            # "<left_parenthesis><mfrac_alt><base><base></mfrac><right_parenthesis>"  # Not supported by Tex2ASCIIMath
        ],

    "<limit>": ["<munder><mo lspace=\"0em\" rspace=\"0em\">lim</mo><mrow><mi><identifier></mi><mo stretchy=\"false\">&#x2192;</mo><limit_argument></mrow></munder><mrow><left_parenthesis><function><right_parenthesis></mrow>"],
    "<limit_argument>": ["<mn>&#x221e;</mn>", "<mn>0</mn>", "<mo>-</mo><mn>&#x221e;</mn>"],

    "<trigonometric>":
        [
            "<mi>sin</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>cos</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>tan</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>sec</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>csc</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>cot</mi><left_parenthesis><expression><right_parenthesis>",
        ],

    "<inverse_trigonometric>":
        [
            "<mi>arcsin</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arccos</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arctan</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arcsec</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arccsc</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arccot</mi><left_parenthesis><expression><right_parenthesis>",
        ],

    "<hyperbolic>":
        [
            "<mi>sinh</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>cosh</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>tanh</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>sech</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>csch</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>coth</mi><left_parenthesis><expression><right_parenthesis>",
        ],

    "<inverse_hyperbolic>":
        [
            "<mi>arcsinh</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arccosh</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arctanh</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arcsech</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arccsch</mi><left_parenthesis><expression><right_parenthesis>",
            "<mi>arccoth</mi><left_parenthesis><expression><right_parenthesis>",
        ],

    "<math>": ["<left_bracket>math<right_bracket>"],
    "</math>": ["<left_bracket>/math<right_bracket>"],

    "<mrow>": ["<left_bracket>mrow<right_bracket>"],
    "</mrow>": ["<left_bracket>/mrow<right_bracket>"],

    "<msub>": ["<left_bracket>msub<right_bracket>"],
    "</msub>": ["<left_bracket>/msub<right_bracket>"],

    "<msup>": ["<left_bracket>msup<right_bracket>"],
    "</msup>": ["<left_bracket>/msup<right_bracket>"],

    "<msubsup>": ["<left_bracket>msubsup<right_bracket>"],
    "</msubsup>": ["<left_bracket>/msubsup<right_bracket>"],

    "<msqrt>": ["<left_bracket>msqrt<right_bracket>"],
    "</msqrt>": ["<left_bracket>/msqrt<right_bracket>"],

    "<mroot>": ["<left_bracket>mroot<right_bracket>"],
    "</mroot>": ["<left_bracket>/mroot<right_bracket>"],

    "<munder>": ["<left_bracket>munder<right_bracket>"],
    "</munder>": ["<left_bracket>/munder<right_bracket>"],

    "<munderover>": ["<left_bracket>munderover<right_bracket>"],
    "</munderover>": ["<left_bracket>/munderover<right_bracket>"],

    "<mfrac>": ["<left_bracket>mfrac<right_bracket>"],
    "</mfrac>": ["<left_bracket>/mfrac<right_bracket>"],
    # "<mfrac_alt>": ["<left_bracket>mfrac linethickness=\"0\"<right_bracket>"],        # Not supported by Tex2ASCIIMath

    "<mo>": ["<left_bracket>mo<right_bracket>"],
    "</mo>": ["<left_bracket>/mo<right_bracket>"],

    "<mi>": ["<left_bracket>mi<right_bracket>"],
    "</mi>": ["<left_bracket>/mi<right_bracket>"],

    "<mn>": ["<left_bracket>mn<right_bracket>"],
    "</mn>": ["<left_bracket>/mn<right_bracket>"],

    "<left_parenthesis>": ["<mo>&#x28;</mo>"],
    "<right_parenthesis>": ["<mo>&#x29;</mo>"],

    "<left_bracket>": ["<"],
    "<right_bracket>": [">"],

    "<number>": ["NUMBER"],
    "<identifier>": ["IDENTIFIER"],
}

MATHML_MARKERS = [__MATHML_EBNF_GRAMMAR__["<number>"], __MATHML_EBNF_GRAMMAR__["<identifier>"]]

__MATHML_REFINED_EBNF_GRAMMAR__ = __MATHML_EBNF_GRAMMAR__.copy()
__MATHML_REFINED_EBNF_GRAMMAR__.update({
    "<number>": ["<integer>", "<float>"],
    "<integer>": ["<digit>+"],
    "<float>": ["<digit>+.<digit>+"],
    "<identifier>": ["<letter><digit>?"],
    "<letter>": letters,
    "<digit>": digits,
})

__PYTHON_EXPRESSION_GRAMMAR__: Grammar = {
    "<start>": ["<expression>"],
    "<expression>": ["<factor><operation><expression>", ("<factor>", opts(prob=0.4))],
    "<operation>": [" + ", " - ", " * ", " / ", " ** ", " // ", " % "],
    "<factor>": ["<number>", ("<function>", opts(prob=0.4))],
    "<function>": ["<base_func>", "<power_log_func>", "<trigonometric_func>",
                   "<angular_func>", "<hyperbolic_func>", ("<special_func>", opts(prob=0.1))],

    "<base_func>":
        [
            "math.ceil( <expression> )",
            "math.comb( <integer> , <integer> )",
            "math.copysign( <expression> , <expression> )",
            "math.fabs( <expression> )",
            "math.factorial( <integer> )",
            "math.floor( <expression> )",
            "math.fmod( <expression> , <expression> )",
            "math.fsum( <iterable> )",
            "math.gcd( <integer_sequence> )",
            "math.isqrt( <integer> )",
            "math.lcm( <integer_sequence> )",
            "math.ldexp( <number> , <integer> )",
            "math.nextafter( <expression> , <expression> )",
            "math.perm( <integer> , <integer> )",
            "math.prod( <iterable> )",
            "math.remainder( <expression> , <expression> )",
            "math.trunc( <expression> )",
        ],

    "<power_log_func>":
        [
            "math.exp( <expression> )",
            "math.expm1( <expression> )",
            "math.log( <expression> )",
            "math.log( <expression> , <expression> )",
            "math.log1p( <expression> )",
            "math.log2( <expression> )",
            "math.log10( <expression> )",
            "math.pow( <expression> , <expression> )",
            "math.sqrt( <expression> )",
        ],

    "<trigonometric_func>":
        [
            "math.acos( <small_float> )",
            "math.asin( <small_float> )",
            "math.atan( <expression> )",
            "math.atan2( <expression> , <expression>)",
            "math.cos( <expression> )",
            "math.sin( <expression> )",
            "math.tan( <expression> )",
            "math.hypot( <float_sequence> )",
            "math.dist( <iterable> , <iterable> )",
        ],

    "<angular_func>":
        [
            "math.degrees( <expression> )",
            "math.radians( <expression> )",
        ],

    "<hyperbolic_func>":
        [
            "math.acosh( <expression> )",
            "math.asinh( <expression> )",
            "math.atanh( <expression> )",
            "math.cosh( <expression> )",
            "math.sinh( <expression> )",
            "math.tanh( <expression> )",
        ],

    "<special_func>":
        [
            "math.erf( <expression> )",
            "math.erfc( <expression> )",
            "math.gamma( <expression> )",
            "math.lgamma( <expression> )",
        ],

    "<iterable>": ["[ <number> , <iterable-extension>*]"],
    "<iterable-extension>": ["<number> , "],

    "<integer_sequence>": ["<integer_number>", "<integer_number> , <integer_sequence_extension>+"],
    "<integer_sequence_extension>": ["<integer_number> , "],

    "<float_sequence>": ["<float_number>", "<float_number> , <float_sequence_extension>+"],
    "<float_sequence_extension>": ["<float_number> , "],

    "<number>":
    [
        "<integer_number>",
        "<float_number>",
        # ("<constant>", opts(prob=0.1))
    ],

    # "<constant>": ["math.pi", "math.e", "math.inf", "math.tau", "math.nan"],
    "<integer_number>": ["<integer>", "( - <integer> )"],
    "<float_number>": ["<float>", "( - <float> )"],
    "<small_float>": ["<float>"],

    # Placeholders
    "<integer>": ["INTEGER"],
    "<float>": ["FLOAT"],
}

PYTHON_EXPRESSION_MARKERS = [__PYTHON_EXPRESSION_GRAMMAR__["<integer>"], __PYTHON_EXPRESSION_GRAMMAR__["<float>"]]

__PYTHON_EXPRESSION_REFINED_GRAMMAR__ = __PYTHON_EXPRESSION_GRAMMAR__.copy()
__PYTHON_EXPRESSION_REFINED_GRAMMAR__.update({
    "<integer>": ["<digit>*"],
    "<float>": ["<non_zero_digit><digit>*.<digit>+", "<small_float>"],
    "<small_float>": ["0.<digit>+"],
    "<non_zero_digit>": digits[1:],
    "<digit>": digits,
})


MARKDOWN_GRAMMAR = convert_and_validate(__MARKDOWN_EBNF_GRAMMAR__)
LINQ_GRAMMAR = convert_and_validate(__LINQ_EBNF_GRAMMAR__)
SQL_GRAMMAR = convert_and_validate(__SQL_EBNF_BASE_GRAMMAR__)
SQL_REFINED_GRAMMAR = convert_and_validate(__SQL_REFINED_EBNF_BASE_GRAMMAR__)
SQL_EXTENDED_GRAMMAR = convert_and_validate(__SQL_EBNF_EXTENDED_GRAMMAR__)
SQL_EXTENDED_REFINED_GRAMMAR = convert_and_validate(__SQL_EBNF_EXTENDED_GRAMMAR__)
MATHML_GRAMMAR = convert_and_validate(__MATHML_EBNF_GRAMMAR__)
MATHML_REFINED_GRAMMAR = convert_and_validate(__MATHML_REFINED_EBNF_GRAMMAR__)
PYTHON_EXPRESSION_GRAMMAR = convert_and_validate(__PYTHON_EXPRESSION_GRAMMAR__)
PYTHON_EXPRESSION_REFINED_GRAMMAR = convert_and_validate(__PYTHON_EXPRESSION_REFINED_GRAMMAR__)


MARKER_MAPPING = {
    "markdown": MARKDOWN_MARKERS,
    "linq": LINQ_MARKERS,
    "sql": SQL_MARKERS,
    "mathml": MATHML_MARKERS,
    "expression": PYTHON_EXPRESSION_MARKERS,
}
