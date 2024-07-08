from pathlib import Path
from itertools import chain, repeat

from func_timeout import func_timeout

from modelizer.utils import Multiprocessing
from modelizer.subjects.mstic import sql2kql
from modelizer.generators.fuzzer import AbstractFuzzer
from modelizer.generators.utils import PlaceholderProcessor
from modelizer.tokenizer.query import SQLTokenizer, KQLTokenizer
from modelizer.generators.grammars import SQL_GRAMMAR, SQL_MARKERS, LINQ_GRAMMAR, LINQ_MARKERS


def sql2kql_safe(query: str) -> str | None:
    try:
        # trick to terminate buggy conversion
        kql = func_timeout(5, sql2kql, args=(query,))
        # filter out invalid conversions
        if len(kql) == 0 or kql[0] == "|":
            raise ValueError
    except:
        kql = None
    return kql


class SQLFuzzer(AbstractFuzzer):
    def __init__(self, root_dir: Path):
        super().__init__("SQL", root_dir, SQL_GRAMMAR, SQL_MARKERS, ["sql", "kql"])

    def update_placeholders(self, subjects: list[str]) -> list[str]:
        if self.markers is None:
            return subjects
        else:
            subjects = list(Multiprocessing.chunk_generator(subjects))
            markers = list(repeat(self.markers, len(subjects)))
            params = list(zip(subjects, markers))
            return list(chain(*Multiprocessing.parallel_run(self.__postprocessing2__, params, text="Updating placeholders")))

    @staticmethod
    def __postprocessing2__(params: tuple[list[str], list[str]]) -> list[str]:
        def __find_id_value__(d, search_pos: int, inverse: bool = False, terminal_symbol: str = " ") -> tuple[str, int, int]:
            search_space = d[:search_pos] if inverse else d[search_pos:]
            table_pos = search_space.find("TABLE")
            id_start_pos = table_pos + 6
            id_end_pos = search_space.find(terminal_symbol, table_pos + 7)
            id_value = search_space[id_start_pos:id_end_pos]
            if not inverse:
                id_start_pos += search_pos
                id_end_pos += search_pos
            return id_value, id_start_pos, id_end_pos

        subjects, markers = params
        processor = PlaceholderProcessor(markers)
        results = []
        for data in [processor.deduplicate_placeholders(subject) for subject in subjects]:
            # Align table names in JOIN conditions
            join_pos = data.find("JOIN")
            while join_pos != -1:
                from_id = __find_id_value__(data, join_pos, inverse=True)
                target_id = __find_id_value__(data, join_pos)
                join_left = __find_id_value__(data, target_id[2], terminal_symbol=".", )
                join_right = __find_id_value__(data, join_left[2], terminal_symbol=".", )
                # update left join
                data = data[:join_left[1]] + from_id[0] + data[join_left[2]:]
                # update right join
                data = data[:join_right[1]] + target_id[0] + data[join_right[2]:]
                # check for other joins
                join_pos = data.find("JOIN", join_pos + 1)
            results.append(data)
        return results

    @staticmethod
    def __convert__(subjects: list[str]) -> list[tuple]:
        results = []
        for q in subjects:
            kql = sql2kql_safe(q)
            if kql is None:
                continue
            else:
                results.append((q, kql))
        return results

    @staticmethod
    def __tokenize__(subjects: list[tuple]) -> list[tuple]:
        sql_tokenizer = SQLTokenizer()
        kql_tokenizer = KQLTokenizer()
        return [(sql_tokenizer.feed(record[0]), kql_tokenizer.feed(record[1])) for record in subjects]


class LINQFuzzer(AbstractFuzzer):
    def __init__(self, root_dir: Path):
        super().__init__("LINQ", root_dir, LINQ_GRAMMAR, LINQ_MARKERS, ["linq", "sql"])

    @staticmethod
    def __convert__(subjects: list[str]) -> list[tuple]:
        raise NotImplementedError

    @staticmethod
    def __tokenize__(subjects: list[tuple]):
        raise NotImplementedError
