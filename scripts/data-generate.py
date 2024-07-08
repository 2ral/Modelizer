DEFAULT_ITERATIONS_COUNT = 10000
DEFAULT_MAX_NONTERMINALS = 20


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from math import ceil
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()

    SUPPORTED_SUBJECTS = ["sql", "linq", "mathml", "mathml-refined", "markdown", "markdown-lorem-ipsum", "expression"]

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--fuzz', type=str, default="markdown", help='Fuzzer Selection')
    arg_parser.add_argument('--iter', type=int, default=DEFAULT_ITERATIONS_COUNT, help='Number of Samples to Generate')
    arg_parser.add_argument('--max', type=int, default=DEFAULT_MAX_NONTERMINALS, help='Maximum Number of Nonterminals/Blocks')
    arg_parser.add_argument('--part', action='store_true', help='Partition synthesized dataset according to the number of Nonterminals/Blocks')
    arg_parser.add_argument('--hash', action='store_true', help='Preload hashes from existing datasets')
    arg_parser.add_argument('--extra', action='store_true', help='Generate extra samples for the validation set')

    args = arg_parser.parse_args()

    match args.fuzz.lower():
        case "sql":
            from modelizer.generators.query import SQLFuzzer
            fuzzer_factory = SQLFuzzer
        case "linq":
            from modelizer.generators.query import LINQFuzzer
            fuzzer_factory = LINQFuzzer
        case "mathml":
            from modelizer.generators.mathml import MathMLFuzzer
            fuzzer_factory = MathMLFuzzer
        case "mathml-refined":
            from modelizer.generators.mathml import MathMLRefinedFuzzer
            fuzzer_factory = MathMLRefinedFuzzer
        case "markdown":
            from modelizer.generators.markup import MarkdownFuzzer
            fuzzer_factory = MarkdownFuzzer
        case "markdown-lorem-ipsum":
            from modelizer.generators.markup import MarkdownLoremIpsumFuzzer
            fuzzer_factory = MarkdownLoremIpsumFuzzer
        case "expression":
            from modelizer.generators.expression import PythonExpressionFuzzer
            fuzzer_factory = PythonExpressionFuzzer
        case _:
            raise ValueError(f"Unsupported fuzzer {args.fuzz}, supported subjects are {SUPPORTED_SUBJECTS}")

    # Initializing the ROOT Directory
    root_dir = Path("data").resolve().joinpath(args.fuzz.lower())
    assert not root_dir.is_file(), f"Root directory path {root_dir.as_posix()} exists and points to a file"
    assert not root_dir.is_symlink(), f"Root directory path {root_dir.as_posix()} exists and points to a symlink"
    root_dir.mkdir(parents=True, exist_ok=True)

    # Preload and merge hashes from existing datasets
    if args.hash:
        from modelizer.utils import pickle_load, pickle_dump
        combined_set = set()
        for p in Path("datasets/train").glob("data_*"):
            loaded_set = pickle_load(p.joinpath(args.fuzz.lower()).joinpath("hashes.pickle"))
            if loaded_set is not None and isinstance(loaded_set, set):
                combined_set.update(loaded_set)
        pickle_dump(combined_set, root_dir.joinpath("hashes.pickle"))

    fuzzer = fuzzer_factory(root_dir)
    fuzzer.fuzz(ceil(args.iter * 1.25) if args.extra else args.iter, args.max, args.part)
