from pathlib import Path
from datetime import datetime

if __name__ == "__main__":
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()
from modelizer.tokenizer.generic import AbstractTokenizer
from modelizer.utils import pickle_load, pickle_dump, Logger, LoggingLevel


ROOT_DIR = "./data"
DEFAULT_SOURCE_PATH = "input.txt"
DEFAULT_TARGET_PATH = "output.txt"


class InputTokenizer(AbstractTokenizer):
    def __init__(self):
        super().__init__()

    def feed(self, data: str):
        # Implement data-specific tokenization routine
        raise NotImplementedError

    def reconstruct(self, tokens: list[str]) -> str:
        # Implement data-specific reconstruction routine
        raise NotImplementedError


class OutputTokenizer(AbstractTokenizer):
    def __init__(self):
        super().__init__()

    def feed(self, data: str):
        # Implement data-specific tokenization routine
        raise NotImplementedError

    def reconstruct(self, tokens: list[str]) -> str:
        # Implement data-specific reconstruction routine
        raise NotImplementedError


def tokenize(filepath: str, tokenizer: AbstractTokenizer) -> list[list]:
    # re-declare this method in the child class if a more complex logic is needed
    with open(filepath, "r") as file:
        result = [tokenizer.feed(line).copy() for line in file]
    return result


def parse(root_dir: str, source_filepath: str, target_filepath: str, source: str = "input", target: str = "output", logger=None) -> str:
    job_time = datetime.now()
    if logger is not None:
        logger.info(f"Task: Dataset Generation | Date and Time: {job_time.strftime('%Y-%m-%d %H:%M:%S')}")

    root_dir = Path(root_dir).resolve()

    dataset_filepath = root_dir.joinpath("dataset.pickle")
    loaded = pickle_load(dataset_filepath)
    dataset = {source: [], target: []} if loaded is None else loaded
    assert source in dataset, f"Dataset does not contain {source}"
    assert target in dataset, f"Dataset does not contain {target}"
    assert len(dataset[source]) == len(dataset[target]), "Source and Target collections sequences must have the same length"
    initial_records = len(dataset[source])

    input_tokenizer = InputTokenizer()
    output_tokenizer = OutputTokenizer()

    # Redefine the dataset population logic if you deal with more complex datastructures like Dataframes ...
    dataset[source].extend(tokenize(source_filepath, input_tokenizer))
    dataset[target].extend(tokenize(target_filepath, output_tokenizer))

    # Ensure that the source and target sequences have the same length
    assert len(dataset[source]) == len(dataset[target]), "Source and Target sequences must have the same length"

    # Save the dataset to a file
    pickle_dump(dataset, dataset_filepath)
    total_records = len(dataset[source])
    new_records = total_records - initial_records
    if logger is not None:
        message = "Dataset created" if total_records == new_records else f"Dataset updated\nNew records: {new_records}"
        logger.info(f"{message}\nTotal records: {total_records}")
        logger.info(f"Time Elapsed: {datetime.now() - job_time}")
    return dataset_filepath.as_posix()


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-s", "--source", type=str, default=DEFAULT_SOURCE_PATH, help="Path to the source file")
    arg_parser.add_argument("-t", "--target", type=str, default=DEFAULT_TARGET_PATH, help="Path to the source file")
    args = arg_parser.parse_args()
    parse(ROOT_DIR, args.source, args.target, logger=Logger(LoggingLevel.INFO))
