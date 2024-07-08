import requests

import modelizer.utils as utils

from time import sleep
from pathlib import Path
from hashlib import sha384
from datetime import datetime
from random import uniform, randint
from abc import ABC, abstractmethod
from shutil import make_archive, rmtree

from tqdm import tqdm

from modelizer.subjects.pandoc import PandocParser
from modelizer.generators.fuzzer import fuzzer_generator
from modelizer.generators.grammars import MARKDOWN_GRAMMAR
from modelizer.tokenizer.markup import MarkdownTokenizer, HTMLTokenizer


class MarkupFuzzer(ABC):
    def __init__(self, root_dir: Path, update_placeholders: bool = False):
        root_dir.mkdir(parents=True, exist_ok=True)
        self.logger = utils.Logger(utils.LoggingLevel.INFO, root_dir)
        self.parser = PandocParser(root_dir)
        self.data_directory = root_dir.joinpath("md_files")
        self.__hashes_fh__ = root_dir.joinpath("hashes.pickle")
        self.__dataset_fh__ = root_dir.joinpath("dataset.pickle")
        self.__dataset_fh_full__ = root_dir.joinpath("dataset_full.pickle")
        self.__update_placeholders__ = update_placeholders

    def fuzz(self, fuzzing_rounds: int = 1, max_nonterminals: int = 20, partition: bool = False) -> str:
        start_time = datetime.now()
        # Generating Markdown Files
        assert not self.data_directory.is_file(), f"Directory path {self.data_directory.as_posix()} exists, but points to a file"
        assert not self.data_directory.is_symlink(), f"Directory path {self.data_directory.as_posix()} exists, but points to a symlink"
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.__fuzz__(fuzzing_rounds, max_nonterminals, partition)

        # Parsing Markdown Files, generating HTML Files
        self.logger.info(f"Task: Parsing")
        job_start_time = datetime.now()
        parsed_files_dir = self.parser.run(self.data_directory, self.__update_placeholders__)
        self.logger.info(f"Parsing completed in {datetime.now() - job_start_time}")

        # Generating Tokens and Saving the Dataset
        job_time = datetime.now()
        loaded = utils.pickle_load(self.__dataset_fh__.as_posix())
        loaded_full = utils.pickle_load(self.__dataset_fh_full__.as_posix())
        dataset = {"markdown": [], "html": []} if loaded is None else loaded
        dataset_full = {"markdown": [], "html": []} if loaded_full is None else loaded_full
        assert "markdown" in dataset, "Dataset must contain 'markdown' collection"
        assert "html" in dataset, "Dataset must contain 'markdown' collection"
        assert len(dataset["markdown"]) == len(dataset["html"]), "Source and Target collections must have the same length"
        initial_records = len(dataset["markdown"])
        files = [file.as_posix() for file in parsed_files_dir.iterdir() if file.suffix == ".pickle"]
        self.logger.info(f"Task: Tokenization | Files to process: {len(files)}")
        result = utils.Multiprocessing.parallel_run(self.tokenize_markdown_html, files, "Tokenizing...")
        for md_r, html_r in result:
            dataset["markdown"].extend(md_r)
            dataset_full["markdown"].extend(md_r)
            dataset["html"].extend(html_r)
            dataset_full["html"].extend(html_r)
        assert len(dataset["markdown"]) == len(dataset["html"]), "Source and Target sequences must have the same length after tokenization"
        self.logger.info("Eliminating duplicates")
        len_before_elimination = len(dataset["markdown"])
        separator = "!sep!"
        dataset["markdown"] = [separator.join(entry) for entry in dataset["markdown"]]
        dataset["html"] = [separator.join(entry) for entry in dataset["html"]]
        dataset = [(md, html) for md, html in set(zip(dataset["markdown"], dataset["html"]))]
        dataset = {"markdown": [dataset[i][0].split(separator) for i in range(len(dataset))],
                   "html": [dataset[i][1].split(separator) for i in range(len(dataset))]}
        assert len(dataset["markdown"]) == len(dataset["html"]), "Source and Target sequences must have the same length after filtration"
        self.logger.info(f"Eliminated {len_before_elimination - len(dataset['markdown'])} duplicates")
        self.logger.info(f"Task: Saving the tokenization results")
        utils.pickle_dump(dataset, self.__dataset_fh__)
        utils.pickle_dump(dataset_full, self.__dataset_fh_full__)
        self.logger.info(f"Dataset saved at {self.__dataset_fh__.as_posix()}")
        self.logger.info("Deleting the temporary files")
        self.zip_and_delete(parsed_files_dir)
        self.zip_and_delete(self.data_directory)
        total_records = len(dataset["markdown"])
        new_records = total_records - initial_records
        message = f"Dataset Generated in {datetime.now() - job_time}" if total_records == new_records \
            else f"Dataset Updated in {datetime.now() - job_time} | New records: {new_records}"
        self.logger.info(f"{message} | Total records: {total_records}")

        self.logger.info(f"Overall Duration: {datetime.now() - start_time}")
        return self.__dataset_fh__.as_posix()

    def coverage_fuzz(self, fuzzing_rounds: int = 1) -> str:
        return self.fuzz(fuzzing_rounds, max_nonterminals=-1)

    @abstractmethod
    def __fuzz__(self, fuzzing_rounds: int = 1, max_nonterminals: int = 25, partition: bool = False):
        pass

    @staticmethod
    def zip_and_delete(handler: Path):
        time_padding = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        assert handler.exists(), f"Directory {handler.as_posix()} does not exist"
        make_archive(handler.with_suffix("").as_posix() + time_padding, "zip", handler.as_posix())
        if handler.is_dir():
            rmtree(handler, ignore_errors=True)
        else:
            handler.unlink()

    @staticmethod
    def tokenize_markdown_html(filepath: str) -> tuple[list, list]:
        md_list, html_list = [], []
        markdown_tokenizer = MarkdownTokenizer()
        html_tokenizer = HTMLTokenizer()
        for md, html in utils.pickle_load(filepath):
            md_list.append(markdown_tokenizer.feed(md))
            html_list.append(html_tokenizer.feed(html))
        return md_list, html_list


class MarkdownFuzzer(MarkupFuzzer):
    def __init__(self, root_dir: Path):
        super(MarkdownFuzzer, self).__init__(root_dir, True)

    def __fuzz__(self, fuzzing_rounds: int = 1, max_nonterminals: int = 20, partition: bool = False):
        job_start_time = datetime.now()
        self.logger.info(f"Task: Markdown Generation | Records to generate: {fuzzing_rounds}")
        if max_nonterminals < 0:
            min_nonterminals = max_nonterminals
            self.logger.info("Gathering formulas with GrammarCoverageFuzzer")
        else:
            self.logger.info("Gathering formulas with ProbabilisticGrammarFuzzer")
            min_nonterminals = 10 if max_nonterminals - 10 <= 0 else max_nonterminals - 10
            max_nonterminals = min_nonterminals + 10 if max_nonterminals < min_nonterminals else max_nonterminals
        initial_files_count = len([f for f in Path(self.data_directory).iterdir() if f.is_file()])
        fuzzing_params = [MARKDOWN_GRAMMAR, min_nonterminals, max_nonterminals, 0, 300]
        generated, _ = fuzzer_generator(fuzzing_rounds, fuzzing_params, self.__hashes_fh__, self.logger, partition)
        self.update_placeholders = True
        for doc in generated:
            doc = doc.replace("\n\n1.", "\n1.").replace("\n\n-", "\n-")
            filepath = self.data_directory.joinpath(datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".md")
            filepath.write_text(doc)

        written_files_count = len([f for f in self.data_directory.iterdir() if f.is_file()]) - initial_files_count
        assert written_files_count == fuzzing_rounds, f"Inconsistent number of files generated:\n" \
                                                      f"Requested: {fuzzing_rounds} | Written to disc: {written_files_count}"
        self.logger.info(f"MD Files generated in {datetime.now() - job_start_time}")


class MarkdownLoremIpsumFuzzer(MarkupFuzzer):
    def __fuzz__(self, fuzzing_rounds: int = 1, max_nonterminals: int = 4, partition: bool = False):
        job_start_time = datetime.now()
        self.logger.info(f"Task: Fetching Lorem-Ipsum Markdown")
        loaded = utils.pickle_load(self.__hashes_fh__.as_posix())
        hashes_dataset = set() if loaded is None else loaded
        initial_files_count = len([f for f in Path(self.data_directory).iterdir() if f.is_file()])

        for _ in tqdm(range(fuzzing_rounds), desc="Requesting..."):
            doc = self.fetch_lorem_markdown(max_blocks=max_nonterminals)
            hash_v = sha384(doc[0].encode("UTF-8")).hexdigest()
            while hash_v in hashes_dataset:
                doc = self.fetch_lorem_markdown(max_blocks=max_nonterminals)
                hash_v = sha384(doc[0].encode("UTF-8")).hexdigest()
            hashes_dataset.add(hash_v)
            filepath = self.data_directory.joinpath(datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".md")
            filepath.write_text(doc)

        written_files_count = len([f for f in self.data_directory.iterdir() if f.is_file()]) - initial_files_count
        assert written_files_count == fuzzing_rounds, f"Inconsistent number of files generated:\n" \
                                                      f"Requested: {fuzzing_rounds} | Written to disc: {written_files_count}"

        if utils.pickle_dump(hashes_dataset, self.__hashes_fh__):
            self.logger.error(f"Hashes dataset cannot be written to {self.__hashes_fh__.as_posix()}")

        self.logger.info(f"MD Files fetched in {datetime.now() - job_start_time}")
        return self.__dataset_fh__.as_posix()

    def coverage_fuzz(self, fuzzing_rounds: int = 1) -> str:
        raise NotImplementedError("Coverage fuzzing is not supported for Lorem-Ipsum Fuzzer")

    @staticmethod
    def fetch_lorem_markdown(url="https://jaspervdj.be/lorem-markdownum/markdown.txt",
                             min_blocks: int = 1, max_blocks: int = 4,
                             no_code: bool = False, no_inline_markup: bool = False,
                             no_lists: bool = False, no_quotes: bool = False) -> str:
        url += f"?num-blocks={randint(min_blocks, max_blocks)}"
        if no_code:
            url += "&no-code=on"
        if no_inline_markup:
            url += "&no-inline-markup=on"
        if no_lists:
            url += "&no-lists=on"
        if no_quotes:
            url += "&no-quotes=on"

        response = requests.get(url)
        min_wait_time = 0.1
        max_wait_time = 1
        while response.status_code != 200:
            sleep(uniform(min_wait_time, max_wait_time))
            min_wait_time *= 1.25
            max_wait_time *= 1.25
            if max_wait_time > 10:
                min_wait_time = 0.1
                max_wait_time = 1
            response = requests.get(url)
        return response.text


def fetch_lorem_markdown(url="https://jaspervdj.be/lorem-markdownum/markdown.txt",
                         min_blocks: int = 1, max_blocks: int = 4,
                         no_code: bool = False, no_inline_markup: bool = False,
                         no_lists: bool = False, no_quotes: bool = False) -> tuple[str, str]:
    response = MarkdownLoremIpsumFuzzer.fetch_lorem_markdown(url, min_blocks, max_blocks, no_code,
                                                             no_inline_markup, no_lists, no_quotes)
    return PandocParser.parse_string(response)
