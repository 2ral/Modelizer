import pandoc

from pathlib import Path
from re import compile as re_compile

from tqdm import tqdm

from modelizer.utils import Multiprocessing, pickle_dump
from modelizer.generators.utils import PlaceholderProcessor
from modelizer.generators.grammars import MARKDOWN_MARKERS

html_placeholder_split_pattern = re_compile(r'TEXT_([0-9]+)TEXT_([0-9]+)')


class PandocParser:
    def __init__(self, root_dir: Path):
        self.data_dir = root_dir.joinpath("pickle_files")
        assert not self.data_dir.is_file(), "Path to pickle files directory already points to a file!"
        assert not self.data_dir.is_symlink(), "Path to pickle files directory already points to a symlink!"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def run(self, markdown_dir: Path, deduplicate: bool = False) -> Path:
        assert markdown_dir.is_dir(), "Path to markdown directory does not point to a directory!"
        dataset = {"markdown": [], "html": []}
        args = [(file, deduplicate) for file in markdown_dir.iterdir() if file.suffix == ".md"]
        results = Multiprocessing.parallel_run(self.parse_file, args, "Parsing Markdown...")
        for filename, md_str, html_str, mapping in tqdm(results, desc="Saving the Parsing Results..."):
            dataset["markdown"].append(md_str)
            dataset["html"].append(html_str)
            pickle_dump(mapping, self.data_dir.joinpath(filename + ".pickle"))
        pickle_dump(dataset, self.data_dir.parent.joinpath("dataset_raw.pickle"))
        return self.data_dir

    @staticmethod
    def parse_string(data: str) -> tuple[str, str]:
        md_doc = pandoc.read(data, format="markdown")
        md_str = pandoc.write(md_doc, format="markdown")
        md_str = PandocParser.remove_injected_markdown_attribute(md_str)
        md_str = md_str.replace("\\", "").replace(".  ", ".   ")
        html_str = pandoc.write(md_doc, format="html")
        html_str = PandocParser.remove_injected_html_attribute(html_str)
        html_str = html_str.replace("<a\n", "<a ").replace("<ahref", "<a href")
        for m in html_placeholder_split_pattern.findall(html_str):
            html_str = html_str.replace(f"TEXT_{m[0]}TEXT_{m[1]}", f"TEXT_{m[0]} TEXT_{m[1]}")
        return md_str, html_str

    @staticmethod
    def parse_file(args: tuple[Path, bool]) -> tuple[str, str, str, list[tuple[str, str]]]:
        mapping = PandocParser.traverse(args[0].read_text(), args[1])
        md_str = "\n\n".join([m[0] for m in mapping])
        html_str = "\n".join([m[1] for m in mapping])
        return args[0].stem, md_str, html_str, mapping

    @staticmethod
    # Traverse the Markdown file element by element, render each element to HTML and save MD-HTML pairs to a list
    def traverse(data: str, deduplicate: bool = False) -> list[tuple[str, str]]:
        processor = PlaceholderProcessor(MARKDOWN_MARKERS)
        doc = pandoc.read(data, format="markdown")
        mapping = []
        # doc[0] is the document metadata, doc[1] is the document body (list of elements)
        for e in doc[1]:
            new_doc = pandoc.types.Pandoc(doc[0], [e])
            new_markdown_str = pandoc.write(new_doc, format="markdown").replace("\\", "").replace(".  ", ".   ")
            new_markdown_str = PandocParser.remove_injected_markdown_attribute(new_markdown_str)
            if deduplicate:
                new_markdown_str = processor.deduplicate_placeholders(new_markdown_str)
                new_doc = pandoc.read(new_markdown_str, format="markdown")
            new_html_str = PandocParser.remove_injected_html_attribute(pandoc.write(new_doc, format="html"))
            new_html_str = new_html_str.replace("<a\n", "<a ").replace("<ahref", "<a href")
            for m in html_placeholder_split_pattern.findall(new_html_str):
                new_html_str = new_html_str.replace(f"TEXT_{m[0]}TEXT_{m[1]}", f"TEXT_{m[0]} TEXT_{m[1]}")
            mapping.append((new_markdown_str, new_html_str))
        return mapping

    @staticmethod
    def remove_injected_html_attribute(html_data: str) -> str:
        while html_data.find("id=") != -1:
            start_pos = html_data.find("id=") - 1
            end_pos = html_data.find(">", start_pos)
            html_data = html_data[:start_pos] + html_data[end_pos:]
        if "<h" in html_data:
            html_data = html_data.replace("\n", "", 1)
        return html_data

    @staticmethod
    def remove_injected_markdown_attribute(md_data: str) -> str:
        while md_data.find("{") != -1:
            start_pos = md_data.find("{") - 1
            end_pos = md_data.find("}", start_pos)
            md_data = md_data[:start_pos] + md_data[end_pos + 1:]
        return md_data


def markdown2html(data: str):
    _, html_str = PandocParser.parse_string(data)
    return html_str
